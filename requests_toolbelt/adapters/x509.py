# -*- coding: utf-8 -*-
"""A X509Adapter for use with the requests library.

This file contains an implementation of the X509Adapter that allows users to
authenticate a request using an arbitrary X.509 certificate and private key
supplied directly from memory without persisting them as ``.pem`` files.

The adapter builds standard library :class:`ssl.SSLContext` objects (the only
kind of context supported by urllib3 2.x, which removed its pyOpenSSL
integration) and hands them to requests/urllib3 through the ``ssl_context``
pool keyword. The public interface still accepts PEM/DER byte arrays in
memory. A fresh context is created for every connection pool so that
per-request ``verify`` settings cannot leak between requests: each context is
configured with exactly the trust bundle of the request that owns the pool.
The brief materialization required by OpenSSL's ``load_cert_chain`` API
happens in a restricted temporary file that is removed immediately
afterwards.
"""

import os
import ssl
import tempfile
from datetime import datetime, timezone

import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_der_private_key,
    load_pem_private_key,
)
from requests.adapters import HTTPAdapter

from .. import exceptions as exc

try:
    from urllib3.util.ssl_ import create_urllib3_context
except ImportError:  # pragma: no cover - urllib3 always ships this today
    create_urllib3_context = None


class X509Adapter(HTTPAdapter):
    r"""Adapter for use with X.509 client certificates.

    Provides an interface for Requests sessions to contact HTTPS urls and
    authenticate with an X.509 cert by implementing the Transport Adapter
    interface. This class will need to be manually instantiated and mounted
    to the session.

    The certificate and private key are provided as in-memory byte arrays,
    either PEM or DER encoded. Encrypted private keys are supported through
    the ``password`` parameter. Server certificate verification is not
    altered by this adapter: the session's ``verify`` setting and any
    per-request ``verify`` override keep working, both for direct
    connections and when tunnelling through a proxy.

    :param pool_connections: The number of urllib3 connection pools to
           cache.
    :param pool_maxsize: The maximum number of connections to save in the
            pool.
    :param max_retries: The maximum number of retries each connection
        should attempt. Note, this applies only to failed DNS lookups,
        socket connections and connection timeouts, never to requests where
        data has made it to the server. By default, Requests does not retry
        failed connections. If you need granular control over the
        conditions under which we retry, import urllib3's ``Retry`` class
        and pass that instead.
    :param pool_block: Whether the connection pool should block for
            connections.

    :param bytes cert_bytes:
        bytes object containing the contents of an X.509 certificate using
        the encoding specified by the ``encoding`` parameter.
    :param bytes pk_bytes:
        bytes object containing the contents of a private key using the
        encoding specified by the ``encoding`` parameter.
    :param password:
        string or utf8 encoded bytes containing the passphrase used for the
        private key. None if unencrypted. Defaults to None.
    :param encoding:
        Enumeration detailing the encoding method used on the ``cert_bytes``
        and ``pk_bytes`` parameters. Can be either PEM or DER.
        Defaults to PEM.
    :type encoding:
        :class:`cryptography.hazmat.primitives.serialization.Encoding`

    Usage::

      >>> import requests
      >>> from requests_toolbelt.adapters.x509 import X509Adapter
      >>> s = requests.Session()
      >>> a = X509Adapter(max_retries=3,
      ...                 cert_bytes=b'...', pk_bytes=b'...')
      >>> s.mount('https://', a)
    """

    __attrs__ = HTTPAdapter.__attrs__ + [
        '_cert_pem', '_key_pem', '_password']

    def __init__(self, *args, **kwargs):
        self._check_version()
        cert_bytes = kwargs.pop('cert_bytes', None)
        pk_bytes = kwargs.pop('pk_bytes', None)
        password = kwargs.pop('password', None)
        encoding = kwargs.pop('encoding', Encoding.PEM)

        if cert_bytes is None or not isinstance(cert_bytes, bytes):
            raise ValueError('Invalid cert content provided. '
                             'You must provide an X.509 cert '
                             'formatted as a byte array.')
        if pk_bytes is None or not isinstance(pk_bytes, bytes):
            raise ValueError('Invalid private key content provided. '
                             'You must provide a private key '
                             'formatted as a byte array.')

        if isinstance(password, bytes):
            password_bytes = password
        elif password:
            password_bytes = password.encode('utf8')
        else:
            password_bytes = None

        # Validate and normalize the material once, at construction time,
        # so bad passphrases and mismatched keys fail before any request.
        self._cert_pem, self._key_pem = _load_and_normalize(
            cert_bytes, pk_bytes, password_bytes, encoding)
        self._password = password_bytes

        super(X509Adapter, self).__init__(*args, **kwargs)

    def _new_ssl_context(self):
        """Build a fresh SSLContext carrying the client cert/key.

        A new context is returned for every connection pool. This keeps
        trust material and ``verify_mode`` scoped to a single request
        configuration instead of mutating one shared context.
        """
        context = create_urllib3_context()
        # urllib3 re-applies verify_mode on every connection from the
        # request's verify argument and performs hostname matching itself
        # when check_hostname is False. Leaving it True would make
        # verify=False raise ("check_hostname needs a SSL context with
        # CERT_REQUIRED"). Hostname verification is still enforced whenever
        # the peer certificate is verified; it is simply owned by urllib3
        # instead of OpenSSL.
        context.check_hostname = False
        _load_pem_chain_into_context(context, self._cert_pem,
                                     self._key_pem, self._password)
        return context

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self._new_ssl_context()
        return super(X509Adapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs['ssl_context'] = self._new_ssl_context()
        return super(X509Adapter, self).proxy_manager_for(*args, **kwargs)

    def build_connection_pool_key_attributes(self, request, verify,
                                             cert=None):
        host_params, pool_kwargs = super(
            X509Adapter, self).build_connection_pool_key_attributes(
                request, verify, cert)
        # Give this exact verify configuration its own context. urllib3 keys
        # pools on the ssl_context object, so distinct verify values land in
        # distinct pools and never share accumulated CAs or verify state.
        pool_kwargs['ssl_context'] = self._new_ssl_context()
        return host_params, pool_kwargs

    def _check_version(self):
        if (requests.__build__ < 0x021200 or
                create_urllib3_context is None or
                not hasattr(ssl.SSLContext, 'load_cert_chain')):
            raise exc.VersionMismatchError(
                "The X509Adapter requires at least Requests 2.12.0 with a "
                "urllib3 version that supports the standard library "
                "ssl.SSLContext (urllib3 1.26+ or 2.x). Requests version "
                "{} was found instead.".format(requests.__version__)
            )


def create_ssl_context(cert_byes, pk_bytes, password=None,
                       encoding=Encoding.PEM):
    """Create an SSL Context with the supplied cert/password.

    :param bytes cert_byes: bytes containing the cert encoded using the
           method supplied in the ``encoding`` parameter
    :param bytes pk_bytes: bytes containing the private key encoded using
           the method supplied in the ``encoding`` parameter
    :param bytes password: passphrase for the supplied private key.
           None if unencrypted. Defaults to None.
    :param encoding:
        :class:`cryptography.hazmat.primitives.serialization.Encoding`
        detailing the encoding method used on the ``cert_byes`` and
        ``pk_bytes`` parameters. Can be either PEM or DER.
        Defaults to PEM.

    A wrong passphrase or a certificate/key pair that does not match fails
    here, while the context is being built, rather than on the first
    request.
    """
    cert_pem, key_pem = _load_and_normalize(cert_byes, pk_bytes, password,
                                            encoding)
    context = create_urllib3_context()
    context.check_hostname = False
    _load_pem_chain_into_context(context, cert_pem, key_pem, password)
    return context


def check_cert_dates(cert):
    """Verify that the supplied client cert is currently valid."""

    now = datetime.now(timezone.utc)
    not_after = getattr(cert, 'not_valid_after_utc', None)
    not_before = getattr(cert, 'not_valid_before_utc', None)
    if not_after is None:  # pragma: no cover - old cryptography versions
        not_after = cert.not_valid_after.replace(tzinfo=timezone.utc)
        not_before = cert.not_valid_before.replace(tzinfo=timezone.utc)
    if not_after < now or not_before > now:
        raise ValueError('Client certificate expired: Not After: '
                         '{:%Y-%m-%d %H:%M:%SZ} '
                         'Not Before: {:%Y-%m-%d %H:%M:%SZ}'
                         .format(not_after, not_before))


def _load_and_normalize(cert_bytes, pk_bytes, password, encoding):
    """Parse PEM/DER input and return PEM cert and key byte strings.

    The private key is decrypted here purely to validate the passphrase and
    the cert/key relationship. When a passphrase is supplied the returned
    key PEM is re-encrypted with that same passphrase, so the decrypted key
    never has to be written to disk later.
    """
    backend = default_backend()

    if encoding == Encoding.PEM:
        cert = x509.load_pem_x509_certificate(cert_bytes, backend)
        key = load_pem_private_key(pk_bytes, password, backend)
    elif encoding == Encoding.DER:
        cert = x509.load_der_x509_certificate(cert_bytes, backend)
        key = load_der_private_key(pk_bytes, password, backend)
    else:
        raise ValueError('Invalid encoding provided: Must be PEM or DER')

    check_cert_dates(cert)

    cert_pem = cert.public_bytes(Encoding.PEM)
    if password:
        key_pem = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8,
            BestAvailableEncryption(password))
    else:
        key_pem = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8,
                                    NoEncryption())

    # Prove the pair matches through the same OpenSSL path used at request
    # time. The temporary file is removed before returning, and error
    # messages from OpenSSL reference file paths rather than key contents.
    probe_context = create_urllib3_context()
    probe_context.check_hostname = False
    _load_pem_chain_into_context(probe_context, cert_pem, key_pem,
                                 password)
    return cert_pem, key_pem


def _load_pem_chain_into_context(context, cert_pem, key_pem, password):
    """Load a PEM cert/key pair into ``context`` via a private temp file.

    OpenSSL's :meth:`ssl.SSLContext.load_cert_chain` only accepts file
    paths, so the PEM material is written once to a 0600 temporary file and
    removed before this function returns. When the key is encrypted the file
    only ever contains the encrypted PEM; the passphrase itself is never
    written.
    """
    tmp = tempfile.NamedTemporaryFile(mode='wb', suffix='.pem',
                                      delete=False)
    path = tmp.name
    try:
        tmp.write(cert_pem)
        tmp.write(b'\n')
        tmp.write(key_pem)
        tmp.flush()
        tmp.close()
        os.chmod(path, 0o600)
        context.load_cert_chain(path, password=password)
    finally:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.unlink(path)
        except OSError:
            pass
