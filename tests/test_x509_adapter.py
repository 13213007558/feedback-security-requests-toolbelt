# -*- coding: utf-8 -*-
"""Tests for the in-memory X509 client certificate adapter.

The TLS peers here are threads driving socketpair endpoints with material
generated per-test by ``trustme``. Nothing in this module or its cassettes
contains a real private key.
"""
import importlib
import inspect
import os
import ssl
import tempfile
from unittest import mock

import pytest
import requests
import requests.exceptions as requests_exc
import trustme
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    Encoding,
    PrivateFormat,
    load_pem_private_key,
)
from urllib3 import ProxyManager
from urllib3.exceptions import MaxRetryError

from requests_toolbelt.adapters.x509 import (
    X509Adapter,
    check_cert_dates,
    create_ssl_context,
)
from .tls_helpers import (
    http_connect_proxy,
    https_connect_proxy,
    run_tls_target,
    server_ssl_context,
    socketpair_route,
)

REQUESTS_SUPPORTS_SSL_CONTEXT = requests.__build__ >= 0x021200
TARGET_HOST = 'target.example.test'
PROXY_HOST = 'proxy.example.test'

pytestmark = pytest.mark.skipif(
    not REQUESTS_SUPPORTS_SSL_CONTEXT,
    reason='X509Adapter requires Requests 2.12.0+ with stdlib ssl context',
)


@pytest.fixture
def ca():
    return trustme.CA()


@pytest.fixture
def client_material(ca):
    cert = ca.issue_cert('client.example.test')
    return cert.cert_chain_pems[0].bytes(), cert.private_key_pem.bytes()


@pytest.fixture
def ca_bundle_file(ca):
    handle = tempfile.NamedTemporaryFile(suffix='.pem', delete=False)
    handle.write(ca.cert_pem.bytes())
    handle.close()
    yield handle.name
    os.unlink(handle.name)


@pytest.fixture
def foreign_ca_file():
    other = trustme.CA()
    handle = tempfile.NamedTemporaryFile(suffix='.pem', delete=False)
    handle.write(other.cert_pem.bytes())
    handle.close()
    yield handle.name
    os.unlink(handle.name)


def make_adapter(cert_bytes, key_bytes, **kwargs):
    adapter = X509Adapter(cert_bytes=cert_bytes, pk_bytes=key_bytes, **kwargs)
    session = requests.Session()
    session.trust_env = False
    session.mount('https://', adapter)
    return session


def target_context(ca, host=TARGET_HOST, require_client_cert=True):
    server_cert = ca.issue_cert(host)
    return server_ssl_context(
        server_cert,
        client_ca_pem=ca.cert_pem.bytes().decode('ascii')
        if require_client_cert else None)


class TestConstruction(object):
    def test_pem_unencrypted(self, client_material):
        cert_bytes, key_bytes = client_material
        adapter = X509Adapter(cert_bytes=cert_bytes, pk_bytes=key_bytes)
        assert isinstance(adapter._new_ssl_context(), ssl.SSLContext)

    def test_der_with_encrypted_private_key(self, ca):
        cert = ca.issue_cert('client.example.test')
        password = b'passphrase'
        der_cert = x509.load_pem_x509_certificate(
            cert.cert_chain_pems[0].bytes()).public_bytes(Encoding.DER)
        key = load_pem_private_key(cert.private_key_pem.bytes(), password=None)
        der_key = key.private_bytes(
            Encoding.DER, PrivateFormat.PKCS8,
            BestAvailableEncryption(password))
        adapter = X509Adapter(cert_bytes=der_cert, pk_bytes=der_key,
                              password=password, encoding=Encoding.DER)
        assert isinstance(adapter._new_ssl_context(), ssl.SSLContext)

    def test_password_as_string(self, client_material):
        cert_bytes, key_bytes = client_material
        key = load_pem_private_key(key_bytes, password=None)
        encrypted = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8,
            BestAvailableEncryption(b'unicode-pw'))
        adapter = X509Adapter(cert_bytes=cert_bytes, pk_bytes=encrypted,
                              password='unicode-pw')
        assert isinstance(adapter._new_ssl_context(), ssl.SSLContext)

    def test_wrong_password_fails_at_construction(self, ca):
        cert = ca.issue_cert('client.example.test')
        key = load_pem_private_key(cert.private_key_pem.bytes(), password=None)
        encrypted = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8,
            BestAvailableEncryption(b'correct'))
        with pytest.raises(ValueError):
            X509Adapter(cert_bytes=cert.cert_chain_pems[0].bytes(),
                        pk_bytes=encrypted, password=b'incorrect')

    def test_cert_key_mismatch_fails_at_construction(self, ca):
        first = ca.issue_cert('first.example.test')
        second = ca.issue_cert('second.example.test')
        with pytest.raises(ssl.SSLError):
            X509Adapter(
                cert_bytes=first.cert_chain_pems[0].bytes(),
                pk_bytes=second.private_key_pem.bytes())

    def test_invalid_encoding(self, client_material):
        cert_bytes, key_bytes = client_material
        with pytest.raises(ValueError):
            create_ssl_context(cert_bytes, key_bytes, encoding='not-real')

    @pytest.mark.parametrize('kwargs', [
        {'cert_bytes': None, 'pk_bytes': b'x'},
        {'cert_bytes': b'x', 'pk_bytes': None},
        {'cert_bytes': 'not-bytes', 'pk_bytes': b'x'},
    ])
    def test_invalid_material_types(self, kwargs):
        with pytest.raises(ValueError):
            X509Adapter(**kwargs)

    def test_expired_cert_rejected(self):
        import datetime
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'expired')])
        expired = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime(2010, 1, 1,
                                                tzinfo=datetime.timezone.utc))
            .not_valid_after(datetime.datetime(2011, 1, 1,
                                               tzinfo=datetime.timezone.utc))
            .sign(key, hashes.SHA256()))
        with pytest.raises(ValueError):
            X509Adapter(
                cert_bytes=expired.public_bytes(Encoding.PEM),
                pk_bytes=key.private_bytes(
                    Encoding.PEM, PrivateFormat.PKCS8,
                    serialization.NoEncryption()))

    def test_no_private_key_material_in_error_text(self, ca):
        cert = ca.issue_cert('client.example.test')
        key = load_pem_private_key(cert.private_key_pem.bytes(), password=None)
        encrypted = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8,
            BestAvailableEncryption(b'correct'))
        try:
            X509Adapter(cert_bytes=cert.cert_chain_pems[0].bytes(),
                        pk_bytes=encrypted, password=b'incorrect')
        except ValueError as err:
            message = str(err)
        else:  # pragma: no cover - construction must fail
            pytest.fail('expected ValueError')
        assert b'PRIVATE KEY' not in message.encode()
        assert encrypted not in message.encode()

    def test_create_ssl_context_helper_returns_context(self, client_material):
        cert_bytes, key_bytes = client_material
        context = create_ssl_context(cert_bytes, key_bytes)
        assert isinstance(context, ssl.SSLContext)

    def test_check_cert_dates_helper(self, client_material):
        cert_bytes, _ = client_material
        cert = x509.load_pem_x509_certificate(cert_bytes)
        check_cert_dates(cert)


class TestDirectConnection(object):
    def test_mtls_and_ca_bundle_verify(self, ca, client_material,
                                       ca_bundle_file):
        context = target_context(ca)
        seen = {}
        session = make_adapter(*client_material)
        with socketpair_route(
                lambda sock: run_tls_target(sock, context, seen)):
            response = session.get(
                'https://%s/' % TARGET_HOST, verify=ca_bundle_file,
                proxies={})
        assert response.status_code == 200
        assert response.content == b'ok'
        assert seen.get('cert') is not None

    def test_session_verify_default_rejects_untrusted_ca(
            self, ca, client_material, foreign_ca_file):
        context = target_context(ca)
        session = make_adapter(*client_material)
        with socketpair_route(lambda sock: run_tls_target(sock, context)):
            with pytest.raises(requests_exc.SSLError):
                session.get('https://%s/' % TARGET_HOST,
                            verify=foreign_ca_file, proxies={})

    def test_trust_does_not_leak_between_requests(self, ca, client_material,
                                                  ca_bundle_file,
                                                  foreign_ca_file):
        context = target_context(ca)
        session = make_adapter(*client_material)
        with socketpair_route(lambda sock: run_tls_target(sock, context)):
            with pytest.raises(requests_exc.SSLError):
                session.get('https://%s/' % TARGET_HOST,
                            verify=foreign_ca_file, proxies={})
        # A previous foreign bundle must not remain trusted on a later pool.
        with socketpair_route(lambda sock: run_tls_target(sock, context)):
            response = session.get(
                'https://%s/' % TARGET_HOST, verify=ca_bundle_file,
                proxies={})
        assert response.status_code == 200

    def test_hostname_mismatch_rejected(self, ca, client_material,
                                        ca_bundle_file):
        context = target_context(ca, host=TARGET_HOST)
        session = make_adapter(*client_material)
        with socketpair_route(lambda sock: run_tls_target(sock, context)):
            with pytest.raises(requests_exc.SSLError):
                session.get('https://wrong-name.example.test/',
                            verify=ca_bundle_file, proxies={})

    def test_verify_false_overrides_session_verification(
            self, ca, client_material):
        context = target_context(ca)
        session = make_adapter(*client_material)
        session.verify = True
        with socketpair_route(lambda sock: run_tls_target(sock, context)):
            response = session.get(
                'https://%s/' % TARGET_HOST, verify=False, proxies={})
        assert response.status_code == 200

    def test_per_request_bundle_overrides_session_verify(
            self, ca, client_material, ca_bundle_file):
        # Session points at a foreign default bundle; the request supplies
        # the correct one and must succeed (request level wins).
        context = target_context(ca)
        foreign = trustme.CA()
        bad_handle = tempfile.NamedTemporaryFile(suffix='.pem', delete=False)
        bad_handle.write(foreign.cert_pem.bytes())
        bad_handle.close()
        session = make_adapter(*client_material)
        session.verify = bad_handle.name
        try:
            with socketpair_route(
                    lambda sock: run_tls_target(sock, context)):
                response = session.get(
                    'https://%s/' % TARGET_HOST, verify=ca_bundle_file,
                    proxies={})
            assert response.status_code == 200
        finally:
            os.unlink(bad_handle.name)


class TestPoolWiring(object):
    def test_verify_arguments_reach_the_pool_layer(self, client_material):
        session = make_adapter(*client_material)
        request = requests.Request('GET',
                                   'https://%s/' % TARGET_HOST).prepare()
        adapter = session.get_adapter(request.url)

        _, kwargs_true = adapter.build_connection_pool_key_attributes(
            request, True)
        _, kwargs_false = adapter.build_connection_pool_key_attributes(
            request, False)
        _, kwargs_bundle = adapter.build_connection_pool_key_attributes(
            request, '/some/ca-bundle.pem')

        assert kwargs_true['cert_reqs'] == 'CERT_REQUIRED'
        assert kwargs_false['cert_reqs'] == 'CERT_NONE'
        assert kwargs_bundle['cert_reqs'] == 'CERT_REQUIRED'
        assert kwargs_bundle['ca_certs'] == '/some/ca-bundle.pem'
        # Each configuration gets its own context carrying the client cert.
        contexts = {kwargs_true['ssl_context'], kwargs_false['ssl_context'],
                    kwargs_bundle['ssl_context']}
        assert len(contexts) == 3

    def test_poolmanager_and_proxy_manager_receive_ssl_context(
            self, client_material):
        adapter = X509Adapter(cert_bytes=client_material[0],
                              pk_bytes=client_material[1])
        assert isinstance(
            adapter.poolmanager.connection_pool_kw['ssl_context'],
            ssl.SSLContext)
        proxy_manager = adapter.proxy_manager_for(
            'http://%s:3128' % PROXY_HOST)
        assert isinstance(
            proxy_manager.connection_pool_kw['ssl_context'], ssl.SSLContext)


class TestHttpConnectProxy(object):
    PROXY_URL = 'http://%s:3128' % PROXY_HOST

    def test_tunnel_uses_target_trust_and_client_cert(
            self, ca, client_material, ca_bundle_file):
        context = target_context(ca)
        seen = {}
        session = make_adapter(*client_material)
        with socketpair_route(
                lambda sock: http_connect_proxy(sock, context, seen)):
            response = session.get(
                'https://%s/' % TARGET_HOST, verify=ca_bundle_file,
                proxies={'https': self.PROXY_URL})
        assert response.status_code == 200
        assert seen.get('cert') is not None

    def test_tunnel_wrong_target_ca_fails(self, ca, client_material,
                                          foreign_ca_file):
        context = target_context(ca)
        session = make_adapter(*client_material)
        with socketpair_route(
                lambda sock: http_connect_proxy(sock, context)):
            with pytest.raises(requests_exc.SSLError):
                session.get('https://%s/' % TARGET_HOST,
                            verify=foreign_ca_file,
                            proxies={'https': self.PROXY_URL})

    def test_tunnel_verify_false(self, ca, client_material):
        context = target_context(ca)
        session = make_adapter(*client_material)
        with socketpair_route(
                lambda sock: http_connect_proxy(sock, context)):
            response = session.get(
                'https://%s/' % TARGET_HOST, verify=False,
                proxies={'https': self.PROXY_URL})
        assert response.status_code == 200


class TestHttpsConnectProxy(object):
    """An HTTPS proxy has two distinct TLS trust decisions."""

    PROXY_URL = 'https://%s:443' % PROXY_HOST

    def _build(self):
        proxy_ca = trustme.CA()
        target_ca = trustme.CA()
        client_cert = target_ca.issue_cert('client.example.test')
        proxy_context = server_ssl_context(proxy_ca.issue_cert(PROXY_HOST))
        target_context = server_ssl_context(
            target_ca.issue_cert(TARGET_HOST),
            client_ca_pem=target_ca.cert_pem.bytes().decode('ascii'))

        def bundle(cert_authority):
            handle = tempfile.NamedTemporaryFile(
                suffix='.pem', delete=False)
            handle.write(cert_authority.cert_pem.bytes())
            handle.close()
            return handle.name

        cert_file = tempfile.NamedTemporaryFile(suffix='.pem', delete=False)
        cert_file.write(client_cert.cert_chain_pems[0].bytes())
        cert_file.close()
        key_file = tempfile.NamedTemporaryFile(suffix='.pem', delete=False)
        key_file.write(client_cert.private_key_pem.bytes())
        key_file.close()
        return proxy_ca, target_ca, proxy_context, target_context, \
            cert_file.name, key_file.name, bundle

    def _manager(self, proxy_bundle, target_bundle, cert_file, key_file):
        return ProxyManager(
            self.PROXY_URL,
            proxy_ssl_context=ssl.create_default_context(cafile=proxy_bundle),
            ca_certs=target_bundle,
            cert_file=cert_file,
            key_file=key_file)

    def test_proxy_and_target_trust_are_separate(self):
        (proxy_ca, target_ca, proxy_context, target_context,
         cert_file, key_file, bundle) = self._build()
        proxy_bundle = bundle(proxy_ca)
        target_bundle = bundle(target_ca)
        seen = {}
        try:
            manager = self._manager(
                proxy_bundle, target_bundle, cert_file, key_file)
            with socketpair_route(lambda sock: https_connect_proxy(
                    sock, proxy_context, target_context, seen)):
                response = manager.urlopen(
                    'GET', 'https://%s/' % TARGET_HOST, retries=0)
            assert response.status == 200
            assert seen.get('cert') is not None
        finally:
            for path in (proxy_bundle, target_bundle, cert_file, key_file):
                os.unlink(path)

    def test_wrong_proxy_ca_fails_at_proxy_layer(self):
        (proxy_ca, target_ca, proxy_context, target_context,
         cert_file, key_file, bundle) = self._build()
        wrong_proxy_bundle = bundle(trustme.CA())
        target_bundle = bundle(target_ca)
        try:
            manager = self._manager(
                wrong_proxy_bundle, target_bundle, cert_file, key_file)
            with socketpair_route(lambda sock: https_connect_proxy(
                    sock, proxy_context, target_context)):
                with pytest.raises(MaxRetryError) as exc_info:
                    manager.urlopen(
                        'GET', 'https://%s/' % TARGET_HOST, retries=0)
            reason = exc_info.value.reason
            # The origin TLS is never reached; the proxy handshake itself
            # fails, which urllib3 reports as a ProxyError.
            assert type(reason).__name__ == 'ProxyError'
        finally:
            for path in (wrong_proxy_bundle, target_bundle, cert_file,
                         key_file):
                os.unlink(path)

    def test_wrong_target_ca_fails_at_target_layer(self):
        (proxy_ca, target_ca, proxy_context, target_context,
         cert_file, key_file, bundle) = self._build()
        proxy_bundle = bundle(proxy_ca)
        wrong_target_bundle = bundle(trustme.CA())
        try:
            manager = self._manager(
                proxy_bundle, wrong_target_bundle, cert_file, key_file)
            with socketpair_route(lambda sock: https_connect_proxy(
                    sock, proxy_context, target_context)):
                with pytest.raises(MaxRetryError) as exc_info:
                    manager.urlopen(
                        'GET', 'https://%s/' % TARGET_HOST, retries=0)
            # Outer proxy TLS succeeded; the inner origin TLS is what fails.
            # urllib3 raises its own SSLError for the tunnelled origin.
            assert type(exc_info.value.reason).__name__ == 'SSLError'
        finally:
            for path in (proxy_bundle, wrong_target_bundle, cert_file,
                         key_file):
                os.unlink(path)


class TestVersionCompatibility(object):
    """Document and enforce the version combination the adapter supports.

    The adapter deliberately depends on capabilities rather than a single
    hard-coded version, so these probes pin the required integration points
    instead of brittle exact version strings.
    """

    def test_requests_exposes_ssl_context_integration(self):
        # Requests >= 2.12 threads an ssl_context through pool construction.
        assert requests.__build__ >= 0x021200
        assert hasattr(
            requests.adapters.HTTPAdapter,
            'build_connection_pool_key_attributes')

    def test_urllib3_uses_stdlib_ssl_context(self):
        # urllib3 1.26.x and 2.x ship create_urllib3_context and accept a
        # stdlib ssl.SSLContext; the pyOpenSSL shim is gone in 2.x.
        from urllib3.util.ssl_ import create_urllib3_context
        context = create_urllib3_context()
        assert isinstance(context, ssl.SSLContext)
        assert hasattr(context, 'load_cert_chain')

    def test_pyopenssl_shim_is_not_required(self):
        # The adapter builds a stdlib context and keeps no pyOpenSSL state,
        # so it works on stacks where pyOpenSSL is not installed.
        module = importlib.import_module(
            'requests_toolbelt.adapters.x509')
        assert not hasattr(module, 'PyOpenSSLContext')
        source = inspect.getsource(module)
        assert 'import OpenSSL' not in source
        assert 'contrib.pyopenssl' not in source

    def test_recorded_combination_imports_cleanly(self):
        import urllib3
        # Sanity: the exact stack exercised by this suite imports together.
        assert tuple(int(part) for part in
                     urllib3.__version__.split('.')[:2]) >= (1, 26)


class TestKeyMaterialHandling(object):
    def test_temp_chain_files_are_removed(self, client_material):
        import glob
        tmp_dir = tempfile.gettempdir()
        before = set(glob.glob(os.path.join(tmp_dir, 'tmp*.pem')))
        adapter = X509Adapter(cert_bytes=client_material[0],
                              pk_bytes=client_material[1])
        adapter._new_ssl_context()
        adapter._new_ssl_context()
        after = set(glob.glob(os.path.join(tmp_dir, 'tmp*.pem')))
        assert after - before == set()

    def test_encrypted_key_never_decrypted_on_disk(self, ca):
        cert = ca.issue_cert('client.example.test')
        password = b'disk-safe-passphrase'
        key = load_pem_private_key(cert.private_key_pem.bytes(),
                                   password=None)
        encrypted_key = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8,
            BestAvailableEncryption(password))
        captured = []
        real_load = ssl.SSLContext.load_cert_chain

        def spying_load(context, path, password=None):
            with open(path, 'rb') as handle:
                captured.append(handle.read())
            return real_load(context, path, password=password)

        with mock.patch.object(ssl.SSLContext, 'load_cert_chain',
                               spying_load):
            adapter = X509Adapter(
                cert_bytes=cert.cert_chain_pems[0].bytes(),
                pk_bytes=encrypted_key, password=password)
            adapter._new_ssl_context()
        assert captured, 'expected the chain file to be inspected'
        contents = b''.join(captured)
        # The on-disk PEM is the encrypted form, never the plaintext key.
        assert b'ENCRYPTED' in contents
        assert cert.private_key_pem.bytes() not in contents


class TestPickling(object):
    def test_adapter_roundtrip_preserves_material(self, client_material):
        import pickle
        adapter = X509Adapter(cert_bytes=client_material[0],
                              pk_bytes=client_material[1])
        restored = pickle.loads(pickle.dumps(adapter))
        assert restored._cert_pem == adapter._cert_pem
        assert restored._key_pem == adapter._key_pem
        assert isinstance(
            restored.poolmanager.connection_pool_kw['ssl_context'],
            ssl.SSLContext)

    def test_encrypted_adapter_roundtrip(self, ca):
        import pickle
        cert = ca.issue_cert('client.example.test')
        password = b'pickle-pass'
        key = load_pem_private_key(cert.private_key_pem.bytes(),
                                   password=None)
        encrypted = key.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8,
            BestAvailableEncryption(password))
        adapter = X509Adapter(
            cert_bytes=cert.cert_chain_pems[0].bytes(),
            pk_bytes=encrypted, password=password)
        restored = pickle.loads(pickle.dumps(adapter))
        assert restored._password == password
        assert isinstance(restored._new_ssl_context(), ssl.SSLContext)
