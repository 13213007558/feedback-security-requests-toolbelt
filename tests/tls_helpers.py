# -*- coding: utf-8 -*-
"""In-process TLS stand-ins used by the X509Adapter tests.

The test environment forbids binding listening sockets, so the "servers"
and "proxies" here are threads driving the other end of a
:func:`socket.socketpair`. All key material is generated on the fly with
``trustme``; no real private keys are checked into the repository.
"""
import socket
import ssl
import threading
from contextlib import contextmanager
from unittest import mock

import urllib3.util.connection as u3conn


RESPONSE = (b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n'
            b'Connection: close\r\n\r\nok')


class MemoryBioTLSServer(object):
    """TLS server built on MemoryBIO so it can wrap an existing socket.

    Plain ``SSLContext.wrap_socket`` cannot nest two TLS connections (the
    HTTPS-proxy case), whereas ``wrap_bio`` can. This helper speaks the
    inner TLS layer over whatever transport socket it is given.
    """

    def __init__(self, transport, context):
        self.transport = transport
        self._in = ssl.MemoryBIO()
        self._out = ssl.MemoryBIO()
        self._ssl = context.wrap_bio(self._in, self._out, server_side=True)

    def _flush(self):
        data = self._out.read()
        if data:
            self.transport.sendall(data)

    def _feed_transport(self):
        data = self.transport.recv(65536)
        if not data:
            raise ConnectionError('peer closed the transport')
        self._in.write(data)

    def handshake(self):
        while True:
            try:
                self._ssl.do_handshake()
                break
            except ssl.SSLWantReadError:
                self._flush()
                self._feed_transport()
        self._flush()

    def read(self, length=4096):
        """Read decrypted bytes, refilling the input BIO only on demand."""
        while True:
            try:
                return self._ssl.read(length)
            except ssl.SSLWantReadError:
                self._flush()
                self._feed_transport()

    def read_request(self):
        data = b''
        while b'\r\n\r\n' not in data:
            data += self.read()
        return data

    def send_response(self):
        self._ssl.write(RESPONSE)
        self._flush()

    def peer_cert(self):
        return self._ssl.getpeercert()


def server_ssl_context(trustme_cert, client_ca_pem=None):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    trustme_cert.configure_cert(context)
    if client_ca_pem is not None:
        context.verify_mode = ssl.CERT_REQUIRED
        context.load_verify_locations(cadata=client_ca_pem)
    return context


def _drain_and_reply(ssl_socket):
    buf = b''
    while b'\r\n\r\n' not in buf:
        chunk = ssl_socket.recv(4096)
        if not chunk:
            return
        buf += chunk
    ssl_socket.sendall(RESPONSE)


def run_tls_target(raw_socket, server_context, peer_cert_sink=None):
    """Emulate a direct HTTPS origin on a raw socketpair endpoint."""
    try:
        wrapped = server_context.wrap_socket(raw_socket, server_side=True)
    except (ssl.SSLError, OSError):
        # The client rejected our certificate and closed the connection.
        raw_socket.close()
        return
    try:
        if peer_cert_sink is not None:
            peer_cert_sink['cert'] = wrapped.getpeercert()
        _drain_and_reply(wrapped)
    except (ssl.SSLError, OSError):
        pass
    finally:
        wrapped.close()


def http_connect_proxy(raw_socket, target_context, peer_cert_sink=None):
    """Plaintext HTTP proxy that answers CONNECT then speaks TLS inwards."""
    request = b''
    try:
        while b'\r\n\r\n' not in request:
            chunk = raw_socket.recv(4096)
            if not chunk:
                raw_socket.close()
                return
            request += chunk
        if not request.startswith(b'CONNECT'):
            raise AssertionError('expected CONNECT, got %r' % request[:20])
        raw_socket.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
        wrapped = target_context.wrap_socket(raw_socket, server_side=True)
        if peer_cert_sink is not None:
            peer_cert_sink['cert'] = wrapped.getpeercert()
        _drain_and_reply(wrapped)
        wrapped.close()
    except (ssl.SSLError, OSError):
        raw_socket.close()


def https_connect_proxy(raw_socket, proxy_context, target_context,
                        peer_cert_sink=None):
    """TLS-to-the-proxy proxy, then a nested TLS tunnel to the target."""
    try:
        outer = proxy_context.wrap_socket(raw_socket, server_side=True)
        request = b''
        while b'\r\n\r\n' not in request:
            chunk = outer.recv(4096)
            if not chunk:
                outer.close()
                return
            request += chunk
        if not request.startswith(b'CONNECT'):
            raise AssertionError('expected CONNECT, got %r' % request[:20])
        outer.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
        inner = MemoryBioTLSServer(outer, target_context)
        inner.handshake()
        if peer_cert_sink is not None:
            peer_cert_sink['cert'] = inner.peer_cert()
        inner.read_request()
        inner.send_response()
    except (ssl.SSLError, OSError):
        # Expected on the negative trust tests when the client aborts one
        # of the two TLS layers.
        try:
            raw_socket.close()
        except OSError:
            pass


@contextmanager
def socketpair_route(handler):
    """Route urllib3's connect() to one end of a socketpair.

    ``handler`` runs in a daemon thread on the other end, emulating the
    remote peer without any network port.
    """
    client_sock, server_sock = socket.socketpair()
    thread = threading.Thread(target=handler, args=(server_sock,))
    thread.daemon = True
    thread.start()
    holder = [client_sock]

    def fake_create_connection(address, *args, **kwargs):
        return holder.pop()

    fake_addrinfo = [(socket.AF_INET, socket.SOCK_STREAM, 6, '',
                      ('127.0.0.1', 443))]
    with mock.patch.object(u3conn, 'create_connection',
                           fake_create_connection), \
            mock.patch.object(u3conn.socket, 'getaddrinfo',
                              lambda *a, **k: fake_addrinfo):
        yield thread
    thread.join(timeout=5)
