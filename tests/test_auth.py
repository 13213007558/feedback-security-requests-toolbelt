# -*- coding: utf-8 -*-
import io
import requests
import unittest
try:
    from unittest import mock
except ImportError:
    import mock

from requests.adapters import BaseAdapter
from requests.exceptions import UnrewindableBodyError

from requests_toolbelt.auth.guess import GuessAuth, GuessProxyAuth
from requests_toolbelt.multipart.encoder import MultipartEncoder
from . import get_betamax


class TestGuessAuth(unittest.TestCase):
    def setUp(self):
        self.session = requests.Session()
        self.recorder = get_betamax(self.session)

    def cassette(self, name):
        return self.recorder.use_cassette(
            'httpbin_guess_auth_' + name,
            match_requests_on=['method', 'uri', 'digest-auth']
        )

    def test_basic(self):
        with self.cassette('basic'):
            r = self.session.request(
                'GET', 'http://httpbin.org/basic-auth/user/passwd',
                auth=GuessAuth('user', 'passwd'))

        assert r.json() == {'authenticated': True, 'user': 'user'}

    def test_digest(self):
        with self.cassette('digest'):
            r = self.session.request(
                'GET', 'http://httpbin.org/digest-auth/auth/user/passwd',
                auth=GuessAuth('user', 'passwd'))

        assert r.json() == {'authenticated': True, 'user': 'user'}

    def test_no_auth(self):
        with self.cassette('none'):
            url = 'http://httpbin.org/get?a=1'
            r = self.session.request('GET', url,
                                     auth=GuessAuth('user', 'passwd'))

            j = r.json()
            assert j['args'] == {'a': '1'}
            assert j['url'] == url
            assert 'user' not in r.text
            assert 'passwd' not in r.text


class TestGuessProxyAuth(unittest.TestCase):

    @mock.patch('requests_toolbelt.auth.http_proxy_digest.HTTPProxyDigestAuth.handle_407')
    def test_handle_407_header_digest(self, mock_handle_407):
        r = requests.Response()
        r.status_code = 407
        r.request = mock.Mock(body=None, headers={})
        r.headers['Proxy-Authenticate'] = 'Digest nonce="d2b19757d3d656a283c99762cbd1097b", opaque="1c311ad1cc6e6183b83bc75f95a57893", realm="me@kennethreitz.com", qop=auth'

        guess_auth = GuessProxyAuth(None, None, "user", "passwd")
        guess_auth.handle_407(r)

        mock_handle_407.assert_called_with(r)

    @mock.patch('requests.auth.HTTPProxyAuth.__call__')
    @mock.patch('requests.cookies.extract_cookies_to_jar')
    def test_handle_407_header_basic(self, extract_cookies_to_jar, proxy_auth_call):
        req = mock.Mock()
        r = mock.Mock()
        r.status_code = 407
        r.headers = dict()
        r.request.copy.return_value = req
        r.request.headers = {}
        r.request.body = None

        proxy_auth_call.return_value = requests.Response()

        kwargs = {}
        r.headers['Proxy-Authenticate'] = 'Basic realm="Fake Realm"'
        guess_auth = GuessProxyAuth(None, None, "user", "passwd")
        guess_auth.handle_407(r, *kwargs)

        proxy_auth_call.assert_called_with(req)


def make_response(status_code, headers=None, content=b'response'):
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    response._content = content
    response._content_consumed = True
    return response


class RecordingAdapter(BaseAdapter):
    """Fake adapter that replays scripted responses and records sends."""

    def __init__(self, responses):
        super(RecordingAdapter, self).__init__()
        self.responses = list(responses)
        self.sent = []

    def send(self, request, **kwargs):
        body = request.body
        if hasattr(body, 'read'):
            body = body.read()
        self.sent.append((dict(request.headers), body))
        response = self.responses.pop(0)
        response.request = request
        response.connection = self
        return response

    def close(self):
        pass


class TestGuessAuthRetryChain(unittest.TestCase):
    url = 'http://example.com/upload'

    def make_session(self, responses):
        adapter = RecordingAdapter(responses)
        session = requests.Session()
        session.mount('http://', adapter)
        return session, adapter

    def sent_bodies(self, adapter):
        return [body for _, body in adapter.sent]

    def test_proxy_then_origin_basic_replays_full_body(self):
        body = b'upload-payload' * 64
        session, adapter = self.make_session([
            make_response(407, {'Proxy-Authenticate': 'Basic realm="proxy"'}),
            make_response(401, {'WWW-Authenticate': 'Basic realm="origin"'}),
            make_response(200),
        ])
        auth = GuessProxyAuth('user', 'passwd', 'proxyuser', 'proxypasswd')
        r = session.post(self.url, data=io.BytesIO(body), auth=auth)

        assert r.status_code == 200
        # Auth retries nest history the same way requests itself does.
        assert [resp.status_code for resp in r.history] == [401]
        assert [resp.status_code for resp in r.history[0].history] == [407]
        assert len(adapter.sent) == 3
        # Every attempt, including the second authentication, carries the
        # complete body: never empty, never just the trailing part.
        assert self.sent_bodies(adapter) == [body, body, body]
        assert 'Proxy-Authorization' not in adapter.sent[0][0]
        assert 'Proxy-Authorization' in adapter.sent[1][0]
        assert 'Authorization' not in adapter.sent[1][0]
        assert 'Proxy-Authorization' in adapter.sent[2][0]
        assert 'Authorization' in adapter.sent[2][0]

    def test_proxy_then_origin_digest_replays_full_body(self):
        body = b'digest-upload' * 64
        session, adapter = self.make_session([
            make_response(407, {'Proxy-Authenticate':
                                'Digest realm="proxy", nonce="pnonce", '
                                'qop=auth'}),
            make_response(401, {'WWW-Authenticate':
                                'Digest realm="origin", nonce="ononce", '
                                'qop=auth'}),
            make_response(200),
        ])
        auth = GuessProxyAuth('user', 'passwd', 'proxyuser', 'proxypasswd')
        r = session.post(self.url, data=io.BytesIO(body), auth=auth)

        assert r.status_code == 200
        assert len(adapter.sent) == 3
        assert self.sent_bodies(adapter) == [body, body, body]
        proxy_header = adapter.sent[1][0]['Proxy-Authorization']
        origin_header = adapter.sent[2][0]['Authorization']
        assert proxy_header.startswith('Digest ')
        assert origin_header.startswith('Digest ')

    def test_origin_basic_replays_full_body(self):
        body = b'origin-upload' * 64
        session, adapter = self.make_session([
            make_response(401, {'WWW-Authenticate': 'Basic realm="origin"'}),
            make_response(200),
        ])
        r = session.post(self.url, data=io.BytesIO(body),
                         auth=GuessAuth('user', 'passwd'))

        assert r.status_code == 200
        assert self.sent_bodies(adapter) == [body, body]

    def test_non_replayable_body_raises_on_407(self):
        session, adapter = self.make_session([
            make_response(407, {'Proxy-Authenticate': 'Basic realm="proxy"'}),
        ])
        auth = GuessProxyAuth('user', 'passwd', 'proxyuser', 'proxypasswd')
        encoder = MultipartEncoder({'field': 'value'})

        with self.assertRaises(UnrewindableBodyError):
            session.post(self.url, data=encoder, auth=auth)

        # The chain stops at the step that cannot be replayed: only the
        # original request was sent, no truncated retry went out.
        assert len(adapter.sent) == 1

    def test_non_replayable_body_raises_on_401(self):
        session, adapter = self.make_session([
            make_response(401, {'WWW-Authenticate': 'Basic realm="origin"'}),
        ])
        encoder = MultipartEncoder({'field': 'value'})

        with self.assertRaises(UnrewindableBodyError):
            session.post(self.url, data=encoder,
                         auth=GuessAuth('user', 'passwd'))

        assert len(adapter.sent) == 1

    def test_repeated_basic_challenges_terminate(self):
        session, adapter = self.make_session(
            [make_response(401, {'WWW-Authenticate': 'Basic realm="x"'})
             for _ in range(5)] + [make_response(200)])
        r = session.get('http://example.com/', auth=GuessAuth('user', 'p'))

        # Credentials are sent once; the repeated challenge is returned
        # instead of looping forever.
        assert r.status_code == 401
        assert len(adapter.sent) == 2

    def test_digest_stale_challenges_terminate(self):
        body = b'stale-upload' * 64
        stale = {'Proxy-Authenticate':
                 'Digest realm="proxy", nonce="n1", qop=auth, stale=true'}
        session, adapter = self.make_session([
            make_response(407, {'Proxy-Authenticate':
                                'Digest realm="proxy", nonce="n0", '
                                'qop=auth'}),
            make_response(407, stale),
            make_response(407, stale),
            make_response(407, stale),
            make_response(407, stale),
        ])
        auth = GuessProxyAuth(None, None, 'proxyuser', 'proxypasswd')
        r = session.post(self.url, data=io.BytesIO(body), auth=auth)

        # The initial answer plus at most two stale retries, then the
        # last 407 is returned instead of looping forever.
        assert r.status_code == 407
        assert len(adapter.sent) == 4
        assert self.sent_bodies(adapter) == [body] * 4

    def test_basic_digest_switching_terminates(self):
        session, adapter = self.make_session([
            make_response(401, {'WWW-Authenticate': 'Basic realm="x"'}),
            make_response(401, {'WWW-Authenticate':
                                'Digest realm="x", nonce="n", qop=auth'}),
            make_response(401, {'WWW-Authenticate': 'Basic realm="x"'}),
            make_response(401, {'WWW-Authenticate':
                                'Digest realm="x", nonce="n2", qop=auth'}),
            make_response(200),
        ])
        r = session.get('http://example.com/', auth=GuessAuth('user', 'p'))

        # Basic, then Digest are each tried once; when the server
        # switches back to Basic after credentials were already sent,
        # the chain stops instead of ping-ponging.
        assert r.status_code == 401
        assert len(adapter.sent) == 3
        assert adapter.sent[1][0]['Authorization'].startswith('Basic ')
        assert adapter.sent[2][0]['Authorization'].startswith('Digest ')
