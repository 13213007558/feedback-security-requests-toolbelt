# -*- coding: utf-8 -*-
"""The module containing the code for GuessAuth."""
from requests import auth
from requests import cookies
from requests.compat import basestring

try:
    from requests.exceptions import UnrewindableBodyError
except ImportError:
    # requests < 2.8.0 does not ship UnrewindableBodyError.
    from requests.exceptions import RequestException

    class UnrewindableBodyError(RequestException):
        """The request body could not be rewound for an auth retry."""

from . import _digest_auth_compat as auth_compat, http_proxy_digest


class GuessAuth(auth.AuthBase):
    """Guesses the auth type by the WWW-Authentication header.

    Retrying a request after an authentication challenge requires the
    request body to be replayed. Bodies that can be seeked back to their
    starting position (e.g. files) are rewound automatically. Bodies that
    can only be consumed once (e.g. a
    :class:`~requests_toolbelt.multipart.encoder.MultipartEncoder` or a
    generator) cannot be replayed: instead of sending a truncated or
    empty body, an :exc:`requests.exceptions.UnrewindableBodyError` is
    raised before anything is resent.
    """

    #: How many times a single request may be resent with authentication
    #: before giving up and returning the last response to the user.
    max_auth_retries = 10

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.auth = None
        self.pos = None
        self._num_retries = 0

    def _rewind_body(self, r):
        """Rewind the request body so the request can be replayed.

        :raises UnrewindableBodyError: if the body cannot be returned to
            its starting position. Nothing is resent in that case, so the
            server never sees a truncated or empty body.
        """
        body = r.request.body
        if body is None or isinstance(body, basestring):
            return
        if self.pos is None:
            raise UnrewindableBodyError(
                'Unable to resend the request to answer an '
                'authentication challenge: the request body (a %s) can '
                'only be consumed once and has already been sent. Use a '
                'seekable body (e.g. bytes or a file object) so the '
                'request can be replayed.' % type(body).__name__)
        try:
            body.seek(self.pos)
        except (AttributeError, IOError, OSError, ValueError):
            raise UnrewindableBodyError(
                'Unable to resend the request to answer an '
                'authentication challenge: rewinding the request body '
                '(a %s) to position %s failed.'
                % (type(body).__name__, self.pos))

    def _handle_basic_auth_401(self, r, kwargs):
        if 'Authorization' in r.request.headers:
            # We already sent our credentials and the server is still
            # challenging us: they were rejected. Stop instead of
            # resending them forever.
            return r

        self._rewind_body(r)

        # Consume content and release the original connection
        # to allow our new request to reuse the same one.
        r.content
        r.close()
        prep = r.request.copy()
        if not hasattr(prep, '_cookies'):
            prep._cookies = cookies.RequestsCookieJar()
        cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
        prep.prepare_cookies(prep._cookies)

        self.auth = auth.HTTPBasicAuth(self.username, self.password)
        prep = self.auth(prep)
        _r = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep

        return _r

    def _handle_digest_auth_401(self, r, kwargs):
        if not isinstance(self.auth, auth_compat.HTTPDigestAuth):
            self.auth = auth_compat.HTTPDigestAuth(self.username,
                                                   self.password)
        try:
            self.auth.init_per_thread_state()
        except AttributeError:
            # If we're not on requests 2.8.0+ this method does not exist and
            # is not relevant.
            pass

        # Check that the attr exists because much older versions of requests
        # set this attribute lazily. For example:
        # https://github.com/kennethreitz/requests/blob/33735480f77891754304e7f13e3cdf83aaaa76aa/requests/auth.py#L59
        if (hasattr(self.auth, 'num_401_calls') and
                self.auth.num_401_calls is None):
            self.auth.num_401_calls = 1

        # The digest auth object never saw the original request, so it
        # cannot rewind the body by itself. Rewind it here (raising if
        # the body cannot be replayed) and share the position so the
        # digest implementation can rewind on its own retries.
        self._rewind_body(r)
        self.auth.pos = self.pos

        # Digest auth would resend the request by itself. We can take a
        # shortcut here.
        return self.auth.handle_401(r, **kwargs)

    def handle_401(self, r, **kwargs):
        """Resends a request with auth headers, if needed."""
        while (r.status_code == 401 and
                self._num_retries < self.max_auth_retries):
            www_authenticate = r.headers.get('www-authenticate', '').lower()

            if 'basic' in www_authenticate:
                new_r = self._handle_basic_auth_401(r, kwargs)
            elif 'digest' in www_authenticate:
                new_r = self._handle_digest_auth_401(r, kwargs)
            else:
                break

            if new_r is r:
                # The handler declined to resend the request: stop
                # instead of looping on a repeated challenge.
                break
            self._num_retries += 1
            r = new_r

        return r

    def __call__(self, request):
        self._num_retries = 0

        try:
            self.pos = request.body.tell()
        except AttributeError:
            self.pos = None

        if self.auth is not None:
            return self.auth(request)

        request.register_hook('response', self.handle_401)
        return request


class GuessProxyAuth(GuessAuth):
    """
    Guesses the auth type by WWW-Authentication and Proxy-Authentication
    headers
    """
    def __init__(self, username=None, password=None,
                 proxy_username=None, proxy_password=None):
        super(GuessProxyAuth, self).__init__(username, password)
        self.proxy_username = proxy_username
        self.proxy_password = proxy_password
        self.proxy_auth = None

    def _handle_basic_auth_407(self, r, kwargs):
        if 'Proxy-Authorization' in r.request.headers:
            # We already sent our proxy credentials and the proxy is
            # still challenging us: they were rejected. Stop instead of
            # resending them forever.
            return r

        self._rewind_body(r)

        # Consume content and release the original connection
        # to allow our new request to reuse the same one.
        r.content
        r.close()
        prep = r.request.copy()
        if not hasattr(prep, '_cookies'):
            prep._cookies = cookies.RequestsCookieJar()
        cookies.extract_cookies_to_jar(prep._cookies, r.request, r.raw)
        prep.prepare_cookies(prep._cookies)

        self.proxy_auth = auth.HTTPProxyAuth(self.proxy_username,
                                             self.proxy_password)
        prep = self.proxy_auth(prep)
        _r = r.connection.send(prep, **kwargs)
        _r.history.append(r)
        _r.request = prep

        return _r

    def _handle_digest_auth_407(self, r, kwargs):
        if not isinstance(self.proxy_auth,
                          http_proxy_digest.HTTPProxyDigestAuth):
            self.proxy_auth = http_proxy_digest.HTTPProxyDigestAuth(
                username=self.proxy_username,
                password=self.proxy_password)

        try:
            self.proxy_auth.init_per_thread_state()
        except AttributeError:
            pass

        if 'Proxy-Authorization' not in r.request.headers:
            # This is the first challenge for this request, not a stale
            # rejection of credentials we already sent.
            self.proxy_auth.stale_rejects = 0

        # HTTPProxyDigestAuth.handle_407 does not rewind the body, so do
        # it here (raising if the body cannot be replayed).
        self._rewind_body(r)

        return self.proxy_auth.handle_407(r, **kwargs)

    def handle_407(self, r, **kwargs):
        while (r.status_code == 407 and
                self._num_retries < self.max_auth_retries):
            proxy_authenticate = r.headers.get('Proxy-Authenticate',
                                               '').lower()

            if 'basic' in proxy_authenticate:
                new_r = self._handle_basic_auth_407(r, kwargs)
            elif 'digest' in proxy_authenticate:
                new_r = self._handle_digest_auth_407(r, kwargs)
            else:
                break

            if new_r is r:
                # The handler declined to resend the request: stop
                # instead of looping on a repeated challenge.
                break
            self._num_retries += 1
            r = new_r

        return r

    def __call__(self, request):
        if self.proxy_auth is not None:
            request = self.proxy_auth(request)

        request.register_hook('response', self.handle_407)
        return super(GuessProxyAuth, self).__call__(request)
