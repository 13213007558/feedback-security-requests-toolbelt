# -*- coding: utf-8 -*-
"""

requests_toolbelt.multipart.decoder
===================================

This holds all the implementation details of the MultipartDecoder

"""

import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict


def _split_on_find(content, bound):
    point = content.find(bound)
    return content[:point], content[point + len(bound):]


def _parse_boundary(content_type, encoding):
    ct_info = tuple(x.strip() for x in content_type.split(';'))
    mimetype = ct_info[0]
    if mimetype.split('/')[0].lower() != 'multipart':
        raise NonMultipartContentTypeException(
            "Unexpected mimetype in content-type: '{}'".format(mimetype)
        )
    for item in ct_info[1:]:
        attr, value = _split_on_find(
            item,
            '='
        )
        if attr.lower() == 'boundary':
            return encode_with(value.strip('"'), encoding)


class ImproperBodyPartContentException(Exception):
    pass


class NonMultipartContentTypeException(Exception):
    pass


class MultipartStreamException(Exception):
    """Base exception for errors raised while streaming multipart bodies."""


class TruncatedMultipartException(MultipartStreamException):
    """The stream ended before the closing boundary was received."""


def _header_parser(string, encoding):
    major = sys.version_info[0]
    if major == 3:
        string = string.decode(encoding)
    headers = email.parser.HeaderParser().parsestr(string).items()
    return (
        (encode_with(k, encoding), encode_with(v, encoding))
        for k, v in headers
    )


class BodyPart(object):
    """

    The ``BodyPart`` object is a ``Response``-like interface to an individual
    subpart of a multipart response. It is expected that these will
    generally be created by objects of the ``MultipartDecoder`` class.

    Like ``Response``, there is a ``CaseInsensitiveDict`` object named headers,
    ``content`` to access bytes, ``text`` to access unicode, and ``encoding``
    to access the unicode codec.

    """

    def __init__(self, content, encoding):
        self.encoding = encoding
        headers = {}
        # Split into header section (if any) and the content
        if b'\r\n\r\n' in content:
            first, self.content = _split_on_find(content, b'\r\n\r\n')
            if first != b'':
                headers = _header_parser(first.lstrip(), encoding)
        else:
            raise ImproperBodyPartContentException(
                'content does not contain CR-LF-CR-LF'
            )
        self.headers = CaseInsensitiveDict(headers)

    @property
    def text(self):
        """Content of the ``BodyPart`` in unicode."""
        return self.content.decode(self.encoding)


class MultipartDecoder(object):
    """

    The ``MultipartDecoder`` object parses the multipart payload of
    a bytestring into a tuple of ``Response``-like ``BodyPart`` objects.

    The basic usage is::

        import requests
        from requests_toolbelt import MultipartDecoder

        response = requests.get(url)
        decoder = MultipartDecoder.from_response(response)
        for part in decoder.parts:
            print(part.headers['content-type'])

    If the multipart content is not from a response, basic usage is::

        from requests_toolbelt import MultipartDecoder

        decoder = MultipartDecoder(content, content_type)
        for part in decoder.parts:
            print(part.headers['content-type'])

    For both these usages, there is an optional ``encoding`` parameter. This is
    a string, which is the name of the unicode codec to use (default is
    ``'utf-8'``).

    """
    def __init__(self, content, content_type, encoding='utf-8'):
        #: Original Content-Type header
        self.content_type = content_type
        #: Response body encoding
        self.encoding = encoding
        #: Parsed parts of the multipart response body
        self.parts = tuple()
        self._find_boundary()
        self._parse_body(content)

    def _find_boundary(self):
        self.boundary = _parse_boundary(self.content_type, self.encoding)

    @staticmethod
    def _fix_first_part(part, boundary_marker):
        bm_len = len(boundary_marker)
        if boundary_marker == part[:bm_len]:
            return part[bm_len:]
        else:
            return part

    def _parse_body(self, content):
        boundary = b''.join((b'--', self.boundary))

        def body_part(part):
            fixed = MultipartDecoder._fix_first_part(part, boundary)
            return BodyPart(fixed, self.encoding)

        def test_part(part):
            return (part != b'' and
                    part != b'\r\n' and
                    part[:4] != b'--\r\n' and
                    part != b'--')

        parts = content.split(b''.join((b'\r\n', boundary)))
        self.parts = tuple(body_part(x) for x in parts if test_part(x))

    @classmethod
    def from_response(cls, response, encoding='utf-8'):
        content = response.content
        content_type = response.headers.get('content-type', None)
        return cls(content, content_type, encoding)


class StreamingBodyPart(object):
    """

    A ``Response``-like interface to a single subpart of a multipart
    stream, created by :class:`StreamingMultipartDecoder`.

    Unlike :class:`BodyPart`, the content is *not* held in memory up
    front. It is pulled from the underlying stream on demand via
    :meth:`read` or :meth:`iter_content`, so peak memory stays
    proportional to the read chunk size rather than the part size.

    Consumption rules:

    - A part is only readable while it is the decoder's active part.
      Asking the decoder for the next part discards any unread
      remainder of the current one; reading a part after the decoder
      has advanced raises :class:`MultipartStreamException`.
    - Reading a fully consumed part returns ``b''``.
    - Reading after the decoder has been closed raises
      :class:`MultipartStreamException`.

    """

    def __init__(self, decoder, headers):
        self._decoder = decoder
        self._exhausted = False
        self._abandoned = False
        #: Response body encoding
        self.encoding = decoder.encoding
        #: Headers of this part
        self.headers = CaseInsensitiveDict(headers)

    def read(self, size=-1):
        """Read up to ``size`` bytes of this part's content.

        With the default ``size=-1``, read all remaining content of
        this part into memory. Returns ``b''`` once the part is fully
        consumed.
        """
        if self._abandoned:
            raise MultipartStreamException(
                'Cannot read part: the decoder has advanced past it; '
                'unread content was discarded'
            )
        if self._exhausted:
            return b''
        decoder = self._decoder
        if decoder._closed:
            raise MultipartStreamException(
                'Cannot read part: the decoder is closed'
            )
        if decoder._active_part is not self:
            raise MultipartStreamException(
                'Cannot read part: the decoder has advanced past it; '
                'unread content was discarded'
            )
        return decoder._read_from_part(size)

    def iter_content(self, chunk_size=1024):
        """Iterate over this part's content in chunks of at most
        ``chunk_size`` bytes."""
        while True:
            chunk = self.read(chunk_size)
            if not chunk:
                return
            yield chunk

    @property
    def content(self):
        """All remaining unread content of this part, as bytes.

        Note that this buffers the remainder of the part in memory and
        consumes it from the stream.
        """
        return self.read()

    @property
    def text(self):
        """Remaining content of this part in unicode."""
        return self.content.decode(self.encoding)


class StreamingMultipartDecoder(object):
    """

    Incrementally decode a multipart body, yielding one
    :class:`StreamingBodyPart` at a time without buffering the whole
    payload in memory.

    The basic usage with a response is::

        import requests
        from requests_toolbelt import StreamingMultipartDecoder

        response = requests.get(url, stream=True)
        with StreamingMultipartDecoder.from_response(response) as decoder:
            for part in decoder:
                print(part.headers['content-type'])
                for chunk in part.iter_content(65536):
                    process(chunk)

    If the multipart content does not come from a response, pass an
    iterable of byte chunks (or a file-like object with ``read``)::

        decoder = StreamingMultipartDecoder(chunks, content_type)
        for part in decoder:
            ...

    Consumption and resource rules:

    - Parts must be consumed in order. Requesting the next part before
      the current one is fully read silently discards the current
      part's remaining bytes.
    - Calling :meth:`close` (or leaving the ``with`` block) stops the
      iteration and releases the underlying response/connection
      without reading the body to the end.
    - If the underlying stream fails mid-body, the original exception
      propagates to the caller at the point of failure; the decoder is
      then closed and yields no further parts.
    - If the stream ends before the closing boundary,
      :class:`TruncatedMultipartException` is raised.

    A delimiter is only recognized when ``--boundary`` appears at the
    start of a line and is immediately followed by ``\\r\\n`` or
    ``--``; boundary-like prefixes inside content are delivered
    verbatim.

    """

    _HEADER_TERMINATOR = b'\r\n\r\n'

    def __init__(self, stream, content_type, encoding='utf-8',
                 chunk_size=8192):
        #: Original Content-Type header
        self.content_type = content_type
        #: Response body encoding
        self.encoding = encoding
        self.boundary = _parse_boundary(content_type, encoding)
        self._marker = b'--' + self.boundary
        self._delimiter = b'\r\n' + self._marker
        if hasattr(stream, 'read'):
            self._chunks = self._file_chunks(stream, chunk_size)
        else:
            self._chunks = iter(stream)
        self._buffer = bytearray()
        self._eof = False
        self._state = 'preamble'
        self._active_part = None
        self._closed = False
        self._failed = False
        self._response = None

    @staticmethod
    def _file_chunks(fp, chunk_size):
        while True:
            chunk = fp.read(chunk_size)
            if not chunk:
                return
            yield chunk

    @classmethod
    def from_response(cls, response, encoding='utf-8', chunk_size=8192):
        """Create a streaming decoder from a ``requests`` response.

        The response should be requested with ``stream=True`` so its
        body is not downloaded up front. Closing the decoder closes
        the response.
        """
        content_type = response.headers.get('content-type', None)
        decoder = cls(
            response.iter_content(chunk_size),
            content_type,
            encoding=encoding,
            chunk_size=chunk_size,
        )
        decoder._response = response
        return decoder

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed or self._failed:
            raise StopIteration
        if self._state == 'preamble':
            self._consume_preamble()
        if self._active_part is not None and \
                not self._active_part._exhausted:
            self._drain_active_part()
        if self._state == 'done' or self._closed:
            raise StopIteration
        raw_headers = self._read_headers()
        if raw_headers:
            headers = _header_parser(raw_headers, self.encoding)
        else:
            headers = ()
        part = StreamingBodyPart(self, headers)
        self._active_part = part
        self._state = 'body'
        return part

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def close(self):
        """Stop decoding and release the underlying stream/response.

        Any unread part content is abandoned without being read.
        """
        if self._closed:
            return
        self._closed = True
        self._state = 'done'
        self._active_part = None
        close = getattr(self._chunks, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:
                pass
        if self._response is not None:
            self._response.close()

    def _read_more(self):
        while not self._eof:
            try:
                chunk = next(self._chunks)
            except StopIteration:
                self._eof = True
            except Exception:
                self._failed = True
                self.close()
                raise
            else:
                if chunk:
                    self._buffer += chunk
                    return

    def _consume_preamble(self):
        marker = self._marker
        keep = len(marker) - 1
        while True:
            idx = self._buffer.find(marker)
            if idx == -1:
                if self._eof:
                    raise TruncatedMultipartException(
                        'Stream ended before the first boundary'
                    )
                if len(self._buffer) > keep:
                    del self._buffer[:len(self._buffer) - keep]
                self._read_more()
                continue
            end = idx + len(marker)
            while len(self._buffer) < end + 2 and not self._eof:
                self._read_more()
            following = bytes(self._buffer[end:end + 2])
            if following == b'\r\n':
                del self._buffer[:end + 2]
                self._state = 'headers'
                return
            if following == b'--':
                del self._buffer[:end + 2]
                self._state = 'done'
                return
            if self._eof:
                raise TruncatedMultipartException(
                    'Stream ended in the middle of a boundary'
                )
            # Boundary-like bytes in the preamble; discard one byte
            # and keep scanning.
            del self._buffer[:idx + 1]

    def _read_headers(self):
        terminator = self._HEADER_TERMINATOR
        while True:
            if len(self._buffer) >= 2 and \
                    bytes(self._buffer[:2]) == b'\r\n':
                # No headers: the empty line alone terminates the
                # header section.
                del self._buffer[:2]
                return b''
            idx = self._buffer.find(terminator)
            if idx != -1:
                raw = bytes(self._buffer[:idx])
                del self._buffer[:idx + len(terminator)]
                return raw
            if self._eof:
                raise TruncatedMultipartException(
                    'Stream ended in the middle of part headers'
                )
            self._read_more()

    def _finish_part(self, next_state):
        part = self._active_part
        if part is not None:
            part._exhausted = True
        self._active_part = None
        self._state = next_state

    def _read_body_chunk(self, size):
        marker = self._delimiter
        keep = len(marker) - 1
        while True:
            idx = self._buffer.find(marker)
            if idx != -1:
                if idx > 0:
                    emit_len = min(idx, size)
                    out = bytes(self._buffer[:emit_len])
                    del self._buffer[:emit_len]
                    return out
                end = len(marker)
                while len(self._buffer) < end + 2 and not self._eof:
                    self._read_more()
                following = bytes(self._buffer[end:end + 2])
                if following == b'\r\n':
                    del self._buffer[:end + 2]
                    self._finish_part('headers')
                    return b''
                if following == b'--':
                    del self._buffer[:end + 2]
                    self._finish_part('done')
                    return b''
                if self._eof:
                    raise TruncatedMultipartException(
                        'Stream ended in the middle of a boundary'
                    )
                # Boundary-like prefix inside the content: the marker
                # is not followed by CR-LF or '--', so it is content.
                emit_len = min(end, size)
                out = bytes(self._buffer[:emit_len])
                del self._buffer[:emit_len]
                return out
            if self._eof:
                raise TruncatedMultipartException(
                    'Stream ended before the closing boundary'
                )
            if len(self._buffer) > keep:
                emit_len = min(len(self._buffer) - keep, size)
                if emit_len > 0:
                    out = bytes(self._buffer[:emit_len])
                    del self._buffer[:emit_len]
                    return out
            self._read_more()

    def _read_from_part(self, size):
        if size is None or size < 0:
            chunks = []
            while True:
                chunk = self._read_body_chunk(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b''.join(chunks)
        if size == 0:
            return b''
        out = bytearray()
        while len(out) < size:
            chunk = self._read_body_chunk(size - len(out))
            if not chunk:
                break
            out += chunk
        return bytes(out)

    def _drain_active_part(self):
        part = self._active_part
        part._abandoned = True
        while not part._exhausted:
            self._read_body_chunk(65536)
