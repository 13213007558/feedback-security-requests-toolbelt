# -*- coding: utf-8 -*-
import hashlib
import tracemalloc
import unittest

try:
    from unittest import mock
except ImportError:
    import mock

import pytest
import requests

from requests_toolbelt.multipart.decoder import (
    MultipartStreamException,
    StreamingMultipartDecoder,
    TruncatedMultipartException,
)


BOUNDARY = 'test-boundary'
CONTENT_TYPE = 'multipart/mixed; boundary="{}"'.format(BOUNDARY)
MARKER = b'--' + BOUNDARY.encode('ascii')

BINARY_256 = bytes(bytearray(range(256)))

# Content full of boundary-like prefixes that must be delivered verbatim.
TRICKY_CONTENT = (
    b'prefix --test-boundary mid-line\r\n'
    b'\r\n--test-boundar\r\n'
    b'\r\n--test-boundaryX\r\n'
    b'\r\n--test-boundary-extra\r\n'
    + BINARY_256 * 2 +
    b'\r\n--test-b'
)

PARTS = [
    ([(b'Content-Type', b'text/plain'), (b'X-Custom', b'value 1')],
     b'Hello, world!\r\nSecond line'),
    ([], TRICKY_CONTENT),
    ([(b'Content-Type', b'application/octet-stream')], BINARY_256 * 8),
    ([(b'Content-Disposition', b'form-data; name="empty"')], b''),
]


def encode_parts(parts, preamble=b'', epilogue=b'', closing_crlf=True):
    out = [preamble]
    for headers, content in parts:
        out.append(MARKER + b'\r\n')
        for key, value in headers:
            out.append(key + b': ' + value + b'\r\n')
        out.append(b'\r\n')
        out.append(content)
        out.append(b'\r\n')
    out.append(MARKER + b'--')
    if closing_crlf:
        out.append(b'\r\n')
    out.append(epilogue)
    return b''.join(out)


def chunk_iter(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


class ShortReadStream(object):
    """File-like object whose read() never returns more than max_read."""

    def __init__(self, data, max_read):
        self._data = data
        self._pos = 0
        self._max = max_read

    def read(self, size=-1):
        if self._pos >= len(self._data):
            return b''
        if size is None or size < 0:
            size = self._max
        n = min(size, self._max, len(self._data) - self._pos)
        chunk = self._data[self._pos:self._pos + n]
        self._pos += n
        return chunk


def decode_all(body, chunk_size, read_size=4096):
    decoder = StreamingMultipartDecoder(
        chunk_iter(body, chunk_size), CONTENT_TYPE
    )
    decoded = []
    for part in decoder:
        decoded.append((part.headers, b''.join(part.iter_content(read_size))))
    return decoded


def assert_parts_match(decoded, expected=PARTS):
    assert len(decoded) == len(expected)
    for (headers, content), (exp_headers, exp_content) in zip(decoded,
                                                              expected):
        assert content == exp_content
        assert len(headers) == len(exp_headers)
        for key, value in exp_headers:
            assert headers[key] == value


class TestStreamingMultipartDecoder(unittest.TestCase):
    def test_byte_exact_parts_across_chunk_sizes(self):
        body = encode_parts(
            PARTS,
            preamble=b'preamble junk\r\n',
            epilogue=b'epilogue junk\r\n',
        )
        for chunk_size in [1, 2, 3, 5, 7, 13, 64, 1024, len(body)]:
            decoded = decode_all(body, chunk_size)
            assert_parts_match(decoded)

    def test_byte_exact_parts_across_read_sizes(self):
        body = encode_parts(PARTS)
        for read_size in [1, 2, 3, 5, 17, 100, 65536]:
            decoded = decode_all(body, 64, read_size=read_size)
            assert_parts_match(decoded)

    def test_short_read_file_like_stream(self):
        body = encode_parts(PARTS, preamble=b'junk\r\n')
        for max_read in [1, 3, 7]:
            stream = ShortReadStream(body, max_read)
            decoder = StreamingMultipartDecoder(stream, CONTENT_TYPE)
            decoded = [
                (part.headers, b''.join(part.iter_content(11)))
                for part in decoder
            ]
            assert_parts_match(decoded)

    def test_preamble_with_boundary_lookalikes(self):
        preamble = (
            b'This preamble mentions --test-boundary mid-line\r\n'
            b'--test-boundaryZ\r\n'
            b'--test-boundar\r\n'
        )
        body = encode_parts(PARTS[:1], preamble=preamble)
        decoded = decode_all(body, 5)
        assert_parts_match(decoded, PARTS[:1])

    def test_epilogue_is_not_consumed(self):
        epilogue = b'epilogue --test-boundary junk\r\n' * 100
        body = encode_parts(PARTS[:1], epilogue=epilogue)
        pulled = []

        def tracking():
            for chunk in chunk_iter(body, 64):
                pulled.append(chunk)
                yield chunk

        decoder = StreamingMultipartDecoder(tracking(), CONTENT_TYPE)
        decoded = [
            (part.headers, b''.join(part.iter_content(512)))
            for part in decoder
        ]
        assert_parts_match(decoded, PARTS[:1])
        # The decoder stopped at the closing boundary instead of
        # buffering the whole body.
        assert len(pulled) * 64 < len(body)

    def test_closing_boundary_without_trailing_crlf(self):
        body = encode_parts(PARTS, closing_crlf=False)
        decoded = decode_all(body, 9)
        assert_parts_match(decoded)

    def test_empty_body_has_no_parts(self):
        body = MARKER + b'--\r\n'
        assert decode_all(body, 3) == []

    def test_read_size_is_respected(self):
        body = encode_parts(PARTS)
        decoder = StreamingMultipartDecoder(chunk_iter(body, 128),
                                            CONTENT_TYPE)
        part = next(iter(decoder))
        collected = []
        while True:
            chunk = part.read(5)
            assert len(chunk) <= 5
            if not chunk:
                break
            collected.append(chunk)
        assert b''.join(collected) == PARTS[0][1]

    def test_content_and_text_properties(self):
        body = encode_parts(PARTS[:1])
        decoder = StreamingMultipartDecoder(chunk_iter(body, 16),
                                            CONTENT_TYPE)
        part = next(iter(decoder))
        assert part.text == PARTS[0][1].decode('utf-8')
        assert part.read() == b''

    def test_advancing_discards_unread_remainder(self):
        body = encode_parts(PARTS)
        decoder = StreamingMultipartDecoder(chunk_iter(body, 32),
                                            CONTENT_TYPE)
        parts = iter(decoder)
        first = next(parts)
        assert first.read(5) == PARTS[0][1][:5]
        second = next(parts)
        assert b''.join(second.iter_content(10)) == PARTS[1][1]

    def test_reading_part_after_advancing_raises(self):
        body = encode_parts(PARTS)
        decoder = StreamingMultipartDecoder(chunk_iter(body, 32),
                                            CONTENT_TYPE)
        parts = iter(decoder)
        first = next(parts)
        first.read(5)
        next(parts)
        with pytest.raises(MultipartStreamException):
            first.read(1)

    def test_fully_read_part_returns_empty_bytes(self):
        body = encode_parts(PARTS)
        decoder = StreamingMultipartDecoder(chunk_iter(body, 32),
                                            CONTENT_TYPE)
        part = next(iter(decoder))
        part.read()
        assert part.read(10) == b''
        assert list(part.iter_content(10)) == []

    def test_truncated_mid_body(self):
        body = encode_parts(PARTS[:1])[:-20]
        decoder = StreamingMultipartDecoder(chunk_iter(body, 8),
                                            CONTENT_TYPE)
        part = next(iter(decoder))
        with pytest.raises(TruncatedMultipartException):
            part.read()

    def test_truncated_mid_headers(self):
        body = MARKER + b'\r\nContent-Type: text/pla'
        decoder = StreamingMultipartDecoder(chunk_iter(body, 4),
                                            CONTENT_TYPE)
        with pytest.raises(TruncatedMultipartException):
            next(iter(decoder))

    def test_truncated_no_boundary_at_all(self):
        decoder = StreamingMultipartDecoder(chunk_iter(b'junk\r\n' * 10, 3),
                                            CONTENT_TYPE)
        with pytest.raises(TruncatedMultipartException):
            next(iter(decoder))

    def test_truncated_mid_closing_boundary(self):
        body = encode_parts(PARTS[:1])[:-3]  # drop the final "--\r\n"
        decoder = StreamingMultipartDecoder(chunk_iter(body, 7),
                                            CONTENT_TYPE)
        part = next(iter(decoder))
        with pytest.raises(TruncatedMultipartException):
            part.read()

    def test_underlying_failure_propagates_and_closes(self):
        body = encode_parts(PARTS)

        def failing():
            yield body[:40]
            yield body[40:80]
            raise ConnectionError('boom')

        decoder = StreamingMultipartDecoder(failing(), CONTENT_TYPE)
        parts = iter(decoder)
        part = next(parts)
        with pytest.raises(ConnectionError):
            part.read()
        with pytest.raises(StopIteration):
            next(parts)
        with pytest.raises(MultipartStreamException):
            part.read(1)

    def test_close_releases_response_without_draining(self):
        body = encode_parts(PARTS)
        response = mock.NonCallableMagicMock(spec=requests.Response)
        response.headers = {'content-type': CONTENT_TYPE}
        response.iter_content = lambda n: chunk_iter(body, n)
        decoder = StreamingMultipartDecoder.from_response(response,
                                                          chunk_size=16)
        part = next(iter(decoder))
        part.read(4)
        decoder.close()
        assert response.close.called
        with pytest.raises(StopIteration):
            next(iter(decoder))
        with pytest.raises(MultipartStreamException):
            part.read(1)

    def test_context_manager_closes_response(self):
        body = encode_parts(PARTS)
        response = mock.NonCallableMagicMock(spec=requests.Response)
        response.headers = {'content-type': CONTENT_TYPE}
        response.iter_content = lambda n: chunk_iter(body, n)
        with StreamingMultipartDecoder.from_response(response) as decoder:
            part = next(iter(decoder))
            part.read(4)
        assert response.close.called

    def test_from_response_iterates_all_parts(self):
        body = encode_parts(PARTS, preamble=b'junk\r\n')
        response = mock.NonCallableMagicMock(spec=requests.Response)
        response.headers = {'content-type': CONTENT_TYPE}
        response.iter_content = lambda n: chunk_iter(body, n)
        with StreamingMultipartDecoder.from_response(response,
                                                     chunk_size=6) as dec:
            decoded = [
                (part.headers, b''.join(part.iter_content(9)))
                for part in dec
            ]
        assert_parts_match(decoded)

    def test_from_response_failure_mid_stream(self):
        body = encode_parts(PARTS)

        def failing():
            yield body[:80]
            raise requests.exceptions.ChunkedEncodingError('broken')

        response = mock.NonCallableMagicMock(spec=requests.Response)
        response.headers = {'content-type': CONTENT_TYPE}
        response.iter_content = lambda n: failing()
        decoder = StreamingMultipartDecoder.from_response(response)
        part = next(iter(decoder))
        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            part.read()
        assert response.close.called


class TestStreamingMultipartDecoderMemory(unittest.TestCase):
    @staticmethod
    def big_body(payload_size):
        payload = (BINARY_256 * (payload_size // 256 + 1))[:payload_size]
        body = encode_parts(
            [([(b'Content-Type', b'application/octet-stream')], payload)],
        )
        return body, payload

    @staticmethod
    def decode_and_measure(body, stream_chunk, read_chunk):
        pulled_max = [0]
        decoder = StreamingMultipartDecoder(
            chunk_iter(body, stream_chunk), CONTENT_TYPE
        )
        digest = hashlib.md5()
        total = [0]

        def consume():
            for part in decoder:
                for chunk in part.iter_content(read_chunk):
                    digest.update(chunk)
                    total[0] += len(chunk)
                    pulled_max[0] = max(pulled_max[0], len(decoder._buffer))

        tracemalloc.start()
        try:
            consume()
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return peak, pulled_max[0], total[0], digest.hexdigest()

    def test_peak_memory_tracks_chunk_size_not_attachment_size(self):
        stream_chunk = 64 * 1024
        read_chunk = 8 * 1024
        sizes = [256 * 1024, 1024 * 1024, 4 * 1024 * 1024]
        peaks = {}
        for size in sizes:
            body, payload = self.big_body(size)
            peak, buf_max, total, digest = self.decode_and_measure(
                body, stream_chunk, read_chunk
            )
            assert total == size
            assert digest == hashlib.md5(payload).hexdigest()
            # The decoder's internal buffer never exceeds roughly one
            # upstream chunk plus the delimiter length.
            assert buf_max <= stream_chunk + 2 * len(MARKER) + 8
            peaks[size] = peak

        # Peak memory stays essentially flat as the attachment grows
        # 16x: it must not scale with the total attachment size.
        assert peaks[sizes[-1]] < peaks[sizes[0]] + 256 * 1024
        assert peaks[sizes[-1]] < 1024 * 1024

        # Reading the same largest attachment with a 16x bigger read
        # chunk raises the peak proportionally.
        body, payload = self.big_body(sizes[-1])
        big_peak, _, _, _ = self.decode_and_measure(
            body, stream_chunk, 16 * read_chunk
        )
        assert big_peak > 2 * peaks[sizes[-1]]

        print('\npeak buffer by attachment size (read chunk {} B):'
              .format(read_chunk))
        for size in sizes:
            print('  attachment {:>8} B -> peak {:>8} B'
                  .format(size, peaks[size]))
        print('  attachment {:>8} B -> peak {:>8} B (read chunk {} B)'
              .format(sizes[-1], big_peak, 16 * read_chunk))
