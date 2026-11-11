# -*- coding: utf-8 -*-
"""Contract tests for MultipartEncoder's declared length.

These tests verify that the length a MultipartEncoder advertises always
matches the number of bytes it actually streams, across the input shapes used
by a forensic upload client: plain files (possibly seeked to a non-zero
offset), BytesIO, streams exposing only ``len`` and streams that short-read.

When an input cannot honour its promised length (early EOF, a short read that
stalls, a length that changes, or no length information at all) the encoder
must raise MultipartEncodingError instead of looping forever or emitting a
truncated body that still looks like a complete request.

Verification goes through real encoding and chunked reads (including
``read(0)``), and the emitted body is parsed with the standard library's
independent MIME parser so each part's payload can be compared byte for byte
with the original data. No upload service is required.
"""
import io
import os
import random
import signal
import tempfile

import pytest
from email.parser import BytesParser
from email.policy import HTTP

from requests_toolbelt.multipart.encoder import (
    MultipartEncoder, MultipartEncoderMonitor,
)
from requests_toolbelt.exceptions import MultipartEncodingError

BOUNDARY = "length-contract-boundary"
DATA = random.Random(20261111).randbytes(10240)
TEXT = u"forensic note é"
CHUNK_SIZES = (0, 1, 2, 3, 7, 16, 64, 1024, 8192)


class ReadLoopTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise ReadLoopTimeout("encoder kept reading without terminating")


# --------------------------------------------------------------------------
# Input fixtures
# --------------------------------------------------------------------------
class LenOnlyStream(object):
    """A stream exposing length solely through a dynamic ``len`` attribute.

    It has neither ``__len__``, ``fileno`` nor ``getvalue``. ``chunk_cap``
    makes every read a short read; ``early_eof_at`` makes it return b'' while
    it still promises data.
    """

    def __init__(self, data, chunk_cap=None, early_eof_at=None):
        self._buf = io.BytesIO(data)
        self._total = len(data)
        self.chunk_cap = chunk_cap
        self.early_eof_at = early_eof_at
        self.read_calls = 0

    @property
    def len(self):
        return self._total - self._buf.tell()

    def read(self, size=-1):
        self.read_calls += 1
        if (self.early_eof_at is not None
                and self._buf.tell() >= self.early_eof_at):
            return b""
        if size is None or size < 0:
            return self._buf.read()
        if self.chunk_cap is not None:
            size = min(size, self.chunk_cap)
        return self._buf.read(size)


class StallingStream(object):
    """Promises bytes forever via ``len`` but returns b'' once stalled."""

    def __init__(self, data, stall_after):
        self._data = data
        self._pos = 0
        self._stall_after = stall_after

    @property
    def len(self):
        if self._pos < self._stall_after:
            return len(self._data) - self._pos
        return 5

    def read(self, size=-1):
        if self._pos >= self._stall_after:
            return b""
        if size is None or size < 0:
            size = len(self._data)
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


class ShrinkingStream(object):
    """Reports a fixed inflated length; real EOF arrives earlier."""

    def __init__(self, data, declared_len):
        self._buf = io.BytesIO(data)
        self._declared_len = declared_len

    @property
    def len(self):
        return self._declared_len - self._buf.tell()

    def read(self, size=-1):
        return self._buf.read(size)


class GrowingStream(object):
    """Understates its length, then delivers more bytes than promised."""

    def __init__(self, data, declared_len):
        self._buf = io.BytesIO(data)
        self._declared_len = declared_len

    @property
    def len(self):
        return max(0, self._declared_len - self._buf.tell())

    def read(self, size=-1):
        return self._buf.read(size)


class LateGrowingStream(object):
    """Reports zero only once the promised bytes are gone, then has more.

    Exercises the end-of-part probe: the declared budget is never exceeded on
    any single read, yet unreported data remains afterwards.
    """

    def __init__(self, promised, hidden):
        self._data = promised + hidden
        self._promised_len = len(promised)
        self._pos = 0

    @property
    def len(self):
        return max(0, self._promised_len - self._pos)

    def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._data)
        chunk = self._data[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk


class NoLengthStream(object):
    """Readable stream with no discoverable length whatsoever."""

    def __init__(self, data):
        self._buf = io.BytesIO(data)

    def read(self, size=-1):
        return self._buf.read(size)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def drain(encoder):
    """Read the whole encoder through varied chunk sizes (incl. read(0)).

    Fails the test if the encoder does not terminate promptly.
    """
    chunks = []
    produced = 0
    index = 0
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(5)
    try:
        while True:
            size = CHUNK_SIZES[index % len(CHUNK_SIZES)]
            index += 1
            piece = encoder.read(size)
            if size != 0 and not piece:
                break
            chunks.append(piece)
            produced += len(piece)
            assert produced <= encoder.len, "output exceeded declared length"
    finally:
        signal.alarm(0)
    return b"".join(chunks), produced


def parse_parts(body, boundary):
    """Independently parse the multipart body with the stdlib MIME parser."""
    content_type = (
        b"Content-Type: multipart/form-data; boundary=" + boundary.encode()
    )
    message = BytesParser(policy=HTTP).parsebytes(
        content_type + b"\r\n\r\n" + body
    )
    return {
        part.get_param("name", header="content-disposition"):
            part.get_payload(decode=True)
        for part in message.iter_parts()
    }


def assert_contract(fields, expected_payloads, monitored=False):
    encoder = MultipartEncoder(fields, boundary=BOUNDARY)
    if monitored:
        encoder = MultipartEncoderMonitor(encoder)
    declared = encoder.len
    body, produced = drain(encoder)

    assert produced == declared
    assert len(body) == declared

    parsed = parse_parts(body, BOUNDARY)
    assert set(parsed) == set(expected_payloads)
    for name, payload in expected_payloads.items():
        assert parsed[name] == payload
    return body


# --------------------------------------------------------------------------
# Happy path: declared length equals produced bytes for every input shape
# --------------------------------------------------------------------------
def test_plain_file_seeked_to_nonzero_offset():
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(DATA)
        with open(path, "rb") as handle:
            handle.seek(1000)
            expected = DATA[1000:]
            fields = [("file", ("evidence.bin", handle,
                                "application/octet-stream"))]
            body = assert_contract(fields, {"file": expected})
    finally:
        os.unlink(path)

    assert DATA[1000:] in body
    assert DATA[:1000] not in body


def test_bytesio_seeked_to_nonzero_offset_is_rebased():
    stream = io.BytesIO(DATA)
    stream.seek(2000)
    # BytesIO exposes getvalue(): the whole buffer is encoded from the start,
    # so the advertised length must match that full content.
    body = assert_contract([("file", ("evidence.bin", stream))],
                           {"file": DATA})
    assert DATA[:2000] in body


def test_stream_with_len_attribute():
    stream = LenOnlyStream(DATA)
    assert_contract(
        [("file", ("evidence.bin", stream)), ("note", TEXT)],
        {"file": DATA, "note": TEXT.encode("utf-8")},
    )


def test_short_reading_stream_that_keeps_returning_data():
    stream = LenOnlyStream(DATA, chunk_cap=37)
    assert_contract([("file", ("evidence.bin", stream))],
                    {"file": DATA})
    assert stream.read_calls > len(DATA) / 37.0


def test_monitor_reports_identical_length_and_payloads():
    stream = LenOnlyStream(DATA, chunk_cap=37)
    assert_contract(
        [("file", ("evidence.bin", stream)), ("note", TEXT)],
        {"file": DATA, "note": TEXT.encode("utf-8")},
        monitored=True,
    )


def test_read_zero_does_not_consume_input_or_fake_progress():
    stream = LenOnlyStream(DATA, chunk_cap=37)
    monitor = MultipartEncoderMonitor(
        MultipartEncoder([("file", ("evidence.bin", stream))],
                         boundary=BOUNDARY)
    )
    callbacks = []
    monitor.callback = lambda m: callbacks.append(m.bytes_read)

    assert monitor.read(0) == b""
    assert monitor.bytes_read == 0
    assert callbacks == [0]
    assert stream.read_calls == 0
    assert stream._buf.tell() == 0
    assert monitor.len == MultipartEncoder(
        [("file", ("evidence.bin", LenOnlyStream(DATA, chunk_cap=37)))],
        boundary=BOUNDARY,
    ).len

    body, produced = drain(monitor)
    assert produced == monitor.len
    assert monitor.bytes_read == monitor.len
    assert parse_parts(body, BOUNDARY)["file"] == DATA


def test_read_zero_on_bare_encoder_advances_nothing():
    stream = LenOnlyStream(DATA)
    encoder = MultipartEncoder([("file", ("evidence.bin", stream))],
                               boundary=BOUNDARY)
    assert encoder.read(0) == b""
    assert stream.read_calls == 0
    body, produced = drain(encoder)
    assert produced == encoder.len
    assert parse_parts(body, BOUNDARY)["file"] == DATA


# --------------------------------------------------------------------------
# Broken inputs: an explicit failure is required, never a loop or truncation
# --------------------------------------------------------------------------
def test_stream_without_any_length_fails_at_construction():
    with pytest.raises(MultipartEncodingError):
        MultipartEncoder([("file", ("evidence.bin", NoLengthStream(DATA)))],
                         boundary=BOUNDARY)


def test_early_eof_with_outstanding_promised_bytes_fails():
    stream = ShrinkingStream(DATA[:6000], declared_len=len(DATA))
    encoder = MultipartEncoder(
        [("file", ("evidence.bin", stream))], boundary=BOUNDARY
    )
    with pytest.raises(MultipartEncodingError):
        drain(encoder)


def test_stalling_short_read_fails_instead_of_looping():
    stream = StallingStream(DATA, stall_after=6000)
    encoder = MultipartEncoder(
        [("file", ("evidence.bin", stream))], boundary=BOUNDARY
    )
    with pytest.raises(MultipartEncodingError):
        drain(encoder)


def test_growing_input_overshooting_declared_length_fails():
    stream = GrowingStream(DATA + b"TAIL" * 100, declared_len=len(DATA))
    encoder = MultipartEncoder(
        [("file", ("evidence.bin", stream))], boundary=BOUNDARY
    )
    with pytest.raises(MultipartEncodingError):
        drain(encoder)


def test_late_growth_behind_declared_length_fails_via_end_probe():
    stream = LateGrowingStream(b"x" * 10, b"y" * 7)
    encoder = MultipartEncoder(
        [("file", ("evidence.bin", stream))], boundary=BOUNDARY
    )
    with pytest.raises(MultipartEncodingError):
        drain(encoder)


def test_monitor_propagates_length_contract_failure():
    stream = StallingStream(DATA, stall_after=4096)
    monitor = MultipartEncoderMonitor(MultipartEncoder(
        [("file", ("evidence.bin", stream))], boundary=BOUNDARY
    ))
    with pytest.raises(MultipartEncodingError):
        drain(monitor)


def test_failure_never_emits_more_than_the_declared_length():
    stream = ShrinkingStream(DATA[:3000], declared_len=len(DATA))
    encoder = MultipartEncoder(
        [("file", ("evidence.bin", stream))], boundary=BOUNDARY
    )
    emitted = 0
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(5)
    try:
        with pytest.raises(MultipartEncodingError):
            while True:
                piece = encoder.read(128)
                if not piece:
                    break
                emitted += len(piece)
    finally:
        signal.alarm(0)
    assert emitted < encoder.len
