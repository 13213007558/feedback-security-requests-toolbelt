# -*- coding: utf-8 -*-
"""Collection of exceptions raised by requests-toolbelt."""


class StreamingError(Exception):
    """Used in :mod:`requests_toolbelt.downloadutils.stream`."""
    pass


class MultipartEncodingError(Exception):
    """Raised when a multipart body cannot satisfy its declared length.

    The length advertised by :class:`MultipartEncoder` is part of the request
    contract (e.g. the ``Content-Length`` header), so an input that ends early,
    stalls on a short read, grows beyond its reported size, or provides no
    length information at all must fail explicitly instead of looping forever
    or emitting a truncated body that still looks like a complete request.
    """
    pass


class VersionMismatchError(Exception):
    """Used to indicate a version mismatch in the version of requests required.

    The feature in use requires a newer version of Requests to function
    appropriately but the version installed is not sufficient.
    """
    pass


class RequestsVersionTooOld(Warning):
    """Used to indicate that the Requests version is too old.

    If the version of Requests is too old to support a feature, we will issue
    this warning to the user.
    """
    pass
