"""Fetch a URL like a browser."""

from wesearch.fetch.challenge import classify_challenge, classify_http_error
from wesearch.fetch.fetch import egress_ip, fetch, resolve_transport
from wesearch.fetch.types import FetchSession
from wesearch.types.params import (
    ContentParams,
    Extractor,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
)


__all__ = [
    "ContentParams",
    "Extractor",
    "FetchSession",
    "ObserveParams",
    "PolicyParams",
    "RequestParams",
    "RetryParams",
    "Transport",
    "classify_challenge",
    "classify_http_error",
    "egress_ip",
    "fetch",
    "resolve_transport",
]
