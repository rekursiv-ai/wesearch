"""Unified HTTP fetch with selectable transport backends."""

from wesearch.fetch.challenge import classify_challenge, classify_http_error
from wesearch.fetch.common import (
    ValidatedHost,
    ValidatedHosts,
    pinned_host,
    public_host,
)
from wesearch.fetch.fetch import (
    FetchSession,
    egress_ip,
    fetch,
    last_known_egress_ip,
    on_egress_rotation,
    resolve_transport,
    set_last_egress_ip,
)
from wesearch.types.params import (
    ContentParams,
    Extractor,
    ObserveParams,
    PolicyParams,
    RequestParams,
    RetryParams,
    Transport,
    Trust,
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
    "Trust",
    "ValidatedHost",
    "ValidatedHosts",
    "classify_challenge",
    "classify_http_error",
    "egress_ip",
    "fetch",
    "last_known_egress_ip",
    "on_egress_rotation",
    "pinned_host",
    "public_host",
    "resolve_transport",
    "set_last_egress_ip",
]
