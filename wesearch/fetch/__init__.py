"""Unified HTTP fetch with selectable transport backends."""

from wesearch.fetch.challenge import classify_challenge, classify_http_error
from wesearch.fetch.common import ValidatedHost
from wesearch.fetch.fetch import (
    FetchSession,
    RequestParams,
    Transport,
    egress_ip,
    fetch,
    last_known_egress_ip,
    on_egress_rotation,
    resolve_transport,
    set_last_egress_ip,
)


__all__ = [
    "FetchSession",
    "RequestParams",
    "Transport",
    "ValidatedHost",
    "classify_challenge",
    "classify_http_error",
    "egress_ip",
    "fetch",
    "last_known_egress_ip",
    "on_egress_rotation",
    "resolve_transport",
    "set_last_egress_ip",
]
