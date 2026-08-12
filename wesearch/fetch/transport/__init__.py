"""Retrieval transports: how bytes are actually fetched.

One module per transport, selected by the ``transport`` field of
:class:`wesearch.types.params.PolicyParams`. ``curl`` impersonates a browser's
TLS fingerprint, ``stdlib`` is the dependency-free reference path, and
``zendriver`` drives a real headless Chrome for hosts that challenge the
header-only transports.

Deliberately not a facade: :mod:`wesearch.fetch.fetch` imports each
transport by its own path, so a caller reaching for one transport does not pay
the import cost of Chrome.
"""
