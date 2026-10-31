"""Cross-cutting type definitions for wesearch.

The root of the package's type system: these types depend only on standard
library leaves, and every other wesearch module depends on them. Nothing here
imports from ``fetch``, ``providers``, ``paper``, or ``search`` -- which is what
lets a transport, a provider, and an application all name the same vocabulary
without an import cycle.

This ``__init__`` does not flatten symbols into the package namespace. Reach
into the submodule that defines the name:

- ``wesearch.types.params`` -- ``RequestParams`` and its four groups
  (``ContentParams``, ``RetryParams``, ``ObserveParams``, ``PolicyParams``), plus the ``Transport``
  and ``Extractor`` vocabularies.
- ``wesearch.types.errors`` -- the exception hierarchy every layer raises
  and every caller catches.
- ``wesearch.types.urls`` -- ``without_userinfo``, which keeps a URL's
  credentials on the request and out of everything shown.
"""

from __future__ import annotations
