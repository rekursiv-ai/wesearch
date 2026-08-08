"""Cross-cutting type definitions for wesearch.

The root of the package's type system: these types depend only on standard
library leaves, and every other wesearch module depends on them. Nothing here
imports from ``fetch``, ``providers``, ``paper``, or ``search`` -- which is what
lets a transport, a provider, and an application all name the same vocabulary
without an import cycle.

This ``__init__`` does not flatten symbols into the package namespace. Reach
into the submodule that defines the name:

- ``wesearch.types.params`` -- ``RequestParams`` and its four groups
  (``Content``, ``Retry``, ``Observe``, ``Policy``), plus the ``Transport`` and
  ``Trust`` vocabularies.
"""

from __future__ import annotations
