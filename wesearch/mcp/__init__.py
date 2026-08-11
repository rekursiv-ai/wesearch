"""MCP surface: wesearch's tools exposed to agent clients over stdio.

Everything MCP-specific lives here, so the layers below it -- ``fetch``,
``search``, ``paper`` -- stay a plain Python library with no MCP dependency.
The ``mcp`` SDK is an optional extra, and confining it to this package is what
lets a consumer install wesearch without it.

Import each name from the submodule that defines it (this package's ``__init__``
re-exports nothing).
"""

from __future__ import annotations
