"""Typed overrides for the two ``lxml.html`` helpers this repo calls.

``lxml-stubs`` declares ``fromstring(..., **kw)`` with the kwargs unannotated
(``lxml-stubs/html/__init__.pyi:83``), which makes basedpyright report the whole
function as partially unknown and poisons the inferred type of its result. Only
the members used here are redeclared; ``__getattr__`` keeps the rest of the
module resolving through the upstream stubs.
"""

from typing import Any

from lxml.etree import _BaseParser, _Element

def fromstring(
    html: str | bytes,
    base_url: str = ...,
    parser: _BaseParser = ...,
    **kw: object,
) -> _Element: ...
def tostring(
    doc: _Element,
    pretty_print: bool = ...,
    include_meta_content_type: bool = ...,
    encoding: str | type[str] | None = ...,
    method: str = ...,
    with_tail: bool = ...,
    doctype: str | None = ...,
) -> str: ...
def __getattr__(name: str) -> Any: ...
