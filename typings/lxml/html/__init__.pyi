"""Typed overrides for the two ``lxml.html`` helpers this repo calls.

``lxml-stubs`` declares ``fromstring(..., **kw)`` with the kwargs unannotated
(``lxml-stubs/html/__init__.pyi:83``), which makes basedpyright report the whole
function as partially unknown and poisons the inferred type of its result.

Deliberately NO ``__getattr__``: a catch-all returning ``Any`` would silence
type checking for every member this file does not name, and a local stub module
SHADOWS the upstream one rather than chaining to it -- so the fallback such a
hook appears to provide does not exist. A caller needing another ``lxml.html``
member declares it here.
"""

from typing import Literal, overload

from lxml.etree import _BaseParser, _Element

def fromstring(
    html: str | bytes,
    base_url: str = ...,
    parser: _BaseParser = ...,
    **kw: object,
) -> _Element: ...

# Two overloads, because the return type follows the encoding: lxml yields
# ``str`` only for the Unicode encodings and ``bytes`` otherwise. A bare
# ``-> str`` typed ``tostring(node)`` as text while it returned bytes.
@overload
def tostring(
    doc: _Element,
    *,
    encoding: type[str] | Literal["unicode"],
    pretty_print: bool = ...,
    include_meta_content_type: bool = ...,
    method: str = ...,
    with_tail: bool = ...,
    doctype: str | None = ...,
) -> str: ...
@overload
def tostring(
    doc: _Element,
    *,
    encoding: str | None = ...,
    pretty_print: bool = ...,
    include_meta_content_type: bool = ...,
    method: str = ...,
    with_tail: bool = ...,
    doctype: str | None = ...,
) -> bytes: ...
