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
