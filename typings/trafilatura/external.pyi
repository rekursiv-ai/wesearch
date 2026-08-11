"""Typed override for the one ``trafilatura.external`` helper this repo calls.

trafilatura ships no ``py.typed``, so every symbol it exports resolves to
Unknown and poisons the inferred type of anything derived from it. Declaring
only what is used keeps the checker's knowledge intact without vendoring a
stub for the whole package.
"""

from lxml.etree import _Element

def try_readability(htmlinput: _Element) -> _Element: ...
