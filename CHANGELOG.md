# Changelog

All notable wesearch changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

## 0.1.10 - 2026-08-11

### Added

- Selectable HTML extractors. `Policy.extractor` picks `trafilatura` (the
  default, article body only), `html2text` (every text node), `markdownify`
  (the document's elements, keeping nested structure), or `raw`. The
  article-shaped assumption behind trafilatura is wrong for a dictionary
  entry or a Q&A thread, where it returned plausible output with the
  substance missing.
- `fetch-zendriver URL` console script. A blocked fetch already told the user
  to run it; now a default install has it.
- `wesearch.paper.render`: one rendering of a paper or author record, text and
  structured. The tool and MCP surfaces each had their own, and they drifted.

### Changed

- Modules are grouped by what they do, so several import paths moved:
  `wesearch.errors` to `wesearch.types.errors`, `wesearch.search` to a package
  (`wesearch.search.search`, `.searxng`, `.duckduckgo`, `.custom_types`),
  `wesearch.providers` to `wesearch.fetch.providers`, the transports to
  `wesearch.fetch.transport`, and `wesearch.mcp_server` to
  `wesearch.mcp.server`.
- The cookie jar is per-origin rather than flat, so a redirect no longer
  leaks the first host's cookies to the second.

### Fixed

- `Retry` rejects a non-finite `timeout_sec`. NaN failed every comparison and
  so passed the range check, then reached the socket and browser timeout APIs
  and removed the ceiling the class exists to impose.
- `abstract_chars=0` -- the plainest way to ask for no abstract -- raises
  instead of returning the full one.

### Removed

- `search_web` from `wesearch.web`. Call `wesearch.search.search` directly;
  the wrapper only flattened typed results to dicts.
- The `extract` extra. Article extraction (trafilatura) and RSS/Atom parsing
  (defusedxml) are core dependencies, so a plain `pip install wesearch` now
  gets them. `wesearch[extract]` still installs -- an unknown extra is ignored,
  not an error -- but names nothing; drop the suffix.

## 0.1.8 - 2026-08-04

### Added

- Search results carry a completeness signal, so a caller can tell a
  truncated result set from an exhausted one.

### Fixed

- Paper fusion merges records by DOI and arXiv id rather than title, so the
  same paper indexed under two title spellings no longer appears twice.
- Title dedup no longer collapses distinct papers that share a title.
- The MCP server no longer exposes its tools on an unauthenticated surface.

## 0.1.7 - 2026-08-01

### Changed

- README carries a one-line description below the badges; PyPI renders the
  README, so the project page had been showing the previous text.

## 0.1.6 - 2026-08-01

### Changed

- README leads with a Quick Start; the duplicate Install section is folded
  into it, with `uv add` first and pip named as the alternative.

### Fixed

- PyPI and GitHub badge URLs no longer carry a `loop.` prefix, which made
  them dead links.

## 0.1.5 - 2026-07-28

- Initial public release of wesearch: a synchronous web search, fetch, and
  paper-research toolkit with real-browser TLS/HTTP fingerprints, a
  persistent per-`(egress_ip, domain)` cookie/User-Agent profile,
  cross-process rate limiting, an opt-in headless-Chrome backend, HTML
  scraping, and multi-provider scholarly-paper lookup.
