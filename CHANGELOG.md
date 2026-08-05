# Changelog

All notable wesearch changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

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
