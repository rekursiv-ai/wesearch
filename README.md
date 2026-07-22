# wesearch

Web search, fetch, and paper-research toolkit.

`wesearch` is a synchronous, batteries-included library for programmatic
web access: run a search, fetch a page through a real-browser fingerprint,
extract content, and look up scholarly papers across multiple providers.
It is the web layer factored out of a larger agent stack, so it is built to
survive bot-detection, rate limits, and flaky endpoints without a running
browser in the common case.

## What's inside

```
wesearch/
├── search.py        search(...) over DuckDuckGo (+ configured backends);
│                      SearchResult / PaperResult / ImageResult
├── fetch/           the sole HTTP egress
│   ├── fetch.py     fetch(url, request=RequestParams(...)) -> (body, meta)
│   ├── curl.py      curl-cffi transport (TLS/JA3 browser impersonation)
│   ├── stdlib.py    dependency-free urllib transport
│   ├── zendriver.py opt-in real-Chrome backend for JS-gated pages
│   ├── transport_routing.py  per-domain transport selection
│   └── challenge.py bot-challenge detection and classification
├── chrome/          real-browser fingerprints
│   ├── headers.py   Chrome request headers (incl. x-browser-validation)
│   └── useragents.py  vendored, refreshable User-Agent pools
├── profile.py       cross-process per-(egress_ip, domain) cookie + UA jar
├── ratelimit.py     cross-process, per-domain rate limiting
├── scrape.py        get_element_content(html, selector)
├── errors.py        FetchError, BotDetectionError + subclasses
└── paper/           scholarly-paper lookup
    ├── search.py    search(...) across Semantic Scholar, OpenAlex, SearXNG
    ├── details.py   metadata / references / citations / cited_by
    ├── authors.py   author search and publication lists
    ├── fetch.py     PDF download cascade
    └── providers/   per-source backends (openalex, s2, searxng)
```

## Usage

```python
from wesearch.search import search
from wesearch.fetch import RequestParams, fetch
from wesearch.scrape import get_element_content

# Search
hits = search("denoising recursion models", limit=10)
for r in hits:
    print(r.title, r.url)

# Fetch + scrape
body, _meta = fetch("https://example.com", request=RequestParams(timeout_sec=10))
title = get_element_content(body.decode("utf-8"), "h1")

# Papers
from wesearch.paper.search import search as paper_search
from wesearch.paper.ids import normalize_id
from wesearch.paper.details import metadata

for rec in paper_search("attention is all you need", limit=5):
    print(rec.title, rec.year)
meta = metadata(*normalize_id("arXiv:1706.03762"))
```

Each name is imported from the submodule that defines it; the top-level
`__init__` re-exports nothing.

## Transports

`fetch` picks a transport per domain:

- **curl-cffi** (default): TLS/JA3 browser impersonation, no browser process.
- **stdlib**: dependency-free `urllib` fallback.
- **zendriver**: an opt-in headless-Chrome backend for JavaScript-gated
  pages, used only for domains that require it.

A persistent per-`(egress_ip, domain)` profile (cookies + User-Agent) is
loaded and saved transparently, and cross-process rate limiting paces
requests so concurrent workers stay under each site's threshold.

## Roadmap

An MCP server exposing `search` / `fetch` / paper-lookup as tools for
coding agents is planned, so an agent can call the toolkit directly.

## License

Apache-2.0.
