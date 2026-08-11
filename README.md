# wesearch🕸️

[![PyPI version](https://img.shields.io/pypi/v/wesearch.svg)](https://pypi.org/project/wesearch/)
[![CI](https://github.com/rekursiv-ai/wesearch/actions/workflows/package-validation.yml/badge.svg?branch=main)](https://github.com/rekursiv-ai/wesearch/actions/workflows/package-validation.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/discord/1530237005311639592?logo=discord&logoColor=white&label=Discord&color=5865F2)](https://discord.gg/2GZFPPvCqn)

A wascally web research toolkit for agents; web search, web fetch, and publication research.

## Quick Start

```bash
# Mac:
#   # Required for quick install.
#   brew install uv

# Ubuntu/Debian:
#   # Required for quick install.
#   sudo apt-get install -y curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh

uv add wesearch

# Alternatively: python -m pip install wesearch
```

`wesearch` is a synchronous, batteries-included library for programmatic web access: run a
search, fetch a page through a real-browser fingerprint, extract content, and look up scholarly
papers across multiple providers. It is the web layer factored out of a larger agent stack, so
it is built to survive bot-detection, rate limits, and flaky endpoints without a running browser
in the common case.

## Example

```python
from wesearch.search.search import search
from wesearch.fetch import RequestParams, fetch
from wesearch.scrape import get_element_content

# Web search (DuckDuckGo by default)
hits = search("denoising recursion models", num_results=10)
for r in hits:
    print(r.title, r.url)

# Fetch + scrape
body, _session = fetch("https://example.com", request=RequestParams(timeout_sec=10))
title = get_element_content(body.decode("utf-8"), "h1")

# Scholarly papers: Semantic Scholar + OpenAlex, reciprocal-rank-fused by default
from wesearch.paper.search import search as paper_search
from wesearch.paper.ids import normalize_id
from wesearch.paper.details import metadata

result = paper_search("attention is all you need", limit=5)
for rec in result.records:
    print(rec.title, rec.year)

meta = metadata(*normalize_id("arXiv:1706.03762"))
```

Each name is imported from the submodule that defines it; the top-level `__init__` re-exports
nothing.

## What's inside

```
wesearch/
├── types/           the vocabulary every layer shares; imports nothing internal
│   ├── params.py    RequestParams + Content/Retry/Observe/Policy; Transport,
│   │                  Extractor, Trust
│   ├── extractor.py the Extract protocol each extractor satisfies
│   └── errors.py    FetchError, BotDetectionError + subclasses
├── fetch/           the sole HTTP egress
│   ├── fetch.py     fetch(url, request=RequestParams(...)) -> (body, session)
│   ├── transport/   how bytes are retrieved
│   │   ├── curl.py      curl-cffi transport (TLS/JA3 browser impersonation)
│   │   ├── stdlib.py    dependency-free urllib transport
│   │   ├── zendriver.py opt-in real-Chrome backend for JS-gated pages
│   │   └── transport_routing.py  per-domain transport selection
│   ├── extractor/   how a fetched page becomes text
│   │   ├── html2text.py   every text node as Markdown (the default)
│   │   ├── markdownify.py the document's elements as Markdown
│   │   ├── trafilatura.py the scored article body only
│   │   └── raw.py         the source, untouched
│   ├── providers/   per-site fetch strategies (reddit, google_news, x, ...)
│   └── challenge.py bot-challenge detection and classification
├── search/          web search over pluggable backends
│   └── search.py    search(...) over SearXNG / DuckDuckGo / Google;
│                      SearchResult / PaperResult / ImageResult
├── paper/           scholarly-paper lookup
│   ├── search.py    search(...) across Semantic Scholar, OpenAlex, SearXNG
│   ├── details.py   metadata / references / citations
│   ├── authors.py   author search and publication lists
│   ├── fetch.py     PDF download cascade
│   └── providers/   per-source backends (openalex, s2, searxng)
├── mcp/             the MCP surface; the only place the mcp SDK is imported
│   └── server.py    wesearch-mcp, one tool per public function
├── chrome/          real-browser fingerprints
│   ├── headers.py   Chrome request headers (incl. x-browser-validation)
│   └── useragents.py  vendored, refreshable User-Agent pools
├── web.py           fetch_web(...): fetch + provider dispatch + extraction
├── profile.py       cross-process per-(egress_ip, domain) cookie + UA jar
├── ratelimit.py     cross-process, per-domain rate limiting
└── scrape.py        get_element_content(html, selector)
```

## Transports

`fetch` picks a transport per domain:

- **curl-cffi** (default): TLS/JA3 browser impersonation, no browser process.
- **stdlib**: dependency-free `urllib` fallback.
- **zendriver**: an opt-in headless-Chrome backend for JavaScript-gated pages, used only for
  domains that require it.

A persistent per-`(egress_ip, domain)` profile (cookies + User-Agent) is loaded and saved
transparently, and cross-process rate limiting paces requests so concurrent workers stay under
each site's threshold.

## API keys & configuration

Everything works keyless out of the box; the environment variables below raise your rate limits
or unlock a backend, and are all optional unless noted.

- `SEMANTIC_SCHOLAR_API_KEY` -- optional. Without it, Semantic Scholar lookups (`paper.search`,
  `paper.details`) share a low-rate public tier, and `paper.search(..., source="fused")` may
  return `complete=False` when S2 throttles the request. Set it for a higher-throughput tier.
- `OPENALEX_EMAIL` -- optional. Identifies you to OpenAlex's "polite pool" for more headroom than
  anonymous requests.
- `OPENALEX_API_KEY` -- optional. A higher OpenAlex request budget.
- `SEARXNG_URL` -- required only to use a `"searxng"` backend (`search(..., backend="searxng")`
  or `paper.search(..., source="searxng")`); the base URL of a SearXNG instance you control or
  trust.
- `WESEARCH_BROWSER_CONNECTION_TIMEOUT_SEC` -- optional; only affects the `"zendriver"` browser
  transport. Seconds to wait for Chrome's DevTools channel per launch attempt (default `1.0`,
  which fails fast when no usable browser exists so the `curl-then-zendriver` cascade stays
  snappy). Wrapper-launched browsers need much longer -- see the snap note below.

State (the cookie/User-Agent profile jar, cross-process rate-limit lockfiles, the browser
transport's persistent Chrome profile) is written under the OS's standard per-user data
directory -- `XDG_DATA_HOME` (or `~/.local/share`) on Linux, `~/Library/Application Support` on
macOS, `%LOCALAPPDATA%` on Windows -- namespaced per component (see `wesearch/lib/userdirs.py`).
No configuration file is required or read.

**Snap-packaged Chromium (stock Ubuntu) needs two accommodations** for the `"zendriver"`
transport, or it fails every launch with `BrowserUnavailableError`:

1. `WESEARCH_BROWSER_CONNECTION_TIMEOUT_SEC=30` -- the snap wrapper takes several seconds to
   expose DevTools, far beyond the 1-second default, and each retry restarts the browser.
2. `XDG_DATA_HOME` pointing at a **non-hidden** directory (e.g. `~/wesearch-data`) -- snap's
   AppArmor confinement silently blocks writes under hidden home paths like `~/.local/share`,
   so Chrome dies on its profile lock at every launch when the profile jar lives in the default
   location.

A non-snap Chrome or Chromium (e.g. Google Chrome's `.deb` on x86_64) needs neither.

## Development

```bash
uv sync --all-groups
```

Tests are tiered with pytest markers; the default run (`uv run pytest`) executes only the fast
unit tier:

```
addopts = -m 'not ci_smoke and not cuda and not integration and not performance and not cluster and not slow'
```

- `ci_smoke` -- slower package smoke tests, run explicitly in CI.
- `cuda` -- requires a real CUDA device.
- `integration` -- requires networking or external CLIs.
- `performance` -- timing-sensitive.
- `cluster` -- requires live cluster access.
- `slow` -- expensive local correctness tests (JIT, full fixtures, git, bash, a fresh interpreter).
- `real_llm` -- spawns a live LLM CLI; skipped unless `RUN_REAL_LLM=1`.

Run a specific tier with `uv run pytest -m integration`, or everything with
`uv run pytest -m ''`.

The `zendriver` backend tests need a Chrome or Chromium binary on `PATH`; without one they are
skipped automatically. No other system dependency is required to run the fast unit tier.

## MCP server

The toolkit is directly callable by coding agents over MCP:

```bash
pip install 'wesearch[mcp]'
wesearch-mcp                  # serves stdio; register it with your MCP client
```

For Claude Code: `claude mcp add --scope user wesearch -- wesearch-mcp`.

Tools exposed: `paper_search` (fused Semantic Scholar + OpenAlex),
`paper_details`, `paper_references`, `paper_citations`, `paper_pdf`
(downloads into the user cache and returns the path), `author_search`,
`author_papers`, `web_search`, and `web_fetch` (extracted page text, with an
opt-in headless-browser fallback for bot-walled sites). Outputs are
deliberately compact for model consumption; abstracts are truncated and
empty fields dropped. The server is synchronous and per-client — state that
must be shared (rate limits, cookie/UA profiles) is already cross-process
safe on disk, so no daemon is needed.

## See also

Sibling projects in the [rekursiv-ai](https://github.com/rekursiv-ai) family:

- [sagent](https://github.com/rekursiv-ai/sagent) — The self-mutating multi-provider coding-agent CLI and typed Python library.
- [trackinizer](https://github.com/rekursiv-ai/trackinizer) — Centralized agent database for tracking inquiries, work, and the evidence behind conclusions.
- [madcatter](https://github.com/rekursiv-ai/madcatter) — Rich-based Markdown renderer for the terminal; ships the `mdcat` CLI.
- [priml](https://github.com/rekursiv-ai/priml) — Composable PyTorch building blocks: models, optimizers, losses, and a step-based training loop.
- [configgle](https://github.com/rekursiv-ai/configgle) — Hierarchical experiment configuration in typed pure-Python dataclasses instead of YAML.
- [copybarista](https://github.com/rekursiv-ai/copybarista) — Bidirectional source sync for publishing OSS-ready trees from a monorepo.
- [sudoku](https://github.com/rekursiv-ai/sudoku) — Sudoku-Extreme solved end to end with a 7M-parameter recursive transformer.

## Citing

If you find our work useful, please consider citing:

```bibtex
@misc{rekursivai2026wesearch,
      title={Wesearch - A wascally web research toolkit for agents.},
      author={Joshua V. Dillon},
      year={2026},
      howpublished={Github},
      url={https://github.com/rekursiv-ai/wesearch},
}
```
