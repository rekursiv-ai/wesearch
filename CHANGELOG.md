# Changelog

All notable wesearch changes are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.1.12 - 2026-10-03

### Added

- `fetch_zendriver` takes `trust` and `max_redirects`, and `BrowserResult`
  reports `final_url`, where the navigation actually ended. Chrome follows
  redirects itself, so the requested URL is not where the cookies belong.
- `searxng` takes `retries`, defaulting to one. The edge in front of an
  instance throttles a burst with a 429 and a `Retry-After`; the retry
  waits that out (capped at 30 s) instead of surfacing a hard failure.
- `wesearch.lib.zstd_compat`, one zstandard interface that uses the
  standard library's `compression.zstd` on Python 3.14 and the `zstandard`
  package below it, and `wesearch.lib.absent`, the `ABSENT` sentinel.

### Changed

- **Breaking:** `BrowserUnavailableError` subclasses `OSError` instead of
  `RuntimeError`. A host with no usable Chrome now surfaces from `search`
  as a `SearchError`, and a fused paper search loses that one backend
  rather than aborting after Semantic Scholar had already answered. Code
  that caught it as `RuntimeError` must catch `OSError` or the class.
- **Breaking:** `wesearch.lib.custom_json` replaces its loose helpers with
  per-type codecs, and the old names are gone, not aliased. `str_val`,
  `int_val`, `float_val`, `bool_val`, `datetime_val`, `dict_val`, and
  `list_val` become `StrCodec.coerce`, `IntCodec.coerce`,
  `FloatCodec.coerce`, `BoolCodec.coerce`, `DatetimeCodec.coerce`,
  `DictCodec.coerce`, and `ListCodec.coerce`; `dicts_val` becomes
  `ListCodec.mappings`; `dataclass_to_json` and `dataclass_from_json`
  become `DataclassCodec.to_json` and `DataclassCodec.from_json`. The
  `JsonCodec` mixin is removed. `decode_or_none` reads a field whose
  absence matters without collapsing it to a default.
- **Breaking:** the fetch test doubles (`StubSession`, `StubCookies`,
  `const_curl_session`, ...) moved from `wesearch.fetch.test_helpers` to
  `wesearch.fetch.testing`, with no alias.
- **Breaking:** `wesearch.paper.errors.translate_http_error` renames its
  `not_found_on_404` keyword to `treat_404_as_missing`.
- A Cloudflare-fronted error counts as a bot wall only when Cloudflare
  says so: challenge markup in the body, or a `cf-mitigated: challenge`
  header. A 403, 429, or 503 that merely carries `server: cloudflare` or
  `cf-ray` -- an origin's own error, an expired API token, a site's rate
  limit -- now raises a plain `FetchError`. Read as a challenge, it sent
  the request to the browser, which wraps a JSON API's answer in HTML, and
  marked the domain browser-only for good.
- `fetch_web` falls back to the reader proxy only for a proven bot wall
  (`BotDetectionError`) or a 429, failures a different egress can clear. A
  bare 403 or 503 now raises rather than sending the URL to a third party.
- Each MCP tool's description documents its parameters and the fields it
  returns, and a test holds both to what the tool actually accepts and
  sends.
- The bundled desktop and Android Chrome user-agent pools are refreshed
  from the upstream dataset.
- Development: type checking uses a bundled, patched typeshed
  (`packages/rekursiv-ai-typeshed`), which `uv sync` installs as a path
  dependency, and the ty and basedpyright floors are raised. Runtime
  dependencies and the supported Python versions (3.12+) are unchanged.

### Fixed

- The browser transport validates every document request -- the
  navigation and each redirect hop Chrome follows on its own -- before
  Chrome connects, under the default `trust="untrusted"`. Only the
  caller's URL was checked, so a public page could redirect Chrome to
  loopback or a cloud metadata address. `on_redirect` now fires for each
  hop before it is followed, so raising from it aborts the hop.
- A browser fetch follows no more redirect hops than
  `RetryParams.max_redirects` allows (10 by default), as the other
  transports do. Chrome applied its own limit whatever the caller set,
  including 0, which promises no redirects at all.
- A cross-origin redirect drops `Authorization`, not just `Cookie` and the
  client hints, on every transport. The browser transport attaches a
  caller's `Cookie` and `Authorization` only to requests for the original
  origin; it used to install them tab-wide, so every redirect target
  received them.
- One-shot curl requests connect to the address that passed validation,
  as session requests already did, closing a DNS-rebinding gap.
- Origins compare in canonical form, so `https://User@EXAMPLE.com:443/` and
  `https://example.com/` share one cookie jar, and a hop between them no
  longer reads as cross-origin and loses its credentials.
- Cookies harvested by a browser fetch that redirected are stored under
  the domain it landed on, not the one requested.
- A browser fetch of JSON or plain text returns the payload instead of the
  viewer page Chrome generates around it, which callers rejected as a
  malformed response the server never sent.
- A SearXNG instance answering with something other than JSON raises a
  `SearchError` that names what answered -- Cloudflare's rate-limit page
  (error 1015), another HTML page, or an unparseable body -- instead of a
  bare `JSONDecodeError` at line 1, column 1.
- `fetch` runs the caller's `body_validator` on `raw_headers` requests too.
  DuckDuckGo's bot challenge therefore raises `PuzzleChallengeError`
  instead of parsing to an empty result list, and an empty parse logs
  whether it was a challenge, no results, or changed markup.
- Challenge detection reads markers only where a page renders them. A
  `<script>` or `<style>` body that spells out a CAPTCHA widget, or a link
  to a path containing `data-sitekey`, no longer marks an ordinary page as
  a challenge; Cloudflare's marker on a `<script>` tag's own attributes is
  still found.
- MCP tools report the library's error to the client as
  `<ErrorType>: <message>`. The MCP SDK forwards only `ToolError`, so a
  missing `SEARXNG_URL`, a DuckDuckGo CAPTCHA, and a bad DOI all arrived as
  "Error executing tool", and a bot wall's recovery guidance never did.
- A browser fetch that times out while Chrome is still answering no
  longer leaves the pooled browser deaf to every later fetch; the late
  reply killed the connection's listener.
- Browser teardown is bounded. Closing a tab gives up after 5 s, and
  shutdown at exit spends one 5 s budget across every pooled browser,
  where a wedged connection used to hang it. A browser that does not stop
  in time is killed, so it no longer holds its profile lock and fails the
  next launch. A fetch's whole wait is capped at `timeout_sec` plus 30 s.
- On macOS the browser transport launches Chrome for Testing when a
  Puppeteer or Playwright download provides one. Headless stock Chrome
  registers as the user's Chrome and swallowed every link opened from
  other applications while a fetch ran. Without one, Chrome resolves as
  before.
- `fetch-zendriver` on macOS finds and closes a Chrome another process
  left running on the fetch profile, as it already did on Linux.
- A `zstd` response body is decoded across all of its frames rather than
  only the first, and a truncated frame is a decoding failure.
- Year filters on author papers and citations no longer read a JSON `true`
  as the year 1. SearXNG map results report a non-finite coordinate as
  unknown, and an absurdly long citation count on a science result no
  longer raises out of the parse.
- `drive_chrome` passes `--password-store=basic`, so headless Chrome no
  longer hangs on a locked desktop keyring, and it skips a snap-packaged
  Chromium, whose confinement ignores the throwaway profile.
- Fused paper search folds case fully when it matches records by title,
  DOI or arXiv id, so titles that differ only as `ß` and `ss`, or by a
  ligature such as `ﬁ`, merge into one record.
- An OpenAlex search or citation listing with `limit=0` returns no
  records instead of failing with OpenAlex's HTTP 400, so a fused search
  no longer reports OpenAlex as incomplete.

### Removed

- **Breaking:** the `mitigation_statuses` keyword of
  `wesearch.fetch.challenge.classify_http_error`. A challenge is now
  recognized by Cloudflare's own evidence rather than by status, so there
  is no status set left to configure.

## 0.1.11 - 2026-08-19

### Added

- `wesearch.types.schema`: `Field` and `Schema`, one declaration of a
  parameter's type, default, and prose that every surface renders instead
  of restating. A parameter used to be spelled once per surface -- a
  JSON-Schema property, a validation branch, a signature default -- and the
  copies drifted: the MCP tool kept a `browser` bool after the library grew
  a five-valued `transport`, and both descriptions omitted the
  `social media` category the type had always accepted.
- `wesearch.fetch.custom_types` (`FetchParamsSchema`,
  `FetchBodyParamsSchema`) and `wesearch.search.SearchParamsSchema` declare
  the fetch and search parameters on that mechanism, so a tool adapter and
  the MCP server can no longer document the same knob differently.
- `wesearch.search.render`: one rendering of a search result, as text
  (`format_result`) and as a lean dict (`lean_result`), mirroring
  `wesearch.paper.render`. Both read one field table, so a category's extra
  structure cannot reach one surface and not the other.
- `NO_BODY` in `wesearch.types.params`, the sentinel `ContentParams.json`
  now defaults to. `None` cannot mean both "no body" and "the JSON literal
  `null`", so `json=None` used to send nothing and left `null` unsendable
  even where an API requires exactly that.

### Changed

- The four parameter groups are renamed with a `Params` suffix: `Content`
  to `ContentParams`, `Retry` to `RetryParams`, `Observe` to
  `ObserveParams`, and `Policy` to `PolicyParams`. The old names are gone,
  not aliased, so every construction site needs updating.
- `wesearch.web.WebFetchResult` is renamed to `FetchResult`, also without
  an alias.
- The default extractor is `html2text` rather than `trafilatura`. Measured
  by `wesearch/scripts/compare_extractors.py` over an 11-page corpus of the
  shapes that break extractors, `html2text` loses 0 of 37 content probes
  and `trafilatura` loses 12 -- one answer of a StackOverflow thread, most
  of a profile timeline, a dictionary entry's pronunciation -- and nothing
  in trafilatura's `str | None` return signals the loss. Ask for
  `trafilatura` deliberately on a page whose substance is one contiguous
  prose body, where it is both smaller and lossless.
- `search` resolves an omitted `backend` from the category: a non-`general`
  tab exists only on SearXNG, so asking for one is asking for that backend.
  A category named alongside an explicit non-SearXNG backend still raises
  rather than silently overriding the stated choice. `backend` is now
  optional in every overload, so a category still narrows the return type
  without naming a backend.
- The MCP `web_search` tool returns each result's category fields (a
  paper's authors, DOI, and citation count; an image's resolution; a
  torrent's seeders) instead of flattening every hit to
  `url`/`title`/`snippet`. A client could ask for the science tab and
  receive none of what makes it a science result.
- `wesearch.lib.userdirs` returns BASE directories: `data_dir()`,
  `config_dir()`, `cache_dir()`, and `state_dir()` take no application
  name, and the caller joins its own namespace segment
  (`data_dir() / "rekursiv-ai"`). On-disk locations are unchanged.
  `resolve_working_dir` is removed.
- `SearxngCategory` is defined in `wesearch.search.custom_types` rather
  than `wesearch.search.searxng`, so a caller can name a category without
  importing the backend. Both import paths still resolve.
- `wesearch.lib.custom_json.dataclass_from_json` raises `SchemaError` (a
  `ValueError` subclass) on a key that names no settable field. Dropping
  the key turned every misspelling into a silent default: a caller reading
  `{"min_digit": 7}` got the default and no signal.

### Fixed

- `fetch_web` sends a form POST again. Its `json_body` still defaulted to
  `None`, which `NO_BODY` had just redefined as the JSON literal `null`, so
  every call passing `form_body` presented two mutually exclusive bodies and
  raised before reaching the transport.
- Headless-Chrome browsers are closed when the process exits. Nothing else
  ever closed one, and each is roughly 70 MB across some 17 processes, so
  an ordinary exit left them resident -- measured at 378 processes holding
  27.5 GiB, 225 of them outliving the session that spawned them.
- The stdlib transport no longer seeds a stored or pool-drawn `User-Agent`
  over the header set built for the impersonation target. The UA named one
  browser while the `sec-ch-ua` beside it named another, which is a
  stronger bot signal than any single header value.
- Facebook in-app browser strings are excluded from the user-agent pool. A
  desktop TLS fingerprint carrying an in-app WebView UA does not occur in
  the wild.
- `RandomUniformPacer` reserves each caller's slot under a lock before
  sleeping. Concurrent callers all read the same last-grant time and fired
  together, which is exactly the fixed-cadence burst the pacer exists to
  avoid.
- `FileStore` opens its descriptor per transaction. `flock` is
  per-open-file-description, so a cached descriptor excluded neither
  threads sharing it nor a forked child inheriting it -- voiding
  cross-process exclusion -- and leaked a descriptor per short-lived store,
  which `clear_domain_cooldowns` creates one of per matching file.

### Removed

- `CapturedRequest` and `capture_chrome_request` from
  `wesearch.chrome.capture`, replaced by `drive_chrome`. Parity tests read
  the request off the server that received it rather than parsing Chrome's
  net-log bookkeeping.

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
