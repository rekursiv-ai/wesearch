# Security Policy

## Why this file exists

wesearch fetches arbitrary URLs, impersonates real-browser TLS/HTTP
fingerprints, parses untrusted HTML and JSON, and can drive a headless
Chrome. Several of these paths can be pointed at attacker-controlled hosts,
reach internal network addresses, or process hostile documents. Security
reports need a private path so exploit details are not published before
review.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately by emailing hello@rekursiv.ai.

Include:

- Affected version or commit.
- Steps to reproduce.
- Expected impact.
- Any suggested mitigation.

Please do not open public issues for vulnerabilities until we have investigated and coordinated disclosure.

## Scope

Security reports are especially useful for:

- Server-side request forgery (SSRF): fetches reaching loopback, link-local,
  or private-range addresses, DNS-rebinding, or redirect chains that escape
  an intended host allowlist.
- Unsafe URL handling: scheme confusion (`file://`, `gopher://`), credential
  or header injection via crafted URLs, or open-redirect following.
- Untrusted-input parsing: crashes, resource exhaustion, or code paths
  triggered by hostile HTML/JSON/XML responses (including the paper providers'
  API payloads).
- Headless-browser risk: the opt-in Chrome (zendriver) backend executing
  attacker-controlled JavaScript in a context that leaks local state.
- Scraping ethics and safety: fingerprint/rate-limit logic that could be
  abused to evade site protections at scale.
- Exposure of local state (cookie/User-Agent profile jar, egress IP) in logs,
  errors, or fetched-content side channels.
- Dependency, packaging, or supply-chain concerns in the published wheel or
  its dependency set (curl-cffi, zendriver, beautifulsoup4, brotli, zstandard).
