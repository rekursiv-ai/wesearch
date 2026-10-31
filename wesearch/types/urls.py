"""URL credentials: where a ``user:pass@`` URL's secret may and may not go."""

from __future__ import annotations

from urllib.parse import unquote, urlparse

import base64


__all__ = ["without_userinfo"]


def without_userinfo(url: str) -> tuple[str, str]:
    """Return ``url`` without ``user:pass@`` and the Basic credential it carried.

    The credential belongs on the request alone; the bare URL is what a
    message, a log, a link, or a file may show.

    Args:
      url: Any URL.

    Returns:
      bare: ``url`` with its userinfo removed.
      authorization: ``"Basic ..."`` for the userinfo, or ``""`` when it had none.

    """
    parsed = urlparse(url)
    username, password = parsed.username or "", parsed.password or ""
    if not (username or password):
        return url, ""
    pair = f"{unquote(username)}:{unquote(password)}"
    netloc = parsed.netloc[parsed.netloc.rfind("@") + 1 :]
    return (
        parsed._replace(netloc=netloc).geturl(),
        f"Basic {base64.b64encode(pair.encode()).decode()}",
    )
