"""Scholarly-source backends for :mod:`wesearch.paper`.

One module per source under one roof: :mod:`.s2` (Semantic Scholar),
:mod:`.openalex`, and :mod:`.searxng`. Some builds add further scraped sources.
Each exposes sync functions returning the backend-agnostic record types;
backend selection and fusion live in :mod:`wesearch.paper.search`.
"""

from __future__ import annotations
