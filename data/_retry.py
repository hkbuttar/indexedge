"""Shared retry-with-backoff for yfinance's per-symbol calls (`Ticker.info`,
`Ticker.get_shares_full()`) -- real network dependency on an undocumented,
rate-limit-undisclosed third-party API, called sequentially (no batch
endpoint exists for either -- see `data/fundamentals.py` and
`data/shares_outstanding.py`'s own docstrings).

A real production failure motivated this, not a hypothetical one: a
deployed build's sequential fundamentals fetch over 503 symbols stopped
partway through with a clean cutoff around the 350th symbol of a sorted
list (everything after succeeded locally, moments later, against the same
symbols) -- consistent with either a build-step time limit or Yahoo
throttling after many rapid sequential requests from the same IP. This
project's environment has no way to distinguish which from the build log
alone. Retrying with a short backoff helps the case where a given request
was genuinely transient (a momentary throttle, a network blip); it does
nothing for a hard, sustained block, which is a real, disclosed limitation
this cannot fully solve -- `attempts`/`base_delay` are deliberately small
(2 tries, ~1.5s) so a truly-blocked run fails fast rather than spending its
own remaining time budget on retries that can't succeed, which would make
a time-limit-caused truncation worse, not better.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_ATTEMPTS = 2
DEFAULT_BASE_DELAY = 1.5


def retry_with_backoff(fn: Callable[[], T], attempts: int = DEFAULT_ATTEMPTS, base_delay: float = DEFAULT_BASE_DELAY) -> T | None:
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(base_delay * (2 ** attempt))
    return None
