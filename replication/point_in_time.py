"""Point-in-time S&P 500 constituent reconstruction, to reduce survivorship
bias rather than just disclose it: instead of holding today's 503 names
constant across the whole backtest history (which would let names that were
added *because* they later became large-cap winners flatter every historical
period they weren't actually in the index for), this rolls today's real
constituent list backward through `data.wikipedia_constituents`' dated
add/remove event log, one event at a time.

## Reconstruction logic

Membership at an `as_of` date = today's constituents, with every change
that became effective *after* `as_of` undone, most-recent-event-first:
  - undoing an addition means removing that ticker from the set
  - undoing a removal means putting that ticker back into the set

Processing strictly in descending effective-date order (not just "any
order") matters whenever the same ticker was added, later removed, and
`as_of` predates both: undoing the removal first (adding it back), then
undoing the addition (discarding it), nets to "not present" -- correct,
since `as_of` is before the ticker ever joined. Undoing the events in the
other order would leave it incorrectly present. This is event-sourcing
played backward, not an unordered set of patches.

## Disclosed gaps (real ones, not smoothed over)

- The changes log starts 1976-07-01. Reconstruction before that date falls
  back to today's constituent list, i.e. the survivorship bias this module
  exists to reduce is NOT reduced for dates before 1976 -- there's no
  dated source here to reconstruct from. `coverage_notes()` flags this.
- A pure ticker/company renaming that Wikipedia's editors logged as a
  same-company continuation (not an add+remove pair) won't appear in the
  changes table at all, so it's invisible to this reconstruction. This is a
  real, unquantified gap -- not every rename is guaranteed to have been
  captured as a change event, and this module has no independent way to
  detect the omission.
- Effective dates mark when S&P *implemented* the change; this module does
  not attempt to distinguish that from the (usually few-days-earlier)
  announcement date, since the changes table only records the former.
"""

from __future__ import annotations

import pandas as pd

CHANGES_HISTORY_START = pd.Timestamp("1976-07-01")


def reconstruct_membership(as_of: pd.Timestamp, current: pd.DataFrame, changes: pd.DataFrame) -> set[str]:
    """Returns the set of `symbol` (Wikipedia/S&P canonical ticker form,
    dots not dashes) believed to be S&P 500 constituents at `as_of`."""
    membership = set(current["symbol"])

    relevant = changes[changes["effective_date"] > as_of].sort_values("effective_date", ascending=False)
    for row in relevant.itertuples():
        if pd.notna(row.added_ticker):
            membership.discard(row.added_ticker)
        if pd.notna(row.removed_ticker):
            membership.add(row.removed_ticker)

    return membership


def build_membership_history(
    rebalance_dates: list[pd.Timestamp], current: pd.DataFrame, changes: pd.DataFrame
) -> pd.DataFrame:
    """Long-format DataFrame: one row per (rebalance_date, symbol) believed
    to be a constituent at that date. A wide membership matrix is easy to
    derive from this (`.assign(member=True).pivot(...)`) but the long form
    is the more useful default for joining against price/return panels."""
    rows = []
    for date in rebalance_dates:
        for symbol in reconstruct_membership(date, current, changes):
            rows.append((date, symbol))
    return pd.DataFrame(rows, columns=["rebalance_date", "symbol"])


def quarterly_rebalance_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """S&P 500's real quarterly rebalance effective date: the third Friday
    of March/June/September/December (effective after that Friday's close).
    Not an invented schedule -- this is S&P Dow Jones Indices' documented
    quarterly reconstitution timing."""
    dates = []
    for year in range(start.year, end.year + 1):
        for month in (3, 6, 9, 12):
            fridays = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
            fridays = fridays[(fridays.month == month) & (fridays.weekday == 4)]
            third_friday = fridays[2]
            if start <= third_friday <= end:
                dates.append(third_friday)
    return sorted(dates)


def coverage_notes(current: pd.DataFrame, changes: pd.DataFrame) -> list[str]:
    """Human-readable disclosure of this reconstruction's known gaps, meant
    to be surfaced in the README/results output directly rather than left
    implicit in this module's docstring alone."""
    notes = [
        f"Changes history covers {changes['effective_date'].min().date()} to "
        f"{changes['effective_date'].max().date()}. Reconstruction for dates before "
        f"{CHANGES_HISTORY_START.date()} falls back to today's constituent list and does "
        "not correct for survivorship bias in that earlier period.",
        "Pure ticker/company renames not logged as an explicit add+remove pair in the "
        "changes table are invisible to this reconstruction -- a real, unquantified gap.",
    ]

    added_before_start = current[
        (current["date_added"].notna()) & (current["date_added"] < CHANGES_HISTORY_START)
    ]
    if len(added_before_start):
        notes.append(
            f"{len(added_before_start)} of {len(current)} current constituents were added "
            f"before the changes log starts ({CHANGES_HISTORY_START.date()}) -- their "
            "pre-addition history is correctly excluded, but any index membership *prior* "
            "to their addition is only as reliable as the changes log allows."
        )
    return notes
