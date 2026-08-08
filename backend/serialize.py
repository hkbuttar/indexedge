"""JSON-safe serialization for backend responses.

Ported from riskdesk's `backend/serialize.py`: every module in this project
returns real Python objects (dataclasses, pandas Series/DataFrames, numpy
scalars) suited to the scripts and tests that consume them directly, not
pre-flattened into JSON-safe dicts -- that would couple every module's
return type to "whatever the eventual API needs," a dependency running the
wrong direction. This is the one place that conversion happens, so
`backend/main.py`'s route handlers stay thin wiring, not re-implementations
of what each module already computed.

One deliberate deviation from riskdesk's version: DataFrames serialize to a
list of row-records (`[{"_index": ..., "col": val, ...}, ...]`) here, not
riskdesk's `{index: {col: val}}` nested-dict form. riskdesk's DataFrames are
indexed by something meaningful (a position id); most of this project's
are the *results* of a walk-forward evaluation (sampling comparison,
regime breakdowns) with a default RangeIndex, where a JSON object keyed by
stringified "0", "1", "2" would be actively worse for a frontend to consume
than a plain array -- the natural shape for a charting library either way.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import math

import numpy as np
import pandas as pd


def to_jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, (np.floating,)):
        return to_jsonable(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (dt.datetime, dt.date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, pd.Index):
        return [to_jsonable(value) for value in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [
            {**{"_index": to_jsonable(idx)}, **{to_jsonable(col): to_jsonable(v) for col, v in row.items()}}
            for idx, row in obj.iterrows()
        ]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return obj
