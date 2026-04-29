import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


HIGH_RISK_COLUMNS = {
    "method",
    "path_edges",
    "planned_path",
    "baseline_path",
    "timing_info_json",
    "route_edges",
    "debug_info",
    "anchors",
    "conflicts",
}


def _to_arrow_string_series(series: pd.Series) -> pd.Series:
    values = [str(value) for value in series.tolist()]
    try:
        import pyarrow as pa

        return pd.Series(values, index=series.index, dtype=pd.ArrowDtype(pa.string()))
    except Exception:
        return pd.Series(values, index=series.index, dtype="string")


def _to_arrow_typed_series(series: pd.Series, pa_type):
    try:
        import pyarrow as pa

        return pd.Series(series.tolist(), index=series.index, dtype=pd.ArrowDtype(pa_type))
    except Exception:
        return series


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, (list, tuple, np.ndarray, pd.Series)):
        return False
    return bool(result)


def _json_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, set):
        value = sorted(value, key=lambda item: str(item))
    if isinstance(value, tuple):
        value = list(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict), default=str)
    except TypeError:
        return json.dumps(value, ensure_ascii=False, default=str)


def _safe_cell(value: Any) -> Any:
    if _is_missing(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
        return _json_string(value)
    return value


def _dedupe_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for column in columns:
        base = str(column)
        count = seen.get(base, 0)
        deduped.append(base if count == 0 else f"{base}__{count + 1}")
        seen[base] = count + 1
    return deduped


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except ValueError:
            df = pd.DataFrame([df])
    df = df.copy()
    df.columns = _dedupe_columns(df.columns)

    for col in df.columns:
        col_data = df[col].map(_safe_cell)
        col_name = str(col).strip().lower()

        if col_name in HIGH_RISK_COLUMNS:
            df[col] = _to_arrow_string_series(col_data)
            continue

        if pd.api.types.is_bool_dtype(col_data.dtype):
            import pyarrow as pa

            df[col] = _to_arrow_typed_series(col_data, pa.bool_())
            continue

        if pd.api.types.is_integer_dtype(col_data.dtype):
            import pyarrow as pa

            df[col] = _to_arrow_typed_series(col_data, pa.int64())
            continue

        if pd.api.types.is_float_dtype(col_data.dtype):
            import pyarrow as pa

            df[col] = _to_arrow_typed_series(col_data, pa.float64())
            continue

        if pd.api.types.is_datetime64_any_dtype(col_data.dtype):
            df[col] = _to_arrow_string_series(col_data)
            continue

        # If object type, normalize cells that Arrow cannot infer safely.
        if (
            col_data.dtype == "object"
            or pd.api.types.is_string_dtype(col_data.dtype)
            or isinstance(col_data.dtype, pd.CategoricalDtype)
        ):
            df[col] = _to_arrow_string_series(col_data)
            continue

        df[col] = _to_arrow_string_series(col_data)

    return df


def flatten_json_record(record, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if not isinstance(record, Mapping):
        return {prefix or "value": _safe_cell(record)}

    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_json_record(value, full_key))
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            flattened[full_key] = _json_string(value)
        else:
            flattened[full_key] = _safe_cell(value)
    return flattened
