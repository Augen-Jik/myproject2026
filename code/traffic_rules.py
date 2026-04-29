from __future__ import annotations

"""Shared traffic-rule parsing for experiment and demo code paths."""

from dataclasses import dataclass
import re
from typing import Dict


DEFAULT_EDGE_WEIGHT = 2.0

ALL_EDGE_IDS = tuple(
    [f"R{r}C{c}_{d}" for r in range(5) for c in range(5) for d in ("E", "W")]
    + [f"C{c}R{r}_{d}" for c in range(6) for r in range(4) for d in ("N", "S")]
)

ROW_MAP = {
    "农业路": "R0",
    "红专路": "R1",
    "政七街": "R2",
    "黄河路": "R3",
    "纬五路": "R4",
}
COL_MAP = {
    "经一路": "C0",
    "经三路": "C1",
    "经六路": "C2",
    "经八路": "C3",
    "花园路": "C4",
    "未来路": "C5",
}
DIR_MAP = {"向东": "E", "向西": "W", "向北": "N", "向南": "S"}
KEYWORD_WEIGHTS = [
    (["封闭", "管制", "全封", "完全封", "全线封"], 9.5),
    (["连环追尾", "三车", "极度拥堵", "追尾事故"], 7.5),
    (["严重拥堵", "严重"], 7.5),
    (["中度拥堵", "中等"], 5.0),
    (["轻微拥堵", "轻微"], 3.0),
    (["畅通", "正常通行", "通畅", "正常"], 1.2),
]


@dataclass(frozen=True)
class RuleParseResult:
    final_weights: Dict[str, float]
    explicit_weights: Dict[str, float]


def build_default_weights(default_weight: float = DEFAULT_EDGE_WEIGHT) -> Dict[str, float]:
    """Build a full edge-weight dict with a shared neutral default."""
    return {eid: float(default_weight) for eid in ALL_EDGE_IDS}


def parse_constraint_weights(
    constraint: str,
    default_weight: float = DEFAULT_EDGE_WEIGHT,
) -> RuleParseResult:
    """
    Parse natural-language traffic constraints into edge weights.

    `final_weights` always contains all 98 directed edges for reproducibility.
    `explicit_weights` only tracks edges that were explicitly derived from the
    textual constraint so downstream metrics can distinguish parse coverage from
    default-value completion.
    """

    weights = build_default_weights(default_weight)
    explicit_weights: Dict[str, float] = {}

    def assign_weight(edge_id: str, value: float) -> None:
        value = float(value)
        weights[edge_id] = value
        explicit_weights[edge_id] = value

    sentences = re.split(r"[；。;]", constraint or "")
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        matched_weight = None
        for keywords, weight in KEYWORD_WEIGHTS:
            if any(keyword in sentence for keyword in keywords):
                matched_weight = weight
                break
        if matched_weight is None:
            continue

        directions = [value for key, value in DIR_MAP.items() if key in sentence]
        if not directions:
            directions = ["E", "W", "N", "S"]

        range_pattern = re.search(
            r"(经[一三六八]路|花园路|未来路|农业路|红专路|政七街|黄河路|纬五路)"
            r"[至到]"
            r"(经[一三六八]路|花园路|未来路|农业路|红专路|政七街|黄河路|纬五路)",
            sentence,
        )

        for row_name, row_key in ROW_MAP.items():
            if row_name not in sentence:
                continue
            if range_pattern:
                from_name, to_name = range_pattern.group(1), range_pattern.group(2)
                from_col = next((int(code[1]) for name, code in COL_MAP.items() if name in from_name), None)
                to_col = next((int(code[1]) for name, code in COL_MAP.items() if name in to_name), None)
                if from_col is None or to_col is None:
                    continue
                col_lo, col_hi = min(from_col, to_col), max(from_col, to_col)
                for edge_id in ALL_EDGE_IDS:
                    if not edge_id.startswith(row_key):
                        continue
                    match = re.match(r"R\d+C(\d+)_([EW])", edge_id)
                    if match and int(match.group(1)) in range(col_lo, col_hi) and match.group(2) in directions:
                        assign_weight(edge_id, matched_weight)
                continue

            for edge_id in ALL_EDGE_IDS:
                if edge_id.startswith(row_key) and edge_id.split("_")[1] in directions:
                    assign_weight(edge_id, matched_weight)

        for col_name, col_key in COL_MAP.items():
            if col_name not in sentence:
                continue
            if range_pattern:
                from_name, to_name = range_pattern.group(1), range_pattern.group(2)
                from_row = next((int(code[1]) for name, code in ROW_MAP.items() if name in from_name), None)
                to_row = next((int(code[1]) for name, code in ROW_MAP.items() if name in to_name), None)
                if from_row is None or to_row is None:
                    continue
                row_lo, row_hi = min(from_row, to_row), max(from_row, to_row)
                for edge_id in ALL_EDGE_IDS:
                    if not edge_id.startswith(col_key):
                        continue
                    match = re.match(r"C\d+R(\d+)_([NS])", edge_id)
                    if match and int(match.group(1)) in range(row_lo, row_hi) and match.group(2) in directions:
                        assign_weight(edge_id, matched_weight)
                continue

            for edge_id in ALL_EDGE_IDS:
                if edge_id.startswith(col_key) and edge_id.split("_")[1] in directions:
                    assign_weight(edge_id, matched_weight)

    return RuleParseResult(final_weights=weights, explicit_weights=explicit_weights)
