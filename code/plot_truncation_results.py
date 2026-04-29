#!/usr/bin/env python3
"""Plot truncation metrics from truncation_summary.csv to a self-contained HTML file."""

from __future__ import annotations

import argparse
import csv
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots


VARIANT_ORDER = ["short", "medium", "long", "noisy_long"]


def load_rows(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Plot truncation metrics from truncation_summary.csv")
    parser.add_argument("summary_csv", help="Path to truncation_summary.csv")
    parser.add_argument("--output", help="Output html path; defaults next to csv")
    args = parser.parse_args()

    rows = load_rows(args.summary_csv)
    if not rows:
        raise SystemExit("No rows found in summary csv")

    output = args.output or os.path.join(os.path.dirname(args.summary_csv), "truncation_vs_complexity.html")
    model_names = sorted({row["model"] for row in rows})

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Truncation Rate vs Description Complexity",
            "Parse Failure Rate vs Description Complexity",
        ),
    )

    for model in model_names:
        model_rows = {row["variant"]: row for row in rows if row["model"] == model}
        x = [variant for variant in VARIANT_ORDER if variant in model_rows]
        trunc_y = [float(model_rows[v]["truncation_rate"]) for v in x]
        parse_y = [float(model_rows[v]["parse_failure_rate"]) for v in x]
        fig.add_trace(
            go.Scatter(x=x, y=trunc_y, mode="lines+markers", name=f"{model} truncation"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=x, y=parse_y, mode="lines+markers", name=f"{model} parse_fail"),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_white",
        width=1100,
        height=430,
        title="Truncation Robustness Curves",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0.0),
    )
    fig.update_xaxes(title_text="Description Variant", row=1, col=1)
    fig.update_xaxes(title_text="Description Variant", row=1, col=2)
    fig.update_yaxes(title_text="Truncation Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Parse Failure Rate (%)", row=1, col=2)
    fig.write_html(output, include_plotlyjs="cdn")
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
