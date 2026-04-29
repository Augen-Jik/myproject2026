from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
import re
from typing import Iterable

import pandas as pd


FINAL_TABLE_DIR = Path("/root/autodl-tmp/results/final_tables")
PAPER_READY_DIR = Path("/root/autodl-tmp/results/paper_ready")


@dataclass(frozen=True)
class ResolvedArtifact:
    label: str
    preferred_path: str
    path: Path | None
    resolved_by_search: bool = False

    @property
    def exists(self) -> bool:
        return self.path is not None and self.path.exists()

    @property
    def last_modified(self) -> str | None:
        if not self.exists:
            return None
        return pd.Timestamp(self.path.stat().st_mtime, unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass(frozen=True)
class FrozenTableBundle:
    key: str
    title: str
    caption: str
    csv_artifact: ResolvedArtifact
    tex_artifact: ResolvedArtifact | None
    md_artifact: ResolvedArtifact | None
    dataframe: pd.DataFrame
    tex_text: str
    md_text: str
    paper_columns: list[str]


TABLE_SPECS = {
    "method_comparison": {
        "title": "Method Comparison / 主表",
        "caption": "Final frozen results bound to the method-comparison release files.",
        "csv": "/root/autodl-tmp/results/final_tables/method_comparison_final.csv",
        "tex": "/root/autodl-tmp/results/paper_ready/table1_method_comparison_final.tex",
        "md": "/root/autodl-tmp/results/paper_ready/table1_final_preview.md",
    },
    "ablation_study": {
        "title": "Ablation Study / 消融表",
        "caption": "Final frozen ablation results bound to the official ablation release files.",
        "csv": "/root/autodl-tmp/results/final_tables/ablation_study.csv",
        "tex": "/root/autodl-tmp/results/paper_ready/table2_ablation.tex",
        "md": "/root/autodl-tmp/results/paper_ready/table2_final_preview.md",
    },
    "appendix": {
        "title": "Appendix / Exploratory / 附录",
        "caption": "Appendix-only or exploratory results. Not the main frozen claim table.",
        "csv": "/root/autodl-tmp/results/final_tables/method_comparison_appendix.csv",
        "tex": None,
        "md": None,
    },
}


def _candidate_files(search_dirs: Iterable[Path], suffix: str) -> list[Path]:
    candidates: list[Path] = []
    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        candidates.extend(sorted(path for path in base_dir.rglob(f"*{suffix}") if path.is_file()))
    return candidates


def _resolve_artifact(
    label: str,
    preferred_path: str | None,
    search_dirs: Iterable[Path],
    *,
    allow_search: bool = True,
) -> ResolvedArtifact | None:
    if not preferred_path:
        return None

    preferred = Path(preferred_path)
    if preferred.exists():
        return ResolvedArtifact(label=label, preferred_path=preferred_path, path=preferred, resolved_by_search=False)

    if not allow_search:
        return ResolvedArtifact(label=label, preferred_path=preferred_path, path=None, resolved_by_search=False)

    suffix = preferred.suffix or ""
    candidates = _candidate_files(search_dirs, suffix)
    names = [path.name for path in candidates]
    matches = get_close_matches(preferred.name, names, n=1, cutoff=0.45)
    if matches:
        matched_name = matches[0]
        matched_path = next(path for path in candidates if path.name == matched_name)
        return ResolvedArtifact(label=label, preferred_path=preferred_path, path=matched_path, resolved_by_search=True)

    return ResolvedArtifact(label=label, preferred_path=preferred_path, path=None, resolved_by_search=True)


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _parse_tex_columns(tex_text: str) -> list[str]:
    if not tex_text:
        return []
    match = re.search(r"\\toprule\s*(.*?)\\\\\s*\\midrule", tex_text, flags=re.S)
    if not match:
        return []
    header_line = match.group(1).strip()
    columns = []
    for raw_column in header_line.split("&"):
        cleaned = raw_column.strip()
        cleaned = cleaned.replace("\\%", "%").replace("\\&", "&")
        cleaned = re.sub(r"\\textit\{([^}]*)\}", r"\1", cleaned)
        cleaned = re.sub(r"\\[A-Za-z]+", "", cleaned)
        cleaned = cleaned.replace("{", "").replace("}", "").strip()
        if cleaned:
            columns.append(cleaned)
    return columns


def load_frozen_table_bundle(key: str) -> FrozenTableBundle:
    spec = TABLE_SPECS[key]
    search_dirs = [FINAL_TABLE_DIR, PAPER_READY_DIR]

    # Benchmark / appendix CSV bindings must stay pinned to the exact frozen files.
    csv_artifact = _resolve_artifact("CSV", spec["csv"], search_dirs, allow_search=False)
    if csv_artifact is None or not csv_artifact.exists:
        raise FileNotFoundError(f"Missing CSV artifact for {key}: {spec['csv']}")

    tex_artifact = _resolve_artifact("LaTeX", spec["tex"], search_dirs, allow_search=True)
    md_artifact = _resolve_artifact("Markdown", spec["md"], search_dirs, allow_search=True)

    dataframe = pd.read_csv(csv_artifact.path)
    tex_text = _read_text(tex_artifact.path if tex_artifact else None)
    md_text = _read_text(md_artifact.path if md_artifact else None)
    paper_columns = _parse_tex_columns(tex_text)

    return FrozenTableBundle(
        key=key,
        title=spec["title"],
        caption=spec["caption"],
        csv_artifact=csv_artifact,
        tex_artifact=tex_artifact,
        md_artifact=md_artifact,
        dataframe=dataframe,
        tex_text=tex_text,
        md_text=md_text,
        paper_columns=paper_columns,
    )


def frozen_table_cache_signature(key: str) -> tuple:
    spec = TABLE_SPECS[key]
    search_dirs = [FINAL_TABLE_DIR, PAPER_READY_DIR]
    signature: list[object] = [key]
    for label, preferred_path in [
        ("csv", spec["csv"]),
        ("tex", spec["tex"]),
        ("md", spec["md"]),
    ]:
        allow_search = label != "csv"
        artifact = (
            _resolve_artifact(label.upper(), preferred_path, search_dirs, allow_search=allow_search)
            if preferred_path else None
        )
        signature.append(label)
        signature.append(preferred_path or "")
        signature.append(str(artifact.path) if artifact and artifact.path else "")
        signature.append(artifact.path.stat().st_mtime_ns if artifact and artifact.exists else -1)
        signature.append(artifact.path.stat().st_size if artifact and artifact.exists else -1)
    return tuple(signature)


def build_ablation_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df.copy()

    group_keys = [column for column in ["Variant", "Variant (CN)"] if column in raw_df.columns]
    if not group_keys:
        raise ValueError("ablation_study.csv does not contain a Variant column")

    numeric_columns = [
        column
        for column in raw_df.columns
        if pd.api.types.is_numeric_dtype(raw_df[column]) and column not in group_keys
    ]

    first_columns = [
        column
        for column in [
            "LLM Component",
            "GAT Component",
            "Sparse Mode",
        ]
        if column in raw_df.columns
    ]

    order_df = raw_df[group_keys].drop_duplicates().reset_index(drop=True)
    order_df["__variant_order"] = range(len(order_df))

    agg_map = {column: "mean" for column in numeric_columns}
    agg_map.update({column: "first" for column in first_columns})
    summary = raw_df.groupby(group_keys, sort=False, dropna=False).agg(agg_map).reset_index()
    summary = summary.merge(order_df, on=group_keys, how="left").sort_values("__variant_order")
    summary["Scenes"] = raw_df.groupby(group_keys, dropna=False).size().reset_index(name="Scenes")["Scenes"]
    summary = summary.drop(columns="__variant_order").reset_index(drop=True)
    return summary
