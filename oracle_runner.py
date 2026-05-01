"""Oracle runner for immutable derived outputs.

Reads campaign CSVs from `campaign_outputs/`, applies the module oracle in memory, and writes derived files to `oracle_results/<risk>/`.

Usage:
    python oracle_runner.py rf1
    python oracle_runner.py all

Note:
RF1 and RT2 use SBERT-based similarity metrics. `CUDA_VISIBLE_DEVICES` is set to an empty string by default so those models run on CPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

CAMPAIGN_OUTPUT_ROOT = Path("campaign_outputs")
OUTPUT_ROOT = Path("oracle_results")
SUPPORTED_MODULES = ["rf1", "rf2", "rf4", "ra2", "rt1", "rt2"]
KNOWN_PROVIDERS = {"openai", "deepseek", "gemini"}

ORACLE_MAP = {
    "rf1": {"oracle_path": "src/fuzzer_modules/oracles/oracle-rf1.py", "label_functions": {None: "label_pairs"}},
    "rf2": {"oracle_path": "src/fuzzer_modules/oracles/oracle-rf2.py", "label_functions": {None: "label_pairs"}},
    "rf4": {"oracle_path": "src/fuzzer_modules/oracles/oracle-rf4.py", "label_functions": {None: "label_rows"}},
    "ra2": {"oracle_path": "src/fuzzer_modules/oracles/oracle-ra2.py", "label_functions": {None: "label_rows"}},
    "rt1": {"oracle_path": "src/fuzzer_modules/oracles/oracle-rt1.py", "label_functions": {"meta": "label_metamorphic", "expl": "label_explanation"}},
    "rt2": {"oracle_path": "src/fuzzer_modules/oracles/oracle-rt2.py", "label_functions": {None: "label_rows"}},
}


@dataclass
class CampaignFileInfo:
    source_path: Path
    risk: str
    provider: str
    model: str
    variant: str | None = None


@dataclass
class ModuleStats:
    files_found: int = 0
    processed: int = 0
    generated: int = 0
    conflicts: int = 0
    rows_total: int = 0
    rows_fail: int = 0
    save_paths: list[str] = field(default_factory=list)
    inconsistencies: list[str] = field(default_factory=list)


@dataclass
class RunStats:
    modules: dict[str, ModuleStats] = field(default_factory=dict)
    combinations: set[tuple[str, str, str]] = field(default_factory=set)
    inconsistencies: list[str] = field(default_factory=list)
    save_paths: list[str] = field(default_factory=list)

    def module(self, risk: str) -> ModuleStats:
        if risk not in self.modules:
            self.modules[risk] = ModuleStats()
        return self.modules[risk]


def load_oracle_module(path: str):
    name = os.path.basename(path).replace("-", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if not spec or not spec.loader:
        raise RuntimeError(f"Could not load module spec for {path}")
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"ERROR loading {path}: {exc}", flush=True)
        raise
    return mod


def parse_campaign_filename(path: Path, expected_risk: str, expected_variant: str | None = None) -> CampaignFileInfo:
    name = path.stem.lower()
    tokens = name.split("_")

    if not tokens or tokens[0] != expected_risk:
        raise ValueError(f"risk mismatch in filename '{path.name}'")

    start_idx = 1
    variant = None
    if expected_risk == "rt1":
        if len(tokens) < 4:
            raise ValueError(f"invalid rt1 filename format: '{path.name}'")
        variant = tokens[1]
        if variant not in {"meta", "expl"}:
            raise ValueError(f"invalid rt1 variant '{variant}' in '{path.name}'")
        if expected_variant and variant != expected_variant:
            raise ValueError(
                f"rt1 variant mismatch: expected '{expected_variant}' got '{variant}' in '{path.name}'"
            )
        start_idx = 2

    rest = tokens[start_idx:]
    if len(rest) < 2:
        raise ValueError(f"could not parse provider/model in '{path.name}'")

    provider = rest[0]
    model = "_".join(rest[1:])

    if provider not in KNOWN_PROVIDERS:
        provider = rest[0]

    if not model:
        raise ValueError(f"model portion is empty in '{path.name}'")

    return CampaignFileInfo(
        source_path=path,
        risk=expected_risk,
        provider=provider,
        model=model,
        variant=variant,
    )


def extract_thresholds(oracle_mod) -> str:
    thresholds = {}
    for attr in dir(oracle_mod):
        if attr.startswith("_"):
            continue
        if "THETA" not in attr and "THRESHOLD" not in attr:
            continue
        value = getattr(oracle_mod, attr)
        if isinstance(value, (int, float, str, bool)):
            thresholds[attr] = value

    if not thresholds:
        return "N/A"
    return json.dumps(thresholds, sort_keys=True, ensure_ascii=True)


def infer_is_fail(df: pd.DataFrame) -> pd.Series:
    if "is_fail" in df.columns:
        return df["is_fail"].fillna(False).astype(bool)
    elif "label" in df.columns:
        return df["label"].fillna("").astype(str).str.upper().eq("FAIL")
    return pd.Series([False] * len(df), index=df.index, dtype=bool)


def infer_string_series(df: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].where(df[col].notna(), default).astype(str)
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def _normalize_identifier_values(series: pd.Series) -> set[str]:
    values = series.fillna("").astype(str).str.strip().str.lower()
    return {v for v in values.unique().tolist() if v}


def _identifier_warning(
    *,
    field_name: str,
    expected: str,
    observed_values: set[str],
    source_path: Path,
) -> str | None:
    if not observed_values:
        return (
            f"[WARN] {source_path.name}: CSV column '{field_name}' is missing/empty. "
            f"Falling back to filename value '{expected}'."
        )

    expected_norm = expected.strip().lower()
    if observed_values == {expected_norm}:
        return None

    observed = ", ".join(sorted(observed_values))
    return (
        f"[WARN] {source_path.name}: filename {field_name}='{expected_norm}' differs from "
        f"CSV values {{{observed}}}. Using CSV as canonical source."
    )


def build_output_filename(info: CampaignFileInfo) -> str:
    model_norm = info.model.lower()
    if info.risk == "rt1" and info.variant:
        return f"{info.risk}_{info.variant}_{info.provider}_{model_norm}_oracle_results.csv"
    return f"{info.risk}_{info.provider}_{model_norm}_oracle_results.csv"


class ConflictAbortError(RuntimeError):
    pass


def build_minimal_result_frame(
    df: pd.DataFrame,
    *,
    info: CampaignFileInfo,
    oracle_name: str,
    threshold_blob: str,
    consistency_warnings: list[str] | None = None,
) -> pd.DataFrame:
    """Build a minimal derived frame.

    This function creates a compact derived DataFrame containing join keys,
    `is_fail` and `fail_reason`, and only the metric columns produced by the
    oracle (discovered dynamically from `df`). Columns that are known to be
    irrelevant (e.g. `label`, `confidence`, `run_id`, `evaluated_at`,
    `pair_id`) are ignored.
    """
    is_fail = infer_is_fail(df)

    # Base columns always present
    base_cols = ["provider", "model", "seed_id", "variant_id", "is_fail", "fail_reason"]

    out = pd.DataFrame(index=df.index)

    provider_series = infer_string_series(df, ["provider"], default=info.provider)
    model_series = infer_string_series(df, ["model"], default=info.model)

    provider_values = _normalize_identifier_values(provider_series)
    model_values = _normalize_identifier_values(model_series)

    provider_warn = _identifier_warning(
        field_name="provider",
        expected=info.provider,
        observed_values=provider_values,
        source_path=info.source_path,
    )
    model_warn = _identifier_warning(
        field_name="model",
        expected=info.model,
        observed_values=model_values,
        source_path=info.source_path,
    )

    if consistency_warnings is not None:
        if provider_warn:
            consistency_warnings.append(provider_warn)
        if model_warn:
            consistency_warnings.append(model_warn)

    out["provider"] = provider_series
    out["model"] = model_series
    out["seed_id"] = infer_string_series(df, ["seed_id"])
    out["variant_id"] = infer_string_series(df, ["variant_id"], default=info.variant or "")
    out["is_fail"] = is_fail
    out["fail_reason"] = infer_string_series(df, ["fail_reason"], default="")

    # Exclude columns that are metadata or internal/duplicate
    exclude = {
        "provider",
        "model",
        "seed_id",
        "variant_id",
        "label",
        "is_fail",
        "fail_reason",
        "confidence",
        "evaluated_at",
        "run_id",
        "pair_id",
        "_source_file",
        "_provider_model",
    }

    # Discover metric columns produced by the oracle (stable ordering)
    metric_candidates = [c for c in df.columns if c not in exclude and not c.startswith("_")]
    metric_candidates = sorted(metric_candidates)

    # Copy metrics, coercing to numeric when possible
    for col in metric_candidates:
        series = df[col]
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().sum() > 0:
            out[col] = coerced
        else:
            out[col] = series.where(series.notna(), pd.NA)

    # Append oracle metadata
    out["oracle_name"] = oracle_name
    out["oracle_thresholds"] = threshold_blob

    # Construct final column order: base cols, discovered metrics, oracle metadata
    final_cols = base_cols + [c for c in metric_candidates] + ["oracle_name", "oracle_thresholds"]
    return out.reindex(columns=final_cols)


def get_label_function(oracle_mod, module: str, variant: str | None) -> Callable[[pd.DataFrame], pd.DataFrame]:
    fn_name = ORACLE_MAP[module]["label_functions"].get(variant)
    if fn_name is None:
        fn_name = ORACLE_MAP[module]["label_functions"].get(None)

    label_fn = getattr(oracle_mod, fn_name, None)
    if label_fn:
        return label_fn

    available = [a for a in dir(oracle_mod) if callable(getattr(oracle_mod, a)) and not a.startswith("_")]
    raise AttributeError(
        f"Function '{fn_name}' not found in oracle for module '{module}'. Available: {available}"
    )


def collect_files_for_module(module: str, stats: RunStats) -> list[CampaignFileInfo]:
    module_stats = stats.module(module)
    collected: list[CampaignFileInfo] = []

    candidate_paths = sorted(
        [p for p in CAMPAIGN_OUTPUT_ROOT.rglob("*.csv") if p.name.lower().startswith(f"{module}_")],
        key=lambda path: str(path),
    )
    module_stats.files_found += len(candidate_paths)
    for p in candidate_paths:
        try:
            info = parse_campaign_filename(p, expected_risk=module)
            collected.append(info)
        except Exception as exc:
            msg = f"[WARN] {module}: parse inconsistency for {p.name}: {exc}"
            print(msg, flush=True)
            module_stats.inconsistencies.append(msg)
            stats.inconsistencies.append(msg)

    return collected


def run_oracle(module: str, stats: RunStats, on_conflict: str) -> None:
    module = module.lower()
    if module not in ORACLE_MAP:
        msg = f"Unknown module: {module}"
        print(msg, flush=True)
        stats.inconsistencies.append(msg)
        return

    oracle_path = ORACLE_MAP[module]["oracle_path"]
    if not os.path.exists(oracle_path):
        msg = f"Oracle not found for {module}: {oracle_path}"
        print(msg, flush=True)
        stats.inconsistencies.append(msg)
        stats.module(module).inconsistencies.append(msg)
        return

    try:
        oracle_mod = load_oracle_module(oracle_path)
    except Exception as exc:
        msg = f"Failed to load oracle for {module}: {exc}"
        print(msg, flush=True)
        stats.inconsistencies.append(msg)
        stats.module(module).inconsistencies.append(msg)
        return

    threshold_blob = extract_thresholds(oracle_mod)
    oracle_name = Path(oracle_path).stem
    module_stats = stats.module(module)

    files = collect_files_for_module(module, stats)
    if not files:
        print(
            f"[INFO] No source files found for {module} under {CAMPAIGN_OUTPUT_ROOT}",
            flush=True,
        )
        return

    for info in files:
        print(f"Processing {info.source_path}...", flush=True)
        try:
            df = pd.read_csv(info.source_path)
            label_fn = get_label_function(oracle_mod, module, info.variant)
            df_labeled = label_fn(df)
            consistency_warnings: list[str] = []
            df_derived = build_minimal_result_frame(
                df_labeled,
                info=info,
                oracle_name=oracle_name,
                threshold_blob=threshold_blob,
                consistency_warnings=consistency_warnings,
            )

            for msg in consistency_warnings:
                print(msg, flush=True)
                module_stats.inconsistencies.append(msg)
                stats.inconsistencies.append(msg)

            risk_output_dir = OUTPUT_ROOT / module
            risk_output_dir.mkdir(parents=True, exist_ok=True)

            preferred_name = build_output_filename(info)
            destination = risk_output_dir / preferred_name
            had_conflict = destination.exists()
            if had_conflict:
                module_stats.conflicts += 1
                if on_conflict == "cancel":
                    raise ConflictAbortError(
                        f"[CONFLICT] Destination already exists: {destination}. "
                        "Re-run with --on-conflict overwrite to replace it."
                    )
                print(f"[WARN] Destination exists; overwriting: {destination.name}", flush=True)

            df_derived.to_csv(destination, index=False)

            total = len(df_derived)
            fails = int(df_derived["is_fail"].sum()) if "is_fail" in df_derived.columns else 0
            rate = (fails / total) if total else 0.0

            module_stats.processed += 1
            module_stats.generated += 1
            module_stats.rows_total += total
            module_stats.rows_fail += fails
            module_stats.save_paths.append(str(destination))
            stats.save_paths.append(str(destination))
            combinations = df_derived[["provider", "model"]].drop_duplicates()
            for row in combinations.itertuples(index=False):
                stats.combinations.add((module, str(row.provider), str(row.model)))

            print(f"  -> {total} rows, {fails} failures ({rate:.1%})", flush=True)
            print(f"  -> saved derived file: {destination}", flush=True)
        except ConflictAbortError:
            raise
        except Exception as exc:
            msg = f"[ERROR] {module}: failed processing {info.source_path.name}: {exc}"
            print(msg, flush=True)
            module_stats.inconsistencies.append(msg)
            stats.inconsistencies.append(msg)


def print_module_summary(module: str, module_stats: ModuleStats) -> None:
    aggregate_failure_rate = (
        module_stats.rows_fail / module_stats.rows_total if module_stats.rows_total else 0.0
    )
    print(f"\nSummary for {module.upper()}", flush=True)
    print(f"  files_found: {module_stats.files_found}", flush=True)
    print(f"  files_processed: {module_stats.processed}", flush=True)
    print(f"  derived_generated: {module_stats.generated}", flush=True)
    print(f"  conflicts: {module_stats.conflicts}", flush=True)
    print(
        f"  rows_fail/rows_total: {module_stats.rows_fail}/{module_stats.rows_total} "
        f"(failure_rate={aggregate_failure_rate:.1%})",
        flush=True,
    )
    if module_stats.inconsistencies:
        print("  inconsistencies:", flush=True)
        for entry in module_stats.inconsistencies:
            print(f"    - {entry}", flush=True)


def print_final_summary(stats: RunStats, selected_modules: list[str]) -> None:
    print("\n" + "=" * 70, flush=True)
    print("ORACLE RUN SUMMARY", flush=True)
    print("=" * 70, flush=True)

    total_found = 0
    total_processed = 0
    total_generated = 0
    total_conflicts = 0
    total_rows = 0
    total_fails = 0

    for module in selected_modules:
        module_stats = stats.module(module)
        total_found += module_stats.files_found
        total_processed += module_stats.processed
        total_generated += module_stats.generated
        total_conflicts += module_stats.conflicts
        total_rows += module_stats.rows_total
        total_fails += module_stats.rows_fail
        print_module_summary(module, module_stats)

    global_failure_rate = (total_fails / total_rows) if total_rows else 0.0
    print("\nGlobal totals", flush=True)
    print(f"  files_found: {total_found}", flush=True)
    print(f"  files_processed: {total_processed}", flush=True)
    print(f"  derived_generated: {total_generated}", flush=True)
    print(f"  conflicts: {total_conflicts}", flush=True)
    print(f"  rows_fail/rows_total: {total_fails}/{total_rows} (failure_rate={global_failure_rate:.1%})", flush=True)
    print(f"  combinations_risk_provider_model: {len(stats.combinations)}", flush=True)

    if stats.combinations:
        print("  evaluated_combinations:", flush=True)
        for risk, provider, model in sorted(stats.combinations):
            print(f"    - {risk} | {provider} | {model}", flush=True)

    if stats.inconsistencies:
        print("\nConsolidated inconsistencies", flush=True)
        for entry in stats.inconsistencies:
            print(f"  - {entry}", flush=True)
    else:
        print("\nConsolidated inconsistencies", flush=True)
        print("  - none", flush=True)

    if stats.save_paths:
        print("\nSaved paths", flush=True)
        for path in stats.save_paths:
            print(f"  - {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run oracle labeling without modifying campaign_outputs/ source files.")
    parser.add_argument("target", help="module name (rf1, rf2, rf4, ra2, rt1, rt2) or all")
    parser.add_argument(
        "--on-conflict",
        choices=["cancel", "overwrite"],
        default="cancel",
        help="Behavior when destination file already exists in oracle_results",
    )
    args = parser.parse_args()

    target = args.target.lower()
    if target != "all" and target not in SUPPORTED_MODULES:
        print(f"Unknown module: {target}")
        sys.exit(1)

    stats = RunStats()

    modules_to_run = SUPPORTED_MODULES if target == "all" else [target]
    try:
        for module_name in modules_to_run:
            print(f"\n{'=' * 50}\n{module_name.upper()}\n{'=' * 50}", flush=True)
            run_oracle(module_name, stats, on_conflict=args.on_conflict)
    except ConflictAbortError as exc:
        print(str(exc), flush=True)
        print("Execution canceled due to destination conflict.", flush=True)
        sys.exit(2)

    print_final_summary(stats, modules_to_run)
