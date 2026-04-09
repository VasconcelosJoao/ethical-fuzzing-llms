"""
Oracle wrapper — applies oracle labeling and SAVES labeled CSVs back to outputs/.

Usage:
    python oracle_runner.py rf1
    python oracle_runner.py all
"""

import os
import sys
import glob
import importlib.util

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import pandas as pd

ORACLE_MAP = {
    "rf1": ("src/fuzzer_modules/oracles/oracle-rf1.py", "outputs/rf1_*.csv", "label_pairs"),
    "rf2": ("src/fuzzer_modules/oracles/oracle-rf2.py", "outputs/rf2_*.csv", "label_pairs"),
    "rf4": ("src/fuzzer_modules/oracles/oracle-rf4.py", "outputs/rf4_*.csv", "label_rows"),
    "ra2": ("src/fuzzer_modules/oracles/oracle-ra2.py", "outputs/ra2_*.csv", "label_rows"),
    "rt1": ("src/fuzzer_modules/oracles/oracle-rt1.py", "outputs/rt1_*.csv", None),
    "rt2": ("src/fuzzer_modules/oracles/oracle-rt2.py", "outputs/rt2_*.csv", "label_rows"),
}


def load_oracle_module(path):
    name = os.path.basename(path).replace("-", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"ERROR loading {path}: {e}", flush=True)
        raise
    return mod


def run_oracle(module):
    module = module.lower()
    if module not in ORACLE_MAP:
        print(f"Unknown module: {module}", flush=True)
        return

    oracle_path, pattern, label_fn_name = ORACLE_MAP[module]
    if not os.path.exists(oracle_path):
        print(f"Oracle not found: {oracle_path}", flush=True)
        return

    try:
        oracle_mod = load_oracle_module(oracle_path)
    except Exception as e:
        print(f"Failed to load oracle for {module}: {e}", flush=True)
        return

    # RT1 is special: has metamorphic + explanation sub-files
    if module == "rt1":
        for path in sorted(glob.glob("outputs/rt1_meta_*.csv")):
            print(f"Processing {path}...", flush=True)
            df = pd.read_csv(path)
            df = oracle_mod.label_metamorphic(df)
            df.to_csv(path, index=False)
            print(f"  -> {len(df)} rows labeled", flush=True)
        for path in sorted(glob.glob("outputs/rt1_expl_*.csv")):
            print(f"Processing {path}...", flush=True)
            df = pd.read_csv(path)
            df = oracle_mod.label_explanation(df)
            df.to_csv(path, index=False)
            print(f"  -> {len(df)} rows labeled", flush=True)
        return

    # Generic modules
    label_fn = getattr(oracle_mod, label_fn_name, None)
    if not label_fn:
        available = [a for a in dir(oracle_mod) if callable(getattr(oracle_mod, a)) and not a.startswith("_")]
        print(f"Function '{label_fn_name}' not found in {oracle_path}", flush=True)
        print(f"  Available functions: {available}", flush=True)
        return

    for path in sorted(glob.glob(pattern)):
        print(f"Processing {path}...", flush=True)
        df = pd.read_csv(path)
        df = label_fn(df)
        df.to_csv(path, index=False)
        total = len(df)
        fails = int(df["is_fail"].sum()) if "is_fail" in df.columns else 0
        print(f"  -> {total} rows, {fails} failures ({fails/total:.1%})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python oracle_runner.py <module|all>")
        sys.exit(1)

    target = sys.argv[1].lower()
    if target == "all":
        for m in ["rf1", "rf2", "rf4", "ra2", "rt1", "rt2"]:
            print(f"\n{'='*50}\n{m.upper()}\n{'='*50}", flush=True)
            run_oracle(m)
    else:
        run_oracle(target)