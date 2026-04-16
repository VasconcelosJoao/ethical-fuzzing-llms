"""
main_rf4.py — Executes the RF4 (Subgroup Fairness) fuzzing campaign.

For each variant, sends the SAME benchmark question framed from
EACH subgroup's perspective, then logs all responses for
cross-group accuracy and quality comparison.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.exec_module as exe
import src.fuzzer_modules.rf4 as rf4
import src.logger as lg

import pandas as pd
from config import PROVIDER_MODEL, K, SYSTEM_PROMPTS


RF4_COLUMNS = [
    "provider", "model", "seed_id", "variant_id",
    "dimension", "benchmark_id", "domain", "benchmark_task",
    "group", "profile_id", "prompt", "output",
    "expected_elements", "min_elements",
    "label",
]


def main():
    seeds_df = pd.read_csv("data/rf4/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rf4.fuzz_rf4(row, k=K)
        print(f"Generated {K} variants for {row.seed_id}")

    params = {
        "system_prompt": SYSTEM_PROMPTS["RF4"],
        "deepseek": {},
        "openai": {},
        "gemini": {},
    }

    for provider, model in PROVIDER_MODEL.items():
        out_rows = []
        logger = lg.new_run_logger(
            out_dir="logs",
            prefix=f"rf4_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RF4"},
        )

        for row in seeds_df.itertuples():
            print(f"[RF4] {provider}/{model} — {row.seed_id} ({row.dimension}/{row.benchmark_category})...")
            variants = variant_cache[row.seed_id]

            for variant in variants:
                logger.write("variant_start", {
                    "seed_id": variant["seed_id"],
                    "variant_id": variant["variant_id"],
                    "meta": variant["meta"],
                })

                for prompt_info in variant["prompts"]:
                    try:
                        result = exe.execute_single(
                            prompt_info["prompt"], "benchmark_request", provider, model, params
                        )

                        logger.write("group_result", {
                            "variant_id": variant["variant_id"],
                            "group": prompt_info["group"],
                            "output_preview": lg.safe_preview(result.get("text")),
                            "status": "ok",
                        })

                        out_rows.append([
                            provider, model,
                            variant["seed_id"], variant["variant_id"],
                            variant["meta"]["dimension"],
                            variant["benchmark"]["id"],
                            variant["benchmark"]["domain"],
                            variant["benchmark"]["task"],
                            prompt_info["group"],
                            prompt_info["profile_id"],
                            prompt_info["prompt"],
                            result.get("text", ""),
                            "|".join(variant["benchmark"]["expected_elements"]),
                            variant["benchmark"]["min_elements"],
                            "-",
                        ])
                        print(f"  ✓ {variant['variant_id']}:{prompt_info['group']}")

                    except Exception as e:
                        logger.write("group_error", {
                            "variant_id": variant["variant_id"],
                            "group": prompt_info["group"],
                            "error_msg": str(e),
                        })
                        print(f"  ✗ {variant['variant_id']}:{prompt_info['group']}: {e}")

                    time.sleep(0.5)

            print(f"Finished seed {row.seed_id}")

        out_df = pd.DataFrame(out_rows, columns=RF4_COLUMNS)
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/rf4_{provider}_{model}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()