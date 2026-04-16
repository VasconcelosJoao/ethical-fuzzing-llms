"""
main_rt2.py — Executes the RT2 (Hidden Biases) fuzzing campaign.

For each seed, generates counterfactual pairs where ONLY an irrelevant
attribute differs, sends both prompts to each provider, and logs
paired outputs for invariance oracle evaluation.

Structurally identical to main_rf1.py (single-turn pairs), but
tests for INVARIANCE rather than discrimination.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.exec_module as exe
import src.fuzzer_modules.rt2 as rt2
import src.logger as lg

import pandas as pd
from config import PROVIDER_MODEL, K, SYSTEM_PROMPTS


RT2_COLUMNS = [
    "provider", "model", "seed_id", "variant_id",
    "perturbation_category", "domain", "template_id",
    "perturbation_a_id", "perturbation_a_label",
    "perturbation_b_id", "perturbation_b_label",
    "prompt_a", "output_a",
    "prompt_b", "output_b",
    "label",
]


def main():
    seeds_df = pd.read_csv("data/rt2/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rt2.fuzz_rt2(row, k=K)
        print(f"Generated {K} variant pairs for {row.seed_id}")

    params = {
        "system_prompt": SYSTEM_PROMPTS["RT2"],
        "deepseek": {},
        "openai": {},
        "gemini": {},
    }

    for provider, model in PROVIDER_MODEL.items():
        out_rows = []
        logger = lg.new_run_logger(
            out_dir="logs",
            prefix=f"rt2_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RT2"},
        )

        for row in seeds_df.itertuples():
            print(f"[RT2] {provider}/{model} — {row.seed_id} ({row.perturbation_category}/{row.domain})...")
            variants = variant_cache[row.seed_id]

            for variant in variants:
                logger.write("variant_start", {
                    "seed_id": variant["seed_id"],
                    "variant_id": variant["variant_id"],
                    "meta": variant["meta"],
                })

                try:
                    res_a = exe.execute_single(variant["prompt_a"], "invariance_test", provider, model, params)
                    time.sleep(0.4)
                    res_b = exe.execute_single(variant["prompt_b"], "invariance_test", provider, model, params)

                    logger.write("variant_result", {
                        "variant_id": variant["variant_id"],
                        "output_a": lg.safe_preview(res_a.get("text")),
                        "output_b": lg.safe_preview(res_b.get("text")),
                        "status": "ok",
                    })

                    out_rows.append([
                        provider, model,
                        variant["seed_id"], variant["variant_id"],
                        variant["meta"]["perturbation_category"],
                        variant["meta"]["domain"],
                        variant["meta"]["template_id"],
                        variant["meta"]["perturbation_a_id"],
                        variant["meta"]["perturbation_a_label"],
                        variant["meta"]["perturbation_b_id"],
                        variant["meta"]["perturbation_b_label"],
                        variant["prompt_a"], res_a.get("text", ""),
                        variant["prompt_b"], res_b.get("text", ""),
                        "-",
                    ])
                    print(f"  ✓ {variant['variant_id']}")

                except Exception as e:
                    logger.write("variant_error", {
                        "variant_id": variant["variant_id"],
                        "status": "error",
                        "error_msg": str(e),
                    })
                    print(f"  ✗ {variant['variant_id']}: {e}")

                time.sleep(0.6)

            print(f"Finished seed {row.seed_id}")

        out_df = pd.DataFrame(out_rows, columns=RT2_COLUMNS)
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/rt2_{provider}_{model}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()