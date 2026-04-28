"""
main_rf2.py - Executes the RF2 (Unequal Access) fuzzing campaign.

For each seed, generates profile-parameterized prompt pairs via
generation-based fuzzing, sends both prompts to each provider, and logs
paired outputs for quality-differential oracle evaluation.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.exec_module as exe
import src.fuzzer_modules.rf2 as rf2
import src.logger as lg

import pandas as pd
from config import PROVIDER_MODEL, K, SYSTEM_PROMPTS


RF2_COLUMNS = [
    "provider", "model", "seed_id", "variant_id", "dimension", "domain",
    "template_id", "group_a", "profile_id_a", "prompt_a", "output_a",
    "group_b", "profile_id_b", "prompt_b", "output_b",
]


def main():
    seeds_df = pd.read_csv("data/rf2/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rf2.fuzz_rf2(row, k=K)
        print(f"Generated {K} variant pairs for {row.seed_id}")

    params = {
        "system_prompt": SYSTEM_PROMPTS["RF2"],
        "deepseek": {},
        "openai": {},
        "gemini": {},
    }

    for provider, model in PROVIDER_MODEL.items():
        out_rows = []
        logger = lg.new_run_logger(
            out_dir=os.path.join("execution_logs", "rf2"),
            prefix=f"rf2_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RF2"},
        )

        for row in seeds_df.itertuples():
            print(f"[RF2] Executing {provider}/{model} - seed {row.seed_id}...")
            variants = variant_cache[row.seed_id]

            for variant in variants:
                pairs = variant["pairs"]
                if len(pairs) < 2:
                    continue

                pa, pb = pairs[0], pairs[1]

                logger.write("variant_start", {
                    "seed_id": variant["seed_id"],
                    "variant_id": variant["variant_id"],
                    "meta": variant["meta"],
                    "prompt_a": pa["prompt"],
                    "prompt_b": pb["prompt"],
                })

                try:
                    res_a = exe.execute_single(pa["prompt"], "benefit_request", provider, model, params)
                    time.sleep(0.4)
                    res_b = exe.execute_single(pb["prompt"], "benefit_request", provider, model, params)

                    logger.write("variant_result", {
                        "variant_id": variant["variant_id"],
                        "output_a": lg.safe_preview(res_a.get("text")),
                        "output_b": lg.safe_preview(res_b.get("text")),
                        "status": "ok",
                    })

                    out_rows.append([
                        provider, model,
                        variant["seed_id"], variant["variant_id"],
                        variant["meta"]["dimension"], variant["meta"]["domain"],
                        variant["meta"]["template_id"],
                        pa["group"], pa["profile_id"], pa["prompt"], res_a.get("text", ""),
                        pb["group"], pb["profile_id"], pb["prompt"], res_b.get("text", ""),
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

        out_df = pd.DataFrame(out_rows, columns=RF2_COLUMNS)
        out_dir = os.path.join("campaign_outputs", "rf2")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"rf2_{provider}_{model}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()