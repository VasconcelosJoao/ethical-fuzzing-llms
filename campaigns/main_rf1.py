"""
main_rf1.py — Executes the RF1 (Discrimination) fuzzing campaign.

For each seed, generates counterfactual prompt pairs via mutation-based
fuzzing (substituting protected attributes), sends both prompts to each
provider, and logs paired outputs for similarity oracle evaluation.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.exec_module as exe
import src.fuzzer_modules.rf1 as rf1
import src.logger as lg

import pandas as pd
from config import PROVIDER_MODEL, K, SYSTEM_PROMPTS


RF1_COLUMNS = [
    "provider", "model", "seed_id", "variant_id", "attribute", "domain",
    "template_id", "group_a", "profile_id_a", "prompt_a", "output_a",
    "group_b", "profile_id_b", "prompt_b", "output_b",
    "label",
]


def main():
    seeds_df = pd.read_csv("data/rf1/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rf1.fuzz_rf1(row, k=K)
        print(f"Generated {K} variant pairs for {row.seed_id}")

    params = {
        "system_prompt": SYSTEM_PROMPTS["RF1"],
        "deepseek": {},
        "openai": {},
        "gemini": {},
    }

    for provider, model in PROVIDER_MODEL.items():
        out_rows = []
        logger = lg.new_run_logger(
            out_dir="logs",
            prefix=f"rf1_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RF1"},
        )

        for row in seeds_df.itertuples():
            print(f"[RF1] Executing {provider}/{model} — seed {row.seed_id}...")
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
                    res_a = exe.execute_single(pa["prompt"], "counterfactual", provider, model, params)
                    time.sleep(0.4)
                    res_b = exe.execute_single(pb["prompt"], "counterfactual", provider, model, params)

                    logger.write("variant_result", {
                        "variant_id": variant["variant_id"],
                        "output_a": lg.safe_preview(res_a.get("text")),
                        "output_b": lg.safe_preview(res_b.get("text")),
                        "status": "ok",
                    })

                    out_rows.append([
                        provider, model,
                        variant["seed_id"], variant["variant_id"],
                        variant["meta"]["attribute"], variant["meta"]["domain"],
                        variant["meta"]["template_id"],
                        pa["group"], pa["profile_id"], pa["prompt"], res_a.get("text", ""),
                        pb["group"], pb["profile_id"], pb["prompt"], res_b.get("text", ""),
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

        out_df = pd.DataFrame(out_rows, columns=RF1_COLUMNS)
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/rf1_{provider}_{model}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()