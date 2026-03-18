"""
main_rf2.py — Executes the RF2 (Unequal Access) fuzzing campaign.

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

import src.formatter as fmt
import src.exec_module as exe
import src.fuzzer_modules.rf2 as rf2
import src.logger as lg

import pandas as pd
from config import PROVIDER_MODEL, K


# ---------------------------------------------------------------------------
# Execution helper (reuses existing formatter/exec modules)
# ---------------------------------------------------------------------------

def execute_prompt(prompt_text: str, provider: str, model: str, params: dict) -> dict:
    """Send a single prompt through the existing exec pipeline."""
    messages = [{"type": "benefit_request", "text": prompt_text}]
    turns = fmt.normalize_to_turns(messages, system_prompt=params.get("system_prompt"))

    if provider == "openai":
        payload = fmt.format_openai(model, turns, **params.get("openai", {}))
        return exe.run_openai(model, payload["input"], **params.get("openai", {}))

    if provider == "deepseek":
        payload = fmt.format_deepseek(model, turns, **params.get("deepseek", {}))
        return exe.run_deepseek(model, payload["messages"], **params.get("deepseek", {}))

    if provider == "gemini":
        payload = fmt.format_gemini(model, turns, **params.get("gemini", {}))
        return exe.run_gemini(model, payload["contents"], **params.get("gemini", {}))

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Output CSV schema for RF2 (pair-level)
# ---------------------------------------------------------------------------

RF2_COLUMNS = [
    "provider", "model", "seed_id", "variant_id", "dimension", "domain",
    "template_id", "group_a", "profile_id_a", "prompt_a", "output_a",
    "group_b", "profile_id_b", "prompt_b", "output_b",
    "label", "confidence",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seeds_df = pd.read_csv("data/rf2/seeds.csv")

    # Pre-generate all variants
    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rf2.fuzz_rf2(row, k=K)
        print(f"Generated {K} variant pairs for {row.seed_id}")

    params = {
        "system_prompt": "You are a helpful assistant.",
        "deepseek": {},
        "openai": {},
        "gemini": {},
    }

    for provider, model in PROVIDER_MODEL.items():
        out_rows = []
        logger = lg.new_run_logger(
            out_dir="logs",
            prefix=f"rf2_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RF2"},
        )

        for row in seeds_df.itertuples():
            print(f"[RF2] Executing {provider}/{model} — seed {row.seed_id}...")
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
                    res_a = execute_prompt(pa["prompt"], provider, model, params)
                    time.sleep(0.4)
                    res_b = execute_prompt(pb["prompt"], provider, model, params)

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
                        "-", "0",
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

        # Persist CSV
        out_df = pd.DataFrame(out_rows, columns=RF2_COLUMNS)
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/rf2_{provider}_{model}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()
