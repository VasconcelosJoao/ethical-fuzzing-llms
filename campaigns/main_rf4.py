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
sys.path.append(os.path.abspath("ethical_fuzzing/src/"))

import src.formatter as fmt
import src.exec_module as exe
import src.fuzzer_modules.rf4 as rf4
import src.logger as lg

import pandas as pd


PROVIDER_MODEL = {
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-5.2",
    "deepseek": "deepseek-chat",
}
K = 20


# ---------------------------------------------------------------------------
# Execution helper
# ---------------------------------------------------------------------------

def execute_prompt(prompt_text: str, provider: str, model: str, params: dict) -> dict:
    """Send a single prompt."""
    messages = [{"type": "benchmark_request", "text": prompt_text}]
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
# Output CSV schema (one row per group per variant)
# ---------------------------------------------------------------------------

RF4_COLUMNS = [
    "provider", "model", "seed_id", "variant_id",
    "dimension", "benchmark_id", "domain", "benchmark_task",
    "group", "profile_id", "prompt", "output",
    "expected_elements", "min_elements",
    "label", "confidence",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seeds_df = pd.read_csv("data/rf4/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rf4.fuzz_rf4(row, k=K)
        print(f"Generated {K} variants for {row.seed_id}")

    params = {
        "system_prompt": "You are a helpful assistant. Provide complete and accurate information.",
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
                        result = execute_prompt(
                            prompt_info["prompt"], provider, model, params
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
                            "-", "0",
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
