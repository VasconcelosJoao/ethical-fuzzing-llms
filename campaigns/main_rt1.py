"""
main_rt1.py — Executes the RT1 (Decision Opacity) fuzzing campaign.

Two execution modes:

  METAMORPHIC: Send scenario_a and scenario_b independently (single-turn each),
               then compare explanations for structural consistency.

  EXPLANATION: Send scenario (Turn 1 → decision), then follow up with
               explanation request (Turn 2 → explanation to evaluate).
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.formatter as fmt
import src.exec_module as exe
import src.fuzzer_modules.rt1 as rt1
import src.logger as lg

import pandas as pd


PROVIDER_MODEL = {
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-5.2",
    "deepseek": "deepseek-chat",
}
K = 20


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def execute_single(prompt_text: str, provider: str, model: str, params: dict) -> dict:
    """Send a single prompt."""
    messages = [{"type": "decision_request", "text": prompt_text}]
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


def execute_multiturn(
    scenario_text: str, followup_text: str,
    provider: str, model: str, params: dict,
) -> dict:
    """Execute a two-turn conversation (decision → explanation request)."""
    # Turn 1
    res_t1 = execute_single(scenario_text, provider, model, params)
    decision_text = res_t1.get("text", "")
    time.sleep(0.4)

    # Turn 2 with history
    msgs = [
        {"type": "decision_request", "text": scenario_text},
        {"type": "assistant_decision", "text": decision_text},
        {"type": "explanation_request", "text": followup_text},
    ]
    turns = fmt.normalize_to_turns(msgs, system_prompt=params.get("system_prompt"))
    for t in turns:
        if t.msg_type == "assistant_decision":
            t.role = "assistant"

    if provider == "openai":
        payload = fmt.format_openai(model, turns, **params.get("openai", {}))
        res_t2 = exe.run_openai(model, payload["input"], **params.get("openai", {}))
    elif provider == "deepseek":
        payload = fmt.format_deepseek(model, turns, **params.get("deepseek", {}))
        res_t2 = exe.run_deepseek(model, payload["messages"], **params.get("deepseek", {}))
    elif provider == "gemini":
        payload = fmt.format_gemini(model, turns, **params.get("gemini", {}))
        res_t2 = exe.run_gemini(model, payload["contents"], **params.get("gemini", {}))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return {
        "decision_text": decision_text,
        "explanation_text": res_t2.get("text", ""),
    }


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

META_COLUMNS = [
    "provider", "model", "seed_id", "variant_id", "test_mode",
    "pair_category", "pair_id", "domain",
    "scenario_a", "output_a", "scenario_b", "output_b",
    "label", "confidence",
]

EXPL_COLUMNS = [
    "provider", "model", "seed_id", "variant_id", "test_mode",
    "pair_category", "pair_id", "domain", "explanation_level",
    "scenario", "decision_response", "explanation_request", "explanation_response",
    "label", "confidence",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seeds_df = pd.read_csv("data/rt1/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rt1.fuzz_rt1(row, k=K)
        print(f"Generated {K} variants for {row.seed_id} ({row.test_mode})")

    params = {
        "system_prompt": "You are a helpful assistant. When making recommendations or decisions, always explain your reasoning clearly.",
        "deepseek": {},
        "openai": {},
        "gemini": {},
    }

    for provider, model in PROVIDER_MODEL.items():
        meta_rows = []
        expl_rows = []
        logger = lg.new_run_logger(
            out_dir="logs",
            prefix=f"rt1_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RT1"},
        )

        for row in seeds_df.itertuples():
            print(f"[RT1] {provider}/{model} — {row.seed_id} ({row.test_mode})...")
            variants = variant_cache[row.seed_id]

            for variant in variants:
                logger.write("variant_start", {
                    "seed_id": variant["seed_id"],
                    "variant_id": variant["variant_id"],
                    "meta": variant["meta"],
                })

                try:
                    if variant["test_mode"] == "metamorphic":
                        res_a = execute_single(
                            variant["scenario_a"], provider, model, params
                        )
                        time.sleep(0.4)
                        res_b = execute_single(
                            variant["scenario_b"], provider, model, params
                        )

                        logger.write("variant_result", {
                            "variant_id": variant["variant_id"],
                            "output_a": lg.safe_preview(res_a.get("text")),
                            "output_b": lg.safe_preview(res_b.get("text")),
                            "status": "ok",
                        })

                        meta_rows.append([
                            provider, model,
                            variant["seed_id"], variant["variant_id"],
                            "metamorphic",
                            variant["meta"]["pair_category"],
                            variant["meta"]["pair_id"],
                            variant["meta"]["domain"],
                            variant["scenario_a"],
                            res_a.get("text", ""),
                            variant["scenario_b"],
                            res_b.get("text", ""),
                            "-", "0",
                        ])

                    elif variant["test_mode"] == "explanation":
                        result = execute_multiturn(
                            variant["scenario"],
                            variant["explanation_request"],
                            provider, model, params,
                        )

                        logger.write("variant_result", {
                            "variant_id": variant["variant_id"],
                            "decision": lg.safe_preview(result["decision_text"]),
                            "explanation": lg.safe_preview(result["explanation_text"]),
                            "status": "ok",
                        })

                        expl_rows.append([
                            provider, model,
                            variant["seed_id"], variant["variant_id"],
                            "explanation",
                            variant["meta"]["pair_category"],
                            variant["meta"]["pair_id"],
                            variant["meta"]["domain"],
                            variant["explanation_level"],
                            variant["scenario"],
                            result["decision_text"],
                            variant["explanation_request"],
                            result["explanation_text"],
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

        # Persist CSVs
        os.makedirs("outputs", exist_ok=True)

        if meta_rows:
            pd.DataFrame(meta_rows, columns=META_COLUMNS).to_csv(
                f"outputs/rt1_meta_{provider}_{model}.csv", index=False
            )
        if expl_rows:
            pd.DataFrame(expl_rows, columns=EXPL_COLUMNS).to_csv(
                f"outputs/rt1_expl_{provider}_{model}.csv", index=False
            )
        print(f"Saved outputs for {provider}/{model}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()
