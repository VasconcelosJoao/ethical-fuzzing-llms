"""
main_ra2.py - Executes the RA2 (Contestability) fuzzing campaign.

Unlike RF1/RF2 (single-turn pairs), RA2 runs MULTI-TURN conversations:
  Turn 1: User presents scenario → model makes a decision
  Turn 2: User contests/pressures → model responds

The oracle evaluates the model's response to contestation/adversarial pressure.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.formatter as fmt
import src.exec_module as exe
import src.fuzzer_modules.ra2 as ra2
import src.logger as lg

import pandas as pd
from config import PROVIDER_MODEL, K, SYSTEM_PROMPTS


# ---------------------------------------------------------------------------
# Multi-turn execution
# ---------------------------------------------------------------------------

def execute_multiturn(
    scenario_text: str,
    challenge_text: str,
    provider: str,
    model: str,
    params: dict,
) -> dict:
    """Execute a two-turn conversation and return both responses.

    Turn 1: scenario_text → decision_response
    Turn 2: challenge_text → challenge_response (what the oracle evaluates)
    """
    # --- Turn 1: get the model's initial decision ---
    res_t1 = exe.execute_single(scenario_text, "scenario", provider, model, params)
    decision_text = res_t1.get("text", "")

    time.sleep(0.4)

    # --- Turn 2: send challenge with conversation history ---
    system_prompt = params.get("system_prompt")
    msgs_t2 = [
        {"type": "scenario", "text": scenario_text},
        {"type": "assistant_decision", "text": decision_text},
        {"type": "challenge", "text": challenge_text},
    ]
    turns_t2 = fmt.normalize_to_turns(msgs_t2, system_prompt=system_prompt)

    for t in turns_t2:
        if t.msg_type == "assistant_decision":
            t.role = "assistant"

    if provider == "openai":
        payload_t2 = fmt.format_openai(model, turns_t2, **params.get("openai", {}))
        res_t2 = exe.run_openai(model, payload_t2["input"], **params.get("openai", {}))
    elif provider == "deepseek":
        payload_t2 = fmt.format_deepseek(model, turns_t2, **params.get("deepseek", {}))
        res_t2 = exe.run_deepseek(model, payload_t2["messages"], **params.get("deepseek", {}))
    elif provider == "gemini":
        payload_t2 = fmt.format_gemini(model, turns_t2, **params.get("gemini", {}))
        res_t2 = exe.run_gemini(model, payload_t2["contents"], **params.get("gemini", {}))
    else:
        raise ValueError(f"Unknown provider: {provider}")

    challenge_response = res_t2.get("text", "")

    return {
        "decision_text": decision_text,
        "challenge_response": challenge_response,
        "provider": provider,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Output CSV schema
# ---------------------------------------------------------------------------

RA2_COLUMNS = [
    "provider", "model", "seed_id", "variant_id",
    "scenario_category", "scenario_id", "domain", "decision_type",
    "test_type", "register", "challenge_id",
    "scenario_prompt", "decision_response", "challenge_prompt", "challenge_response",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    seeds_df = pd.read_csv("data/ra2/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = ra2.fuzz_ra2(row, k=K)
        print(f"Generated {K} variants for {row.seed_id}")

    params = {
        "system_prompt": SYSTEM_PROMPTS["RA2"],
        "deepseek": {},
        "openai": {},
        "gemini": {},
    }

    for provider, model in PROVIDER_MODEL.items():
        out_rows = []
        logger = lg.new_run_logger(
            out_dir=os.path.join("execution_logs", "ra2"),
            prefix=f"ra2_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RA2"},
        )

        for row in seeds_df.itertuples():
            print(f"[RA2] Executing {provider}/{model} - seed {row.seed_id} ({row.test_type}/{row.register})...")
            variants = variant_cache[row.seed_id]

            for variant in variants:
                logger.write("variant_start", {
                    "seed_id": variant["seed_id"],
                    "variant_id": variant["variant_id"],
                    "meta": variant["meta"],
                })

                try:
                    result = execute_multiturn(
                        scenario_text=variant["scenario"]["setup"],
                        challenge_text=variant["challenge"]["text"],
                        provider=provider,
                        model=model,
                        params=params,
                    )

                    logger.write("variant_result", {
                        "variant_id": variant["variant_id"],
                        "decision_preview": lg.safe_preview(result["decision_text"]),
                        "challenge_preview": lg.safe_preview(result["challenge_response"]),
                        "status": "ok",
                    })

                    out_rows.append([
                        provider, model,
                        variant["seed_id"], variant["variant_id"],
                        variant["meta"]["scenario_category"],
                        variant["meta"]["scenario_id"],
                        variant["meta"]["domain"],
                        variant["meta"]["decision_type"],
                        variant["test_type"], variant["register"],
                        variant["meta"]["challenge_id"],
                        variant["scenario"]["setup"],
                        result["decision_text"],
                        variant["challenge"]["text"],
                        result["challenge_response"],
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

        out_df = pd.DataFrame(out_rows, columns=RA2_COLUMNS)
        out_dir = os.path.join("campaign_outputs", "ra2")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ra2_{provider}_{model}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()