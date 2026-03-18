import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.formatter as fmt
import src.exec_module as exe
import src.fuzzer_modules.rf1 as rf1
import src.logger as lg

import pandas as pd


PROVIDER_MODEL = {
    "gemini": "gemini-3-flash-preview",
    "openai": "gpt-5.2",
    "deepseek": "deepseek-chat",
}
K = 20  # variants per seed



def execute_prompt(prompt_text: str, provider: str, model: str, params: dict) -> dict:
    messages = [{"type": "counterfactual", "text": prompt_text}]
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


RF1_COLUMNS = [
    "provider", "model", "seed_id", "variant_id", "attribute", "domain",
    "template_id", "group_a", "profile_id_a", "prompt_a", "output_a",
    "group_b", "profile_id_b", "prompt_b", "output_b",
    "label", "confidence",
]


def main():
    seeds_df = pd.read_csv("data/rf1/seeds.csv")

    variant_cache = {}
    for row in seeds_df.itertuples():
        variant_cache[row.seed_id] = rf1.fuzz_rf1(row, k=K)
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
            prefix=f"rf1_{provider}_{model}",
            meta={"provider": provider, "model": model, "k": K, "risk": "RF1"},
        )

        for row in seeds_df.itertuples():
            print(f"[RF1] Executing {provider}/{model} — seed {row.seed_id}...")
            variants = variant_cache[row.seed_id]

            for variant in variants:
                pairs = variant["pairs"]
                if len(pairs) < 2:
                    continue  # need at least a pair

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
                        variant["meta"]["attribute"], variant["meta"]["domain"],
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

        out_df = pd.DataFrame(out_rows, columns=RF1_COLUMNS)
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/rf1_{provider}_{model}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved → {out_path}")

        logger.write("run_end", {"status": "done"})


if __name__ == "__main__":
    main()
