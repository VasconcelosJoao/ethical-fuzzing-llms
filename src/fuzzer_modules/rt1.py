"""
RT1 Fuzzer Module — Metamorphic Testing + Generation-based Fuzzing
for Decision Opacity Detection.

Two test modes:

  METAMORPHIC:
    - Pick a pair of scenarios that should yield similar decisions
    - Both prompts already request explanation
    - Compare explanations for structural consistency
    - Metamorphic relation: similar decisions → structurally consistent explanations

  EXPLANATION:
    - Pick a scenario (Turn 1: model decides)
    - Follow up with explanation request at varying depth (Turn 2)
    - Evaluate: did the model explain? Was it accessible?

Key differences from other modules:
  - RF1/RF2: compare outputs for different attributes (same question)
  - RA2: test behavioral response to contestation/pressure
  - RT1: compare EXPLANATIONS for EQUIVALENT decisions (metamorphic)
         + test explanation provision and accessibility
"""

import random
import yaml
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

def pick_pair(pairs_data: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Pick a random metamorphic pair from the given category."""
    pool = pairs_data.get("pairs", [])
    if not pool:
        raise ValueError(f"No pairs for category '{category}'")
    return random.choice(pool)


def pick_explanation_template(
    templates_data: Dict[str, Any],
    level: str,
) -> Dict[str, Any]:
    """Pick a random explanation request template of the given level."""
    pool = templates_data.get("templates", {}).get(level, [])
    if not pool:
        raise ValueError(f"No explanation templates for level '{level}'")
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Main fuzzer entry-point
# ---------------------------------------------------------------------------

def fuzz_rt1(
    seed_row,
    k: int,
    pairs_dir: str = "data/rt1/templates",
    explanation_path: str = "data/rt1/templates/explanation_request.yaml",
) -> List[Dict[str, Any]]:
    """Generate *k* test variants for one seed.

    Parameters
    ----------
    seed_row : namedtuple-like
        Must expose: seed_id, pair_category, test_mode, explanation_level.
    k : int
        Number of variants to generate.

    Returns
    -------
    For METAMORPHIC mode — each variant has:
        - scenario_a, scenario_b: the two prompts (already ask for explanation)
        - pair_id, expected_similarity

    For EXPLANATION mode — each variant has:
        - scenario: decision prompt (Turn 1)
        - explanation_request: follow-up (Turn 2)
        - explanation_level: basic / detailed / challenge
    """
    pairs_data = read_yaml(f"{pairs_dir}/{category}.yaml")
    explanation_data = read_yaml(explanation_path)

    results: List[Dict[str, Any]] = []

    for vid in range(1, k + 1):
        pair = pick_pair(pairs_data, seed_row.pair_category)

        if seed_row.test_mode == "metamorphic":
            results.append({
                "seed_id": seed_row.seed_id,
                "variant_id": f"{seed_row.seed_id}_v{vid}",
                "test_mode": "metamorphic",
                "scenario_a": pair["scenario_a"],
                "scenario_b": pair["scenario_b"],
                "meta": {
                    "pair_category": seed_row.pair_category,
                    "pair_id": pair["id"],
                    "expected_similarity": pair["expected_similarity"],
                    "domain": pair["domain"],
                    "test_mode": "metamorphic",
                },
            })

        elif seed_row.test_mode == "explanation":
            # Use scenario_a as the decision prompt
            expl_tpl = pick_explanation_template(
                explanation_data, seed_row.explanation_level
            )

            results.append({
                "seed_id": seed_row.seed_id,
                "variant_id": f"{seed_row.seed_id}_v{vid}",
                "test_mode": "explanation",
                "scenario": pair["scenario_a"],
                "explanation_request": expl_tpl["text"],
                "explanation_level": seed_row.explanation_level,
                "meta": {
                    "pair_category": seed_row.pair_category,
                    "pair_id": pair["id"],
                    "domain": pair["domain"],
                    "test_mode": "explanation",
                    "explanation_level": seed_row.explanation_level,
                    "explanation_template_id": expl_tpl["id"],
                },
            })

        else:
            raise ValueError(f"Unknown test_mode: '{seed_row.test_mode}'")

    return results
