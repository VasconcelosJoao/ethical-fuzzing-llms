import random
import yaml
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------

def pick_scenario(
    scenarios: Dict[str, Any],
    category: str,
) -> Dict[str, Any]:
    """Pick a random scenario from the given category."""
    pool = scenarios.get("scenarios", {}).get(category, [])
    if not pool:
        raise ValueError(f"No scenarios for category '{category}'")
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------

def pick_contestation(templates: Dict[str, Any], register: str) -> Dict[str, Any]:
    """Pick a random contestation template of the given register."""
    pool = templates.get("templates", {}).get(register, [])
    if not pool:
        raise ValueError(f"No contestation templates for register '{register}'")
    return random.choice(pool)


def pick_adversarial(templates: Dict[str, Any], register: str) -> Dict[str, Any]:
    """Pick a random adversarial template of the given register."""
    pool = templates.get("templates", {}).get(register, [])
    if not pool:
        raise ValueError(f"No adversarial templates for register '{register}'")
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Main fuzzer entry-point
# ---------------------------------------------------------------------------

def fuzz_ra2(
    seed_row,
    k: int,
    scenarios_path: str = "data/ra2/scenarios.yaml",
    contestation_path: str = "data/ra2/templates/contestation.yaml",
    adversarial_path: str = "data/ra2/templates/adversarial.yaml",
) -> List[Dict[str, Any]]:
    """Generate *k* multi-turn test variants for one seed.

    Parameters
    ----------
    seed_row : namedtuple-like
        Must expose: seed_id, scenario_category, test_type, register.
    k : int
        Number of variants to generate.

    Returns
    -------
    List of dicts, each with:
        - seed_id, variant_id
        - scenario: the decision setup prompt (Turn 1)
        - challenge: the contestation/adversarial follow-up (Turn 3)
        - test_type: "contestation" or "adversarial"
        - meta: scenario and template info
    """
    scenarios_data = read_yaml(scenarios_path)

    if seed_row.test_type == "contestation":
        challenge_data = read_yaml(contestation_path)
    elif seed_row.test_type == "adversarial":
        challenge_data = read_yaml(adversarial_path)
    else:
        raise ValueError(f"Unknown test_type: '{seed_row.test_type}'")

    results: List[Dict[str, Any]] = []

    for vid in range(1, k + 1):
        scenario = pick_scenario(scenarios_data, seed_row.scenario_category)

        if seed_row.test_type == "contestation":
            challenge = pick_contestation(challenge_data, seed_row.register)
        else:
            challenge = pick_adversarial(challenge_data, seed_row.register)

        results.append({
            "seed_id": seed_row.seed_id,
            "variant_id": f"{seed_row.seed_id}_v{vid}",
            "scenario": {
                "id": scenario["id"],
                "setup": scenario["setup"],
                "domain": scenario["domain"],
                "decision_type": scenario["decision_type"],
            },
            "challenge": {
                "id": challenge["id"],
                "text": challenge["text"],
            },
            "test_type": seed_row.test_type,
            "register": seed_row.register,
            "meta": {
                "scenario_category": seed_row.scenario_category,
                "scenario_id": scenario["id"],
                "challenge_id": challenge["id"],
                "test_type": seed_row.test_type,
                "register": seed_row.register,
                "domain": scenario["domain"],
                "decision_type": scenario["decision_type"],
            },
        })

    return results
