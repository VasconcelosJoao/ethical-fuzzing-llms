"""
RT2 Fuzzer Module — Mutation-based + Differential Fuzzing
for Hidden Bias Detection.

Generates counterfactual pairs where ONLY an IRRELEVANT attribute
differs between prompts. If the model's output changes, it reveals
hidden/implicit biases — the model is influenced by information
that should have no bearing on the decision.

Key difference from RF1:
  - RF1: perturbs PROTECTED attributes (gender, ethnicity, age)
         → detects explicit/implicit discrimination
  - RT2: perturbs IRRELEVANT attributes (hobbies, food, music, pets)
         → detects hidden biases in reasoning

The oracle verifies INVARIANCE: outputs should be essentially
identical regardless of irrelevant attribute changes.
"""

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
# Selection helpers
# ---------------------------------------------------------------------------

def pick_perturbation_pair(
    perturbations: Dict[str, Any],
    category: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pick two DIFFERENT perturbation values from the same category.

    Returns (perturbation_a, perturbation_b).
    """
    pool = perturbations.get("perturbations", {}).get(category, [])
    if len(pool) < 2:
        raise ValueError(f"Category '{category}' needs ≥2 perturbations, got {len(pool)}")
    return tuple(random.sample(pool, 2))


def pick_template(
    templates_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Pick a random template from the given domain."""
    pool = templates_data.get("templates", [])
    if not pool:
        raise ValueError(f"No templates for domain '{domain}'")
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def render_with_perturbation(template_text: str, perturbation: Dict) -> str:
    """Replace {{perturbation}} placeholder with the perturbation snippet."""
    return template_text.replace("{{perturbation}}", perturbation.get("snippet", ""))


def render_without_perturbation(template_text: str) -> str:
    """Remove {{perturbation}} placeholder entirely (control baseline)."""
    return template_text.replace("{{perturbation}}", "").replace("  ", " ").strip()


# ---------------------------------------------------------------------------
# Main fuzzer entry-point
# ---------------------------------------------------------------------------

def fuzz_rt2(
    seed_row,
    k: int,
    perturbations_path: str = "data/rt2/perturbations.yaml",
    templates_dir: str = "data/rt2/templates",
) -> List[Dict[str, Any]]:
    """Generate *k* counterfactual pairs for one seed.

    Each variant contains:
      - prompt_a: template + perturbation_a (e.g. "gosto de futebol")
      - prompt_b: template + perturbation_b (e.g. "pratico ballet")

    The oracle compares output_a vs output_b for invariance.

    Parameters
    ----------
    seed_row : namedtuple-like
        Must expose: seed_id, perturbation_category, domain, template_key.
    k : int
        Number of variant pairs to generate.

    Returns
    -------
    List of dicts with paired prompts and metadata.
    """
    perturbations = read_yaml(perturbations_path)
    templates_data = read_yaml(f"{templates_dir}/{seed_row.template_key}.yaml")

    results: List[Dict[str, Any]] = []

    for vid in range(1, k + 1):
        tpl = pick_template(templates_data, seed_row.template_key)
        pert_a, pert_b = pick_perturbation_pair(
            perturbations, seed_row.perturbation_category
        )

        rendered_a = render_with_perturbation(tpl["text"], pert_a)
        rendered_b = render_with_perturbation(tpl["text"], pert_b)

        results.append({
            "seed_id": seed_row.seed_id,
            "variant_id": f"{seed_row.seed_id}_v{vid}",
            "prompt_a": rendered_a,
            "prompt_b": rendered_b,
            "perturbation_a": {
                "id": pert_a["id"],
                "label": pert_a["label"],
                "snippet": pert_a["snippet"],
            },
            "perturbation_b": {
                "id": pert_b["id"],
                "label": pert_b["label"],
                "snippet": pert_b["snippet"],
            },
            "meta": {
                "perturbation_category": seed_row.perturbation_category,
                "domain": seed_row.domain,
                "template_id": tpl["id"],
                "perturbation_a_id": pert_a["id"],
                "perturbation_b_id": pert_b["id"],
                "perturbation_a_label": pert_a["label"],
                "perturbation_b_label": pert_b["label"],
            },
        })

    return results
