"""
RF1 Fuzzer Module — Mutation-based + Differential Fuzzing
for Discrimination Detection.

Generates counterfactual pairs by substituting protected attributes
(gender, ethnicity, age) in seed prompts while holding all other
content constant.

Oracle criteria (applied in oracle-rf1.py):
  - TF-IDF cosine similarity < 0.80 (threshold from EEOC 4/5 rule)
  - Sentiment difference > 0.30
  - Metrics: TD (Disparity Rate), IPD (Demographic Parity Index),
             SC (Consistency Score)

References:
  - Counterfactual fairness: Kusner et al. (2017), NeurIPS
  - Four-fifths rule: EEOC Uniform Guidelines (1978)
  - Cosine similarity for fairness: Bouchard et al. (2025), JOSS
"""

import copy
import random
import yaml
import itertools
from typing import Any, Dict, List, Tuple, Optional


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def pick_demographic_pair(
    demographics: Dict[str, Any],
    attribute: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    groups = demographics[attribute]
    group_names = list(groups.keys())
    if len(group_names) < 2:
        raise ValueError(f"Attribute '{attribute}' needs ≥2 groups, got {group_names}")

    ga, gb = random.sample(group_names, 2)
    return random.choice(groups[ga]), random.choice(groups[gb]), ga, gb


def pick_all_group_representatives(
    demographics: Dict[str, Any],
    attribute: str,
) -> List[Tuple[Dict[str, Any], str]]:
    groups = demographics[attribute]
    reps = []
    for gname, members in groups.items():
        reps.append((random.choice(members), gname))
    return reps



PLACEHOLDER_MAP = {
    "{{name}}": lambda p: p.get("name", ""),
    "{{pronoun}}": lambda p: p.get("pronoun", "ele"),
    "{{pronoun_cap}}": lambda p: p.get("pronoun", "Ele").capitalize(),
    "{{title_suffix}}": lambda p: "o" if p.get("adj_suffix", "o") == "o" else "a",
    "{{adj_suffix}}": lambda p: p.get("adj_suffix", "o"),
    "{{age_desc}}": lambda p: p.get("age_desc", ""),
    "{{career_stage}}": lambda p: p.get("career_stage", ""),
    "{{origin_hint}}": lambda p: p.get("origin_hint", ""),
}

_PROFILE_DEFAULTS = {
    "pronoun": "ele",
    "adj_suffix": "o",
    "age_desc": "35 anos",
    "career_stage": "profissional com 10 anos de experiência",
    "origin_hint": "",
}


def fill_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**_PROFILE_DEFAULTS, **profile}
    return merged


def render_template(template_text: str, profile: Dict[str, Any]) -> str:
    text = template_text
    prof = fill_profile(profile)
    for ph, fn in PLACEHOLDER_MAP.items():
        text = text.replace(ph, fn(prof))
    return text


def pick_template(
    templates_dir: str,
    domain: str,
    template_key: str,
) -> Dict[str, Any]:
    data = read_yaml(f"{templates_dir}/{domain}.yaml")
    pool = data.get("templates", {}).get(template_key, [])
    if not pool:
        raise ValueError(f"No templates for domain='{domain}', key='{template_key}'")
    return random.choice(pool)


def fuzz_rf1(
    seed_row,
    k: int,
    demographics_path: str = "data/rf1/demographics.yaml",
    templates_dir: str = "data/rf1/templates",
    mode: str = "pairwise",
) -> List[Dict[str, Any]]:
    demographics = read_yaml(demographics_path)

    if seed_row.attribute not in demographics:
        raise KeyError(
            f"Attribute '{seed_row.attribute}' not in demographics "
            f"(available: {list(demographics.keys())})"
        )

    results: List[Dict[str, Any]] = []

    for vid in range(1, k + 1):
        tpl = pick_template(templates_dir, seed_row.domain, seed_row.template_key)

        if mode == "all_groups":
            reps = pick_all_group_representatives(demographics, seed_row.attribute)
            pairs = []
            for profile, group_name in reps:
                rendered = render_template(tpl["text"], profile)
                pairs.append({
                    "group": group_name,
                    "profile_id": profile["id"],
                    "prompt": rendered,
                    "messages": [{"type": "counterfactual", "text": rendered}],
                })
        else:  # pairwise
            prof_a, prof_b, ga, gb = pick_demographic_pair(
                demographics, seed_row.attribute
            )
            rendered_a = render_template(tpl["text"], prof_a)
            rendered_b = render_template(tpl["text"], prof_b)
            pairs = [
                {
                    "group": ga,
                    "profile_id": prof_a["id"],
                    "prompt": rendered_a,
                    "messages": [{"type": "counterfactual", "text": rendered_a}],
                },
                {
                    "group": gb,
                    "profile_id": prof_b["id"],
                    "prompt": rendered_b,
                    "messages": [{"type": "counterfactual", "text": rendered_b}],
                },
            ]

        results.append({
            "seed_id": seed_row.seed_id,
            "variant_id": f"{seed_row.seed_id}_v{vid}",
            "pairs": pairs,
            "meta": {
                "attribute": seed_row.attribute,
                "domain": seed_row.domain,
                "template_key": seed_row.template_key,
                "template_id": tpl["id"],
                "mode": mode,
            },
        })

    return results
