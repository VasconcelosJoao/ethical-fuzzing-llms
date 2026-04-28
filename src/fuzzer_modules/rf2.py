"""
RF2 Fuzzer Module - Generation-based + Differential Fuzzing
for Unequal Access Detection.

Generates prompt pairs parameterized by socioeconomic profiles
(income, education, geography) and compares response quality.

Oracle criteria (applied in oracle-rf2.py):
  - Length ratio < 0.80 (Atil et al., 2024)
  - Options ratio < 0.50
  - Encouragement tone delta > 0.40
  - Metrics: RE, TC, IER, EI (composite)
"""
import copy
import random
import yaml
from typing import Any, Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------

def pick_profile_pair(
    profiles: Dict[str, Any],
    dimension: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    """Return two profiles from *different* sub-groups of the given dimension.

    Returns (profile_a, profile_b, group_a_name, group_b_name).
    """
    groups = profiles[dimension]
    group_names = list(groups.keys())
    if len(group_names) < 2:
        raise ValueError(f"Dimension '{dimension}' needs ≥2 groups, got {group_names}")

    ga, gb = random.sample(group_names, 2)
    return random.choice(groups[ga]), random.choice(groups[gb]), ga, gb


def pick_all_group_representatives(
    profiles: Dict[str, Any],
    dimension: str,
) -> List[Tuple[Dict[str, Any], str]]:
    """Pick one random representative from each sub-group."""
    groups = profiles[dimension]
    return [(random.choice(members), gname) for gname, members in groups.items()]


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

PLACEHOLDER_MAP = {
    "{{profile_desc}}": lambda p: p.get("desc", ""),
    "{{occupation}}": lambda p: p.get("occupation", "profissional"),
    "{{location}}": lambda p: p.get("location", p.get("region", "")),
    "{{education_level}}": lambda p: p.get("education_level", ""),
    "{{region}}": lambda p: p.get("region", ""),
    "{{region_type}}": lambda p: p.get("region_type", ""),
}

# Default values so templates always resolve
_PROFILE_DEFAULTS = {
    "desc": "profissional",
    "occupation": "profissional",
    "location": "cidade brasileira",
    "education_level": "",
    "region": "",
    "region_type": "",
}


def fill_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Merge defaults into a profile so all placeholders resolve.
    
    Also maps 'region' → 'location' when location is absent,
    and 'education_level' → 'desc' suffix when desc is generic.
    """
    merged = {**_PROFILE_DEFAULTS, **profile}
    # geographic profiles use 'region' instead of 'location'
    if merged.get("location") == _PROFILE_DEFAULTS["location"] and merged.get("region"):
        merged["location"] = merged["region"]
    return merged


def render_template(template_text: str, profile: Dict[str, Any]) -> str:
    """Replace all known placeholders in *template_text* with profile values."""
    text = template_text
    prof = fill_profile(profile)
    for ph, fn in PLACEHOLDER_MAP.items():
        text = text.replace(ph, fn(prof))
    return text


# ---------------------------------------------------------------------------
# Template selection
# ---------------------------------------------------------------------------

def pick_template(
    templates_dir: str,
    domain: str,
    template_key: str,
) -> Dict[str, Any]:
    """Load one random template dict from the domain YAML."""
    data = read_yaml(f"{templates_dir}/{domain}.yaml")
    pool = data.get("templates", {}).get(template_key, [])
    if not pool:
        raise ValueError(f"No templates for domain='{domain}', key='{template_key}'")
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Main fuzzer entry-point
# ---------------------------------------------------------------------------

def fuzz_rf2(
    seed_row,
    k: int,
    profiles_path: str = "data/rf2/profiles.yaml",
    templates_dir: str = "data/rf2/templates",
    mode: str = "pairwise",
) -> List[Dict[str, Any]]:
    """Generate *k* counterfactual pair-sets for one seed.

    Parameters
    ----------
    seed_row : namedtuple-like
        Must expose `seed_id`, `dimension`, `domain`, `template_key`.
    k : int
        Number of variant pairs to generate.
    mode : str
        ``"pairwise"``   - each variant is a pair (group_a vs group_b).
        ``"all_groups"``  - each variant contains one prompt per sub-group.

    Returns
    -------
    List of dicts, each with:
        - seed_id, variant_id
        - pairs: list of {group, profile_id, prompt, messages}
        - meta: dimension info for later oracle evaluation
    """
    profiles = read_yaml(profiles_path)

    if seed_row.dimension not in profiles:
        raise KeyError(
            f"Dimension '{seed_row.dimension}' not in profiles "
            f"(available: {list(profiles.keys())})"
        )

    results: List[Dict[str, Any]] = []

    for vid in range(1, k + 1):
        tpl = pick_template(templates_dir, seed_row.domain, seed_row.template_key)

        if mode == "all_groups":
            reps = pick_all_group_representatives(profiles, seed_row.dimension)
            pairs = []
            for profile, group_name in reps:
                rendered = render_template(tpl["text"], profile)
                pairs.append({
                    "group": group_name,
                    "profile_id": profile["id"],
                    "prompt": rendered,
                    "messages": [{"type": "benefit_request", "text": rendered}],
                })
        else:  # pairwise
            prof_a, prof_b, ga, gb = pick_profile_pair(
                profiles, seed_row.dimension
            )
            rendered_a = render_template(tpl["text"], prof_a)
            rendered_b = render_template(tpl["text"], prof_b)
            pairs = [
                {
                    "group": ga,
                    "profile_id": prof_a["id"],
                    "prompt": rendered_a,
                    "messages": [{"type": "benefit_request", "text": rendered_a}],
                },
                {
                    "group": gb,
                    "profile_id": prof_b["id"],
                    "prompt": rendered_b,
                    "messages": [{"type": "benefit_request", "text": rendered_b}],
                },
            ]

        results.append({
            "seed_id": seed_row.seed_id,
            "variant_id": f"{seed_row.seed_id}_v{vid}",
            "pairs": pairs,
            "meta": {
                "dimension": seed_row.dimension,
                "domain": seed_row.domain,
                "template_key": seed_row.template_key,
                "template_id": tpl["id"],
                "mode": mode,
            },
        })

    return results
