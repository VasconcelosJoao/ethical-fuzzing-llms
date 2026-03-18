"""
RF4 Fuzzer Module — Generation-based Fuzzing for Subgroup Fairness.

Generates the SAME benchmark question framed from EACH subgroup's perspective,
then compares accuracy and quality across ALL groups in the dimension.

Key differences from RF1/RF2:
  - RF1/RF2: pairwise comparison (2 profiles per variant)
  - RF4: ALL-GROUPS comparison (N profiles per variant, one per subgroup)
  - RF4 oracle: accuracy-based (expected elements) + four-fifths rule

The four-fifths rule (regra dos 4/5): the pass rate of any subgroup
should be at least 80% of the pass rate of the best-performing subgroup.
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

def pick_benchmark(benchmarks_data: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Pick a random benchmark from the given category."""
    pool = benchmarks_data.get("benchmarks", {}).get(category, [])
    if not pool:
        raise ValueError(f"No benchmarks for category '{category}'")
    return random.choice(pool)


def pick_framing(framing_data: Dict[str, Any], framing_type: str) -> Dict[str, Any]:
    """Pick a random framing template."""
    pool = framing_data.get("templates", {}).get(framing_type, [])
    if not pool:
        raise ValueError(f"No framing templates for type '{framing_type}'")
    return random.choice(pool)


def get_all_subgroups(
    subgroups_data: Dict[str, Any], dimension: str
) -> List[Dict[str, Any]]:
    """Return one representative from each sub-group in the dimension.

    Returns list of (profile_dict, group_name) tuples.
    """
    groups = subgroups_data.get("subgroups", {}).get(dimension, {})
    if not groups:
        raise ValueError(f"No subgroups for dimension '{dimension}'")

    result = []
    for group_name, members in groups.items():
        profile = random.choice(members)
        result.append((profile, group_name))
    return result


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

_PLACEHOLDER_MAP = {
    "{{desc}}": lambda p: p.get("desc", ""),
    "{{context}}": lambda p: p.get("context", ""),
    "{{cultural_ref}}": lambda p: p.get("cultural_ref", ""),
    "{{example_prefix}}": lambda p: p.get("example_prefix", ""),
    "{{register}}": lambda p: p.get("register", ""),
}


def render_framing(template_text: str, profile: Dict, task: str) -> str:
    """Render a framing template with profile context and benchmark task."""
    text = template_text.replace("{{task}}", task)
    for ph, fn in _PLACEHOLDER_MAP.items():
        text = text.replace(ph, fn(profile))
    return text


# ---------------------------------------------------------------------------
# Main fuzzer entry-point
# ---------------------------------------------------------------------------

def fuzz_rf4(
    seed_row,
    k: int,
    subgroups_path: str = "data/rf4/subgroups.yaml",
    benchmarks_path: str = "data/rf4/benchmarks.yaml",
    framing_path: str = "data/rf4/templates/framing.yaml",
) -> List[Dict[str, Any]]:
    """Generate *k* all-groups test variants for one seed.

    Each variant sends the SAME benchmark question to ALL subgroups
    in the dimension, framed from each subgroup's perspective.

    Parameters
    ----------
    seed_row : namedtuple-like
        Must expose: seed_id, dimension, benchmark_category, framing_type.
    k : int
        Number of variants to generate.

    Returns
    -------
    List of dicts, each with:
        - seed_id, variant_id
        - benchmark: {id, task, expected_elements, min_elements, domain}
        - prompts: list of {group, profile_id, prompt}
        - meta: dimension, benchmark, framing info
    """
    subgroups_data = read_yaml(subgroups_path)
    benchmarks_data = read_yaml(benchmarks_path)
    framing_data = read_yaml(framing_path)

    results: List[Dict[str, Any]] = []

    for vid in range(1, k + 1):
        benchmark = pick_benchmark(benchmarks_data, seed_row.benchmark_category)
        framing = pick_framing(framing_data, seed_row.framing_type)

        representatives = get_all_subgroups(subgroups_data, seed_row.dimension)

        prompts = []
        for profile, group_name in representatives:
            rendered = render_framing(
                framing["text"], profile, benchmark["task"]
            )
            prompts.append({
                "group": group_name,
                "profile_id": profile["id"],
                "prompt": rendered,
            })

        results.append({
            "seed_id": seed_row.seed_id,
            "variant_id": f"{seed_row.seed_id}_v{vid}",
            "benchmark": {
                "id": benchmark["id"],
                "task": benchmark["task"],
                "expected_elements": benchmark["expected_elements"],
                "min_elements": benchmark["min_elements"],
                "domain": benchmark["domain"],
            },
            "prompts": prompts,
            "meta": {
                "dimension": seed_row.dimension,
                "benchmark_category": seed_row.benchmark_category,
                "benchmark_id": benchmark["id"],
                "framing_type": seed_row.framing_type,
                "framing_id": framing["id"],
                "domain": benchmark["domain"],
                "n_groups": len(prompts),
            },
        })

    return results
