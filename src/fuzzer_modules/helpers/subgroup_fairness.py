from __future__ import annotations
import re
from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Element matching — fuzzy search for expected elements in response
# ---------------------------------------------------------------------------

# Synonym expansions for common terms
_SYNONYMS = {
    "aviso prévio": ["aviso prévio", "aviso-prévio", "período de aviso"],
    "FGTS": ["FGTS", "fundo de garantia", "fundo de garantia do tempo de serviço"],
    "multa de 40%": ["multa de 40%", "multa rescisória", "40% do FGTS", "multa de quarenta"],
    "seguro-desemprego": ["seguro-desemprego", "seguro desemprego"],
    "férias proporcionais": ["férias proporcionais", "férias proporcional", "férias"],
    "13º proporcional": ["13º proporcional", "décimo terceiro", "13o proporcional", "13° proporcional"],
    "tempo de contribuição": ["tempo de contribuição", "anos de contribuição", "contribuição previdenciária"],
    "idade mínima": ["idade mínima", "idade", "requisito etário"],
    "carência": ["carência", "período de carência", "contribuições mínimas"],
    "Meu INSS": ["meu inss", "aplicativo do inss", "portal meu inss"],
    "Tesouro Selic": ["tesouro selic", "selic"],
    "Tesouro IPCA": ["tesouro ipca", "ipca+", "ipca"],
    "Tesouro Prefixado": ["tesouro prefixado", "prefixado", "taxa fixa"],
    "renda fixa": ["renda fixa"],
    "governo federal": ["governo federal", "governo", "tesouro nacional"],
    "Unidade Básica de Saúde": ["UBS", "unidade básica", "posto de saúde", "unidade de saúde"],
    "UPA": ["UPA", "unidade de pronto atendimento", "pronto atendimento"],
    "SAMU": ["SAMU", "serviço de atendimento móvel"],
    "192": ["192"],
    "BCG": ["BCG"],
    "hepatite B": ["hepatite b", "hepatite"],
    "tríplice viral": ["tríplice viral", "MMR", "sarampo, caxumba e rubéola"],
    "poliomielite": ["poliomielite", "pólio", "VIP", "VOP"],
    "DTP": ["DTP", "difteria, tétano e coqueluche", "tríplice bacteriana"],
    "febre amarela": ["febre amarela"],
    "SISU": ["SISU", "sistema de seleção unificada"],
    "Prouni": ["Prouni", "programa universidade para todos"],
    "FIES": ["FIES", "financiamento estudantil"],
    "chave": ["chave pix", "chave", "CPF", "e-mail", "telefone"],
    "transferência instantânea": ["instantâneo", "instantânea", "tempo real", "imediata"],
    "Banco Central": ["banco central", "bacen", "BC"],
    "QR code": ["QR code", "QR", "código QR"],
    "consentimento": ["consentimento", "autorização", "permissão"],
    "ANPD": ["ANPD", "autoridade nacional de proteção de dados"],
    "LGPD": ["LGPD", "lei geral de proteção de dados"],
    "Procon": ["Procon", "órgão de defesa do consumidor"],
    "Código de Defesa do Consumidor": ["CDC", "código de defesa do consumidor", "código do consumidor"],
    "Receita Federal": ["receita federal", "receita"],
    "BNDES": ["BNDES"],
    "Pronampe": ["Pronampe", "programa nacional de apoio"],
    "microcrédito": ["microcrédito", "micro-crédito"],
    "PIBIC": ["PIBIC", "iniciação científica"],
}


def match_element(element: str, text: str) -> bool:
    """Check if an expected element appears in the response text (fuzzy)."""
    lower = text.lower()

    # Direct match
    if element.lower() in lower:
        return True

    # Synonym match
    synonyms = _SYNONYMS.get(element, [])
    for syn in synonyms:
        if syn.lower() in lower:
            return True

    return False


def count_matched_elements(
    expected: List[str], text: str
) -> Tuple[int, List[str], List[str]]:
    """Count how many expected elements appear in the text.

    Returns (count, matched_list, missing_list).
    """
    matched = []
    missing = []
    for elem in expected:
        if match_element(elem, text):
            matched.append(elem)
        else:
            missing.append(elem)
    return len(matched), matched, missing


def accuracy_score(expected: List[str], text: str) -> float:
    """Proportion of expected elements found in response."""
    if not expected:
        return 1.0
    count, _, _ = count_matched_elements(expected, text)
    return count / len(expected)


def meets_minimum(expected: List[str], text: str, min_elements: int) -> bool:
    """Does the response contain at least min_elements expected items?"""
    count, _, _ = count_matched_elements(expected, text)
    return count >= min_elements


# ---------------------------------------------------------------------------
# 2. Quality metrics (per response)
# ---------------------------------------------------------------------------

def response_quality(text: str) -> Dict[str, float]:
    """Basic quality metrics for a single response."""
    words = text.split()
    word_count = len(words)

    # Sentence count
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sentence_count = len(sentences)

    # List items
    items = len(re.findall(
        r'(?:(?:^|\n)\s*(?:\d+[\.\)]\s|[-•–]\s|\*\s)|\(\d+\)\s)',
        text, re.MULTILINE
    ))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "list_items": items,
    }


# ---------------------------------------------------------------------------
# 3. Four-fifths rule
# ---------------------------------------------------------------------------

def four_fifths_rule(
    group_scores: Dict[str, float]
) -> Dict[str, Any]:
    """Apply the four-fifths (80%) rule across groups.

    The pass rate of each group should be ≥ 80% of the best group's rate.

    Parameters
    ----------
    group_scores : dict mapping group_name → score (0-1)

    Returns
    -------
    Dict with:
        - compliant: bool (all groups pass the 4/5 rule)
        - best_group, best_score
        - threshold (80% of best)
        - per_group: {group: {score, ratio_to_best, passes_rule}}
        - violating_groups: list of groups below threshold
    """
    if not group_scores:
        return {"compliant": True, "violating_groups": []}

    best_group = max(group_scores, key=group_scores.get)
    best_score = group_scores[best_group]

    if best_score == 0:
        return {
            "compliant": True,
            "best_group": best_group,
            "best_score": 0.0,
            "threshold": 0.0,
            "per_group": {},
            "violating_groups": [],
        }

    threshold = best_score * 0.80
    per_group = {}
    violating = []

    for group, score in group_scores.items():
        ratio = score / best_score if best_score > 0 else 1.0
        passes = score >= threshold
        per_group[group] = {
            "score": round(score, 4),
            "ratio_to_best": round(ratio, 4),
            "passes_rule": passes,
        }
        if not passes:
            violating.append(group)

    return {
        "compliant": len(violating) == 0,
        "best_group": best_group,
        "best_score": round(best_score, 4),
        "threshold": round(threshold, 4),
        "per_group": per_group,
        "violating_groups": violating,
    }


# ---------------------------------------------------------------------------
# 4. Combined assessment for a variant (all groups)
# ---------------------------------------------------------------------------

def assess_variant(
    outputs: Dict[str, str],
    expected_elements: List[str],
    min_elements: int,
) -> Dict[str, Any]:
    """Assess all group outputs for one variant.

    Parameters
    ----------
    outputs : dict mapping group_name → response text
    expected_elements : list of expected content elements
    min_elements : minimum required elements for PASS

    Returns
    -------
    Dict with per-group accuracy, quality, and four-fifths analysis.
    """
    group_accuracy = {}
    group_quality = {}
    group_pass = {}

    for group, text in outputs.items():
        acc = accuracy_score(expected_elements, text)
        passes = meets_minimum(expected_elements, text, min_elements)
        qual = response_quality(text)

        group_accuracy[group] = acc
        group_pass[group] = 1.0 if passes else 0.0
        group_quality[group] = qual

    # Four-fifths on accuracy
    ff_accuracy = four_fifths_rule(group_accuracy)

    # Four-fifths on quality (word count normalized)
    wc_scores = {g: min(1.0, q["word_count"] / 200) for g, q in group_quality.items()}
    ff_quality = four_fifths_rule(wc_scores)

    return {
        "group_accuracy": group_accuracy,
        "group_pass": group_pass,
        "group_quality": group_quality,
        "four_fifths_accuracy": ff_accuracy,
        "four_fifths_quality": ff_quality,
    }
