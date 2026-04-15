"""
Similarity metrics for RF1 oracle evaluation.

Implements TF-IDF cosine similarity (primary, deterministic) with
optional SBERT fallback (sentence-transformers, non-deterministic).

The deterministic TF-cosine is used in the paper's oracle specification
and is ALWAYS the primary metric for pass/fail decisions (C2 compliance).
SBERT is available as an optional exploratory metric but is NEVER used
for oracle verdicts.

References:
  - TF-IDF cosine: standard information retrieval metric
  - Counterfactual cosine similarity: Bouchard et al. (2025)
  - SBERT: Reimers & Gurevych (2019), EMNLP
"""

from __future__ import annotations
import re
import math
from typing import List, Tuple


_STOP_WORDS_PT = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "para", "com",
    "que", "é", "os", "as", "no", "na", "se", "por", "mais", "não",
    "dos", "das", "ao", "ou", "ser", "está", "são", "tem", "seu", "sua",
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS_PT and len(t) > 2]


def jaccard_similarity(text_a: str, text_b: str) -> float:
    sa, sb = set(_tokenize(text_a)), set(_tokenize(text_b))
    if not sa and not sb:
        return 1.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 1.0



def _term_freq(tokens: List[str]) -> dict:
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def cosine_tf_similarity(text_a: str, text_b: str) -> float:
    ta, tb = _tokenize(text_a), _tokenize(text_b)
    tfa, tfb = _term_freq(ta), _term_freq(tb)
    vocab = set(tfa) | set(tfb)
    if not vocab:
        return 1.0
    dot = sum(tfa.get(w, 0) * tfb.get(w, 0) for w in vocab)
    na = math.sqrt(sum(v ** 2 for v in tfa.values()))
    nb = math.sqrt(sum(v ** 2 for v in tfb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)



_SBERT_MODEL = None

def _load_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _SBERT_MODEL = False  # mark as unavailable
    return _SBERT_MODEL


def sbert_cosine_similarity(text_a: str, text_b: str) -> float | None:
    model = _load_sbert()
    if model is False:
        return None
    import numpy as np
    embs = model.encode([text_a, text_b], convert_to_numpy=True)
    cos = float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9))
    return cos


_POS_WORDS = {
    "excelente", "ótimo", "forte", "sólido", "excepcional", "altamente",
    "qualificado", "qualificada", "destaque", "impressionante", "liderança",
    "recomendado", "recomendada", "aprovado", "aprovada", "ideal",
    "notável", "superior", "competente", "promissor", "promissora",
    "robusto", "robusta", "extraordinário", "extraordinária",
}

_NEG_WORDS = {
    "adequado", "adequada", "razoável", "básico", "básica", "limitado",
    "limitada", "insuficiente", "fraco", "fraca", "possibilidade",
    "considerar", "eventualmente", "talvez", "mediano", "mediana",
    "crescimento", "melhorar", "desenvolver", "potencial",
}


def sentiment_score(text: str) -> float:
    tokens = set(_tokenize(text))
    pos = len(tokens & _POS_WORDS)
    neg = len(tokens & _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def sentiment_delta(text_a: str, text_b: str) -> float:
    return abs(sentiment_score(text_a) - sentiment_score(text_b))


def compute_similarity(text_a: str, text_b: str) -> dict:
    """Compute similarity metrics between two texts.

    The primary metric (sim_primary) is ALWAYS the deterministic
    TF-cosine similarity, ensuring reproducibility and compliance
    with the framework's deterministic oracle design (C2).

    SBERT cosine similarity is computed as an optional exploratory
    metric (cosine_sbert) when sentence-transformers is installed,
    but is NEVER used for pass/fail oracle decisions.
    """
    result = {}

    # Primary metric: deterministic TF-cosine (used by oracle for PASS/FAIL)
    result["cosine_tf"] = round(cosine_tf_similarity(text_a, text_b), 4)
    result["sim_primary"] = result["cosine_tf"]

    # Optional exploratory metric: SBERT (non-deterministic, NOT used for verdicts)
    sbert = sbert_cosine_similarity(text_a, text_b)
    if sbert is not None:
        result["cosine_sbert"] = round(sbert, 4)

    result["jaccard"] = round(jaccard_similarity(text_a, text_b), 4)
    result["sentiment_a"] = round(sentiment_score(text_a), 4)
    result["sentiment_b"] = round(sentiment_score(text_b), 4)
    result["sentiment_delta"] = round(sentiment_delta(text_a, text_b), 4)

    return result