"""LING 487 course domain knowledge: aliases, concept graph, chunk type inference, fuzzy matching.

Pure data module — no Flask/SQLAlchemy imports.  Safe to import from any service layer.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Concept alias groups — every term in a group is equivalent for retrieval.
# First entry is the *canonical* form.
# ---------------------------------------------------------------------------

_ALIAS_GROUPS: list[tuple[str, ...]] = [
    # Lecture 4 — Foundations
    ("neural network", "neural net", "nn", "neural networks", "nns"),
    ("weight", "weights", "parameter", "parameters"),
    ("forward pass", "forward propagation"),
    ("backward pass", "backward propagation"),
    ("stochastic gradient descent", "sgd", "gradient descent"),
    # Lecture 5 — Speech vectors
    ("vector", "vectors"),
    ("transformation", "transform", "transformations"),
    ("matrix", "matrices"),
    # Lecture 6 — Inner product
    ("inner product", "dot product", "scalar product"),
    ("cosine similarity", "cosine distance"),
    # Lecture 7 — DP
    ("dynamic programming", "dp"),
    ("subproblem", "subproblems"),
    ("memoization", "memoize"),
    # Lecture 8 — Backprop
    ("backpropagation", "backprop", "back propagation", "back prop"),
    ("gradient", "gradients"),
    ("chain rule",),
    ("loss function", "loss", "cost function"),
    # Lecture 9 — NN applications
    ("softmax classifier", "softmax classification"),
    ("autoencoder", "auto encoder", "auto-encoder", "autoencoders"),
    ("compression", "compress"),
    ("reconstruction", "reconstruct"),
    # Lecture 10 — Spectra / MFCCs
    ("spectra", "spectrum", "spectral"),
    ("formant", "formants"),
    ("mfcc", "mfccs", "mel frequency cepstral coefficient", "mel frequency cepstral coefficients"),
    # Lecture 11 — Bias vs Variance
    ("bias", "underfitting", "underfit"),
    ("variance", "overfitting", "overfit"),
    ("generalization", "generalisation", "generalize", "generalise"),
    # Lecture 12 — Softmax
    ("softmax",),
    ("probability", "probabilities"),
    ("distribution", "distributions"),
    # Lecture 13 — LLMs
    ("large language model", "llm", "llms", "language model"),
    ("autoregressive", "auto regressive", "auto-regressive"),
    ("next token prediction", "token prediction"),
    ("automatic speech recognition", "asr", "speech recognition"),
    ("text to speech", "tts", "speech synthesis"),
    # Lecture 14 — Attention
    ("attention mechanism", "attention"),
    ("query key value", "qkv", "q k v"),
    # Lecture 15 — Transformer
    ("transformer", "transformers"),
    ("multi-head attention", "multihead attention", "multi head attention", "mha"),
    ("feedforward", "feed forward", "feed-forward", "ffn"),
    # Lecture 16 — CNNs / Residuals
    ("convolutional neural network", "cnn", "convnet", "cnns"),
    ("residual connection", "residual", "residuals", "skip connection", "skip connections"),
    ("vanishing gradient", "vanishing gradients"),
    # Lecture 17 — Normalization / Regularization
    ("layer normalization", "layer norm", "layernorm"),
    ("dropout",),
    ("positional encoding", "positional embedding", "position encoding"),
    ("normalization", "normalisation", "normalize", "normalise"),
    ("regularization", "regularisation", "regularize", "regularise"),
    # Lecture 19 — VQ / Distillation / Diffusion
    ("vector quantization", "vq"),
    ("residual vq", "rvq", "residual vector quantization"),
    ("knowledge distillation", "distillation", "teacher student", "teacher-student"),
    ("diffusion", "diffusion model", "diffusion models"),
    # Lecture 20 — Generative AI
    ("generative ai", "generative model", "generative models"),
    ("probability distribution", "probability distributions"),
]

# --- Derived lookup structures ---

_CANONICAL: dict[str, str] = {}
_ALIAS_MAP: dict[str, frozenset[str]] = {}

def _build_alias_lookups() -> None:
    for group in _ALIAS_GROUPS:
        canonical = group[0]
        members = frozenset(group)
        for alias in group:
            _CANONICAL[alias] = canonical
            _ALIAS_MAP[alias] = members

_build_alias_lookups()


def get_canonical(term: str) -> str | None:
    return _CANONICAL.get(term.lower().strip())


def get_aliases(term: str) -> frozenset[str]:
    return _ALIAS_MAP.get(term.lower().strip(), frozenset())


def expand_term(term: str) -> list[str]:
    """All equivalent surface forms (including *term* itself), empty if unknown."""
    group = _ALIAS_MAP.get(term.lower().strip())
    return sorted(group) if group else []


def expand_terms_for_query(tokens: list[str]) -> list[str]:
    """Given query tokens, return additional tokens from alias expansion (deduplicated)."""
    seen = set(tokens)
    extra: list[str] = []
    for tok in tokens:
        for alias in get_aliases(tok):
            for word in alias.lower().split():
                if word not in seen:
                    seen.add(word)
                    extra.append(word)
    return extra


# ---------------------------------------------------------------------------
# Concept families — thematic lecture groupings
# ---------------------------------------------------------------------------

CONCEPT_FAMILIES: dict[str, dict] = {
    "neural_network_foundations": {
        "lectures": [4, 8, 11],
        "label": "Neural Network Foundations",
        "concepts": ["neural network", "backpropagation", "bias", "variance",
                      "forward pass", "backward pass", "weight", "sgd"],
    },
    "speech_processing": {
        "lectures": [5, 10],
        "label": "Speech Processing",
        "concepts": ["vector", "spectra", "formant", "mfcc", "speech recognition"],
    },
    "similarity_matching": {
        "lectures": [6],
        "label": "Similarity & Inner Product",
        "concepts": ["inner product", "cosine similarity"],
    },
    "optimization": {
        "lectures": [7, 8],
        "label": "Optimization & Search",
        "concepts": ["dynamic programming", "backpropagation", "gradient"],
    },
    "classification_output": {
        "lectures": [9, 12],
        "label": "Classification & Output Layers",
        "concepts": ["softmax", "autoencoder", "probability"],
    },
    "language_models": {
        "lectures": [13, 14, 15, 18],
        "label": "Language Models & Attention",
        "concepts": ["large language model", "attention", "transformer", "autoregressive"],
    },
    "network_architecture": {
        "lectures": [15, 16, 17],
        "label": "Architecture Components",
        "concepts": ["transformer", "cnn", "residual", "normalization", "dropout"],
    },
    "advanced_techniques": {
        "lectures": [19, 20],
        "label": "Advanced Generative Techniques",
        "concepts": ["vector quantization", "distillation", "diffusion", "generative ai"],
    },
}


def get_concept_family_for_lecture(lecture_number: int) -> str | None:
    for key, fam in CONCEPT_FAMILIES.items():
        if lecture_number in fam["lectures"]:
            return key
    return None


def get_lectures_in_family(family_key: str) -> list[int]:
    return list(CONCEPT_FAMILIES.get(family_key, {}).get("lectures", []))


# ---------------------------------------------------------------------------
# Cross-lecture adjacency graph (undirected)
# ---------------------------------------------------------------------------

_LECTURE_EDGES: dict[int, list[int]] = {
    4:  [8, 11],
    5:  [10],
    6:  [14],
    7:  [],
    8:  [4, 11],
    9:  [12, 19],
    10: [5, 13],
    11: [4, 8, 17],
    12: [9, 14],
    13: [14, 15, 18],
    14: [6, 12, 13, 15],
    15: [13, 14, 16, 17],
    16: [15, 17],
    17: [11, 15, 16],
    18: [13, 15],
    19: [9, 20],
    20: [19],
}


def get_related_lectures(lecture_number: int) -> list[int]:
    return sorted(set(_LECTURE_EDGES.get(lecture_number, [])))


# ---------------------------------------------------------------------------
# Chunk-type inference from section heading
# ---------------------------------------------------------------------------

_HEADING_TYPE_MAP: dict[str, str] = {
    "core idea":            "definition",
    "key concepts":         "definition",
    "analogy":              "analogy",
    "steps":                "process",
    "mechanism":            "process",
    "example":              "example",
    "clarification":        "clarification",
    "insight":              "insight",
    "focus":                "overview",
    "applications":         "application",
    "correlation":          "concept",
    "qkv":                  "definition",
    "multi-head attention": "definition",
    "feedforward":          "definition",
    "spectra":              "definition",
    "formants":             "definition",
    "mfccs":                "definition",
    "bias":                 "definition",
    "variance":             "definition",
    "softmax classifier":   "definition",
    "autoencoder":          "definition",
    "layer norm":           "definition",
    "dropout":              "definition",
    "positional encoding":  "definition",
    "cnn":                  "definition",
    "residuals":            "definition",
    "vector quantization":  "definition",
    "residual vq":          "definition",
    "distillation":         "definition",
    "diffusion":            "definition",
}


def infer_chunk_type(heading: str) -> str:
    h = heading.lower().strip()
    if h in _HEADING_TYPE_MAP:
        return _HEADING_TYPE_MAP[h]
    if "example" in h:
        return "example"
    if "analog" in h:
        return "analogy"
    if "step" in h:
        return "process"
    return "definition"


# ---------------------------------------------------------------------------
# Fuzzy matching — edit-distance for catching student typos
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Wagner–Fischer edit distance, no external deps."""
    m, n = len(a), len(b)
    if m < n:
        return _edit_distance(b, a)
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(m):
        curr = [i + 1]
        for j in range(n):
            curr.append(min(
                curr[j] + 1,
                prev[j + 1] + 1,
                prev[j] + (0 if a[i] == b[j] else 1),
            ))
        prev = curr
    return prev[n]


_DOMAIN_TERMS: frozenset[str] | None = None


def _get_domain_terms() -> frozenset[str]:
    global _DOMAIN_TERMS
    if _DOMAIN_TERMS is None:
        terms: set[str] = set()
        for group in _ALIAS_GROUPS:
            for alias in group:
                terms.add(alias)
                for word in alias.split():
                    if len(word) >= 4:
                        terms.add(word)
        _DOMAIN_TERMS = frozenset(terms)
    return _DOMAIN_TERMS


def fuzzy_match_domain_term(token: str, *, max_dist: int | None = None) -> str | None:
    """Best edit-distance match to a known domain term, or None if too distant."""
    t = token.lower().strip()
    if len(t) < 5:
        return None
    if max_dist is None:
        max_dist = 1 if len(t) < 8 else 2
    best: str | None = None
    best_d = max_dist + 1
    for dt in _get_domain_terms():
        if abs(len(dt) - len(t)) > max_dist:
            continue
        d = _edit_distance(t, dt)
        if d < best_d:
            best_d = d
            best = dt
    return best if best_d <= max_dist else None


def correct_typos(tokens: list[str]) -> dict[str, str]:
    """Map misspelled token → closest domain term (only for unmatched tokens)."""
    known = _get_domain_terms()
    corrections: dict[str, str] = {}
    for tok in tokens:
        if tok in known:
            continue
        match = fuzzy_match_domain_term(tok)
        if match is not None:
            corrections[tok] = match
    return corrections


# ---------------------------------------------------------------------------
# Lecture-number range extraction ("lectures 13 through 15" → [13,14,15])
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(
    r"(?:lecture|lec\.?|week)s?\s*(\d+)\s*(?:to|through|[-–—])\s*(\d+)", re.IGNORECASE
)


def extract_lecture_range(text: str) -> list[int]:
    out: list[int] = []
    for m in _RANGE_RE.finditer(text):
        lo, hi = int(m.group(1)), int(m.group(2))
        out.extend(range(lo, hi + 1))
    return sorted(set(out))
