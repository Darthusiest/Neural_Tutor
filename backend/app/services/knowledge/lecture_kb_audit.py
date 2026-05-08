"""Audit and sanitize transcript-style lecture concept pack drafts.

This module implements the checklist documented in
``progress/entries/2026-05-07-lecture-kb-audit-checklist.md`` for authoring-side
quality control. It does not modify runtime retrieval/pipeline behavior unless
explicitly called by tooling.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from typing import Any

_EXPORT_STUB_RE = re.compile(r":contentReference\[oaicite:[^\]]*]\{index=\d+}")

# Common drift/overclaim patterns lifted from the checklist.
_DRIFT_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "multimodal_joint_training_overclaim",
        re.compile(r"\bmust\s+(joint[- ]?train|have aligned data)\b", re.I),
        "Avoid universal multimodal claims; alignment/training is method-dependent.",
    ),
    (
        "probing_causality_overclaim",
        re.compile(r"\b(probe accuracy|probing)\b.*\b(proves?|causal|located in)\b", re.I),
        "Probe decodability is not causal dependence by itself.",
    ),
    (
        "speech_easy_modality_overclaim",
        re.compile(r"\bspeech\s+is\s+the\s+easy\s+modality\b", re.I),
        "Avoid claiming speech is universally easy; difficulty is task/data dependent.",
    ),
    (
        "linear_probe_equals_correlation",
        re.compile(r"\b(linear probing|linear probe)\b.*\b(=|equals?)\b.*\bcorrelation\b", re.I),
        "Linear probing and correlation are different analyses; state which one is used.",
    ),
    (
        "feature_axes_native_geometry_overclaim",
        re.compile(r"\b(PCA|SVD|orthogonal)\b.*\b(native|true)\s+feature\s+directions?\b", re.I),
        "Treat analyst-imposed axes as descriptive, not guaranteed native neural geometry.",
    ),
    (
        "stats_as_rct_overclaim",
        re.compile(r"\b(partial correlation|regression)\b.*\b(controlled experiment|true relationship)\b", re.I),
        "Regression/partial correlation are association under assumptions, not randomization.",
    ),
    (
        "cca_only_baseline_overclaim",
        re.compile(r"\bCCA\b.*\b(only|sole)\b.*\bbaseline\b", re.I),
        "CCA is one multivariate baseline among others (e.g., linear probes).",
    ),
    (
        "ablation_never_multi_overclaim",
        re.compile(r"\bnever\s+ablate\s+multiple\b", re.I),
        "Multi-ablation is valid; highlight interaction/non-additivity risks instead.",
    ),
    (
        "steering_always_more_causal",
        re.compile(r"\b(counterfactual steering|steering)\b.*\balways\b.*\bmore causal\b", re.I),
        "Steering and ablation both need controls; neither is universally more causal.",
    ),
    (
        "ssm_single_scalar_memory_overclaim",
        re.compile(r"\b(single scalar k|one k)\b.*\b(explains|captures)\b.*\bmemory\b", re.I),
        "SSM memory behavior is generally modal/matrix-based beyond one scalar toy view.",
    ),
    (
        "interpretability_knob_overclaim",
        re.compile(r"\b(SSM matrices|weights)\b.*\bobvious(ly)?\b.*\bmemory knobs?\b", re.I),
        "Avoid claiming learned deep-stack parameters are directly interpretable knobs.",
    ),
    (
        "probe_safety_guarantee_overclaim",
        re.compile(r"\bprobes?\b.*\b(guarantee|detect)\b.*\b(harmful knowledge|safety)\b", re.I),
        "Probes are diagnostics; safety claims need independent evaluation/red-teaming.",
    ),
)

_TECHNICAL_SLIP_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "cosine_inner_product_confusion",
        re.compile(r"\b(cosine|cosine similarity)\b.*\b(raw inner product)\b.*\b(same|equivalent)\b", re.I),
        "Raw inner product is magnitude-sensitive; cosine assumes L2 normalization.",
    ),
    (
        "pearson_equals_cosine_without_centering",
        re.compile(r"\bPearson\b.*\bcosine\b.*\b(same|equivalent)\b", re.I),
        "Pearson r matches cosine only under centered/consistent scaling assumptions.",
    ),
    (
        "anova_f_vague_projection_ratio",
        re.compile(r"\bANOVA\s+F\b.*\bprojection\b", re.I),
        "Prefer mean-squares/error-term wording for ANOVA F.",
    ),
    (
        "pvalue_probability_hypothesis_true",
        re.compile(r"\bp[- ]?value\b.*\b(probability|chance)\b.*\b(hypothesis|null).*\btrue\b", re.I),
        "p-value is tail probability of a test statistic under H0, not P(H0 is true).",
    ),
    (
        "diffusion_negative_data_confusion",
        re.compile(r"\bdiffusion\b.*\bnegative data\b", re.I),
        "Forward corruption in diffusion is not GAN-style/classification negatives.",
    ),
    (
        "forced_alignment_universal_language_claim",
        re.compile(r"\bforced alignment\b.*\b(all|any)\s+languages?\b", re.I),
        "Forced alignment quality depends on language-specific acoustic models/lexicons.",
    ),
)

_FORBIDDEN_DRIFT_OVERBROAD_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "forbidden_drift_blanket_correlation_ban",
        re.compile(r"\bnever\s+(trust|use)\s+correlation\b", re.I),
        "Avoid blanket bans; encode specific scope mistakes instead.",
    ),
    (
        "forbidden_drift_blanket_ablation_ban",
        re.compile(r"\bnever\s+ablate\b", re.I),
        "Ablations are valid tools; scope warnings to misuse patterns.",
    ),
    (
        "forbidden_drift_always_claim",
        re.compile(r"\balways\b", re.I),
        "Prefer scoped guardrails over universal always/never phrasing.",
    ),
)

_TEXT_FIELDS: tuple[str, ...] = (
    "professor_definition",
    "key_points",
    "examples_used",
    "important_relationships",
    "constraints",
    "forbidden_drift",
)
_LIST_FIELDS = frozenset({"key_points", "examples_used", "important_relationships", "constraints", "forbidden_drift"})


@dataclass(frozen=True)
class LectureKBAuditIssue:
    severity: str
    code: str
    lecture_id: str
    concept_label: str
    field: str
    message: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class LectureKBAuditResult:
    cleaned_payload: Any
    issues: list[LectureKBAuditIssue]

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def summary_dict(self) -> dict[str, Any]:
        return {
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _snippet(value: str, *, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalized_concept_label(concept: dict[str, Any], index: int) -> str:
    for key in ("concept", "concept_name", "name", "id", "concept_id"):
        value = concept.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"concept_{index}"


def _collect_pack_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if "packs" in payload and isinstance(payload.get("packs"), list):
        return [row for row in payload["packs"] if isinstance(row, dict)]
    if "lectures" in payload and isinstance(payload.get("lectures"), list):
        return [row for row in payload["lectures"] if isinstance(row, dict)]
    if "lecture_id" in payload and isinstance(payload.get("concepts"), list):
        return [payload]
    return []


def _iter_field_items(value: Any, *, field_name: str) -> list[tuple[int | None, str]]:
    if field_name in _LIST_FIELDS:
        if isinstance(value, list):
            out: list[tuple[int | None, str]] = []
            for idx, item in enumerate(value):
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    out.append((idx, text))
            return out
        if value is None:
            return []
        text = str(value).strip()
        return [] if not text else [(None, text)]
    if value is None:
        return []
    text = str(value).strip()
    return [] if not text else [(None, text)]


def _maybe_strip_export_stub(text: str) -> tuple[str, bool]:
    cleaned = _EXPORT_STUB_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, cleaned != text


def _run_patterns(
    issues: list[LectureKBAuditIssue],
    *,
    lecture_id: str,
    concept_label: str,
    field: str,
    text: str,
) -> None:
    for code, pattern, message in _DRIFT_PATTERNS:
        if pattern.search(text):
            issues.append(
                LectureKBAuditIssue(
                    severity="warning",
                    code=code,
                    lecture_id=lecture_id,
                    concept_label=concept_label,
                    field=field,
                    message=message,
                    snippet=_snippet(text),
                )
            )
    for code, pattern, message in _TECHNICAL_SLIP_PATTERNS:
        if pattern.search(text):
            issues.append(
                LectureKBAuditIssue(
                    severity="warning",
                    code=code,
                    lecture_id=lecture_id,
                    concept_label=concept_label,
                    field=field,
                    message=message,
                    snippet=_snippet(text),
                )
            )


def audit_lecture_kb_payload(
    payload: Any,
    *,
    strip_export_stubs: bool = True,
) -> LectureKBAuditResult:
    """Audit draft lecture concept packs and optionally sanitize export stubs.

    Supported payload forms:
    - ``{"lecture_id": ..., "concepts": [...]}``
    - ``{"packs": [ ... ]}``
    - ``{"lectures": [ ... ]}``
    - ``[ ... ]`` list of lecture packs
    """

    cleaned = copy.deepcopy(payload)
    packs = _collect_pack_rows(cleaned)
    issues: list[LectureKBAuditIssue] = []

    for pack_idx, pack in enumerate(packs):
        lecture_id_raw = pack.get("lecture_id") or pack.get("lecture_number") or f"pack_{pack_idx}"
        lecture_id = str(lecture_id_raw)
        concepts = pack.get("concepts")
        if not isinstance(concepts, list):
            issues.append(
                LectureKBAuditIssue(
                    severity="error",
                    code="pack_missing_concepts",
                    lecture_id=lecture_id,
                    concept_label="(pack)",
                    field="concepts",
                    message="Each lecture pack must include a concepts[] list.",
                    snippet=_snippet(str(concepts)),
                )
            )
            continue

        for concept_idx, concept in enumerate(concepts):
            if not isinstance(concept, dict):
                issues.append(
                    LectureKBAuditIssue(
                        severity="error",
                        code="concept_not_object",
                        lecture_id=lecture_id,
                        concept_label=f"concept_{concept_idx}",
                        field="concepts",
                        message="Each concept entry must be an object.",
                        snippet=_snippet(str(concept)),
                    )
                )
                continue

            concept_label = _normalized_concept_label(concept, concept_idx)
            for field in _TEXT_FIELDS:
                if field not in concept:
                    continue
                items = _iter_field_items(concept.get(field), field_name=field)
                if not items:
                    continue

                for item_idx, raw_text in items:
                    text = raw_text
                    if strip_export_stubs:
                        stripped, changed = _maybe_strip_export_stub(raw_text)
                        if changed:
                            issues.append(
                                LectureKBAuditIssue(
                                    severity="warning",
                                    code="export_stub_removed",
                                    lecture_id=lecture_id,
                                    concept_label=concept_label,
                                    field=field,
                                    message="Removed export placeholder markup from draft content.",
                                    snippet=_snippet(raw_text),
                                )
                            )
                            text = stripped
                            if item_idx is None:
                                if field in _LIST_FIELDS:
                                    concept[field] = [text] if text else []
                                else:
                                    concept[field] = text
                            elif isinstance(concept[field], list):
                                concept[field][item_idx] = text

                    _run_patterns(
                        issues,
                        lecture_id=lecture_id,
                        concept_label=concept_label,
                        field=field,
                        text=text,
                    )

                if field == "forbidden_drift" and isinstance(concept.get(field), list):
                    for drift_item in concept[field]:
                        drift_text = str(drift_item).strip()
                        if not drift_text:
                            continue
                        for code, pattern, message in _FORBIDDEN_DRIFT_OVERBROAD_PATTERNS:
                            if pattern.search(drift_text):
                                issues.append(
                                    LectureKBAuditIssue(
                                        severity="warning",
                                        code=code,
                                        lecture_id=lecture_id,
                                        concept_label=concept_label,
                                        field=field,
                                        message=message,
                                        snippet=_snippet(drift_text),
                                    )
                                )

            pd = str(concept.get("professor_definition") or "").strip()
            if pd and re.search(r"\b(analogy|think of|imagine)\b", pd, re.I):
                issues.append(
                    LectureKBAuditIssue(
                        severity="warning",
                        code="analogy_in_professor_definition",
                        lecture_id=lecture_id,
                        concept_label=concept_label,
                        field="professor_definition",
                        message="Keep metaphors in examples_used; keep professor_definition literal/course-grounded.",
                        snippet=_snippet(pd),
                    )
                )

    return LectureKBAuditResult(cleaned_payload=cleaned, issues=issues)
