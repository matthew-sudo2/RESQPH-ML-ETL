"""
NLP-based citizen incident report triage for ResQPH.

Parses free-text emergency distress messages to detect medical needs,
vulnerable individuals, flood depth signals, and assigns an urgency
triage priority category ('low', 'medium', 'high', 'critical').
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.utils import config

logger = config.get_logger(__name__)

# Keywords for entity extraction
MEDICAL_KEYWORDS = [
    "insulin", "diabetes", "dialysis", "oxygen", "stroke", "seizure",
    "heart attack", "bleeding", "injured", "injury", "fracture", "unconscious",
    "fainted", "asthma", "inhaler", "medication", "medicine", "sick",
    "fever", "wound", "vital", "labor", "contractions", "prescription",
]

VULNERABLE_KEYWORDS = [
    "grandmother", "grandfather", "grandma", "grandpa", "lola", "lolo",
    "elderly", "senior", "old person", "senior citizen", "baby", "infant",
    "newborn", "toddler", "child", "children", "kid", "kids", "pregnant",
    "buntis", "disabled", "disability", "pwd", "wheelchair", "bedridden",
    "blind", "special needs",
]

DEPTH_PATTERNS = [
    (r"(?:roof|rooftop|ceiling|attic|second floor|2nd floor)", 2.2),
    (r"(?:neck[- ]deep|neck[- ]high|hanggang leeg)", 1.5),
    (r"(?:chest[- ]deep|chest[- ]high|hanggang dibdib)", 1.2),
    (r"(?:submerged|lubog|underwater)", 1.4),
    (r"(?:waist[- ]deep|waist[- ]high|hanggang baywang|hanggang kiap)", 0.8),
    (r"(?:knee[- ]deep|knee[- ]high|hanggang tuhod)", 0.4),
    (r"(?:ankle[- ]deep|ankle[- ]high|hanggang bukong-bukong)", 0.15),
]


@dataclass
class TriageResult:
    """Structured extraction output for citizen emergency text."""

    priority: str
    medical: bool
    vulnerable: bool
    flood_depth_m: float
    urgency_score: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "medical": self.medical,
            "vulnerable": self.vulnerable,
            "flood_depth_m": self.flood_depth_m,
            "urgency_score": self.urgency_score,
            "matched_keywords": self.matched_keywords,
        }


def extract_flood_depth(text_lower: str) -> float:
    """Extract estimated flood depth in meters from text."""
    # Check numeric expressions like '1.5 m' or '4 ft'
    m_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metro)\b", text_lower)
    if m_match:
        try:
            return float(m_match.group(1))
        except ValueError:
            pass

    ft_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|talampakan)\b", text_lower)
    if ft_match:
        try:
            return round(float(ft_match.group(1)) * 0.3048, 2)
        except ValueError:
            pass

    # Keyword depth pattern check
    for pattern, depth in DEPTH_PATTERNS:
        if re.search(pattern, text_lower):
            return depth

    return 0.0


def triage_text(text: str) -> TriageResult:
    """
    Extract emergency entities and determine triage priority for citizen report text.

    Args:
        text: Free-text emergency report.

    Returns:
        TriageResult dataclass with priority, flags, and estimated flood depth.
    """
    cleaned = str(text or "").strip()
    text_lower = cleaned.lower()

    matched: List[str] = []

    # Check medical
    medical_flag = False
    for kw in MEDICAL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            medical_flag = True
            matched.append(f"medical:{kw}")

    # Check vulnerable
    vulnerable_flag = False
    for kw in VULNERABLE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text_lower):
            vulnerable_flag = True
            matched.append(f"vulnerable:{kw}")

    # Check depth & entrapment
    depth_m = extract_flood_depth(text_lower)
    is_trapped = bool(re.search(r"\b(trapped|stranded|isolate|saklolo|rescue|tulong)\b", text_lower))

    # Priority determination rule engine
    # Critical: Trapped in deep water with vulnerable/medical needs, or roof/ceiling flooding
    if (depth_m >= 1.0 and (medical_flag or vulnerable_flag)) or (depth_m >= 2.0) or (medical_flag and vulnerable_flag and is_trapped):
        priority = "critical"
        urgency = 0.95
    elif medical_flag or (depth_m >= 0.7) or (vulnerable_flag and depth_m >= 0.4) or is_trapped:
        priority = "high"
        urgency = 0.75
    elif depth_m >= 0.3 or "impassable" in text_lower or "stranded" in text_lower:
        priority = "medium"
        urgency = 0.50
    else:
        priority = "low"
        urgency = 0.20

    if priority not in config.NLP_TRIAGE_LABELS:
        priority = "medium"

    return TriageResult(
        priority=priority,
        medical=medical_flag,
        vulnerable=vulnerable_flag,
        flood_depth_m=depth_m,
        urgency_score=urgency,
        matched_keywords=matched,
        raw_text=cleaned,
    )


def triage_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply NLP triage across a DataFrame of citizen reports."""
    results = [triage_text(t) for t in df[text_col]]
    out = df.copy()
    out["predicted_priority"] = [r.priority for r in results]
    out["is_medical"] = [r.medical for r in results]
    out["is_vulnerable"] = [r.vulnerable for r in results]
    out["est_flood_depth_m"] = [r.flood_depth_m for r in results]
    out["urgency_score"] = [r.urgency_score for r in results]
    return out


def evaluate_nlp(df: pd.DataFrame, target_col: str = "priority_label") -> Dict[str, Any]:
    """Evaluate triage accuracy and classification metrics."""
    scored = triage_dataframe(df, text_col="text")
    if target_col in scored.columns:
        y_true = scored[target_col].astype(str).str.lower()
        y_pred = scored["predicted_priority"].astype(str).str.lower()
        acc = float(np.mean(y_true == y_pred))
        report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        return {"accuracy": acc, "report": report, "sample_count": len(scored)}
    return {"sample_count": len(scored), "predicted_distribution": scored["predicted_priority"].value_counts().to_dict()}
