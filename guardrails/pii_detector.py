"""PII detection, masking, and policy guardrails for the RAG request path.

Self-contained: no dependency on the retriever, vector store, prompt, or LLM
client. Three independent layers:

    detect(text)          -> which PII patterns are present (pure detection)
    mask(text, ...)       -> redact the supported patterns in place
    decide(findings, ...) -> the policy action: allow / mask / block

Actions: ``allow`` (no PII), ``mask`` (redact and continue), ``block``
(reject the request, e.g. a national ID the policy forbids forwarding).

Design rules:
* Detection, masking, and the policy decision are SEPARATE functions.
* Nothing is blocked automatically just because a pattern looks like a phone
  number; the default policy is "mask when a supported pattern exists".
* Raw detected values are NEVER stored in a returned structure. Only the
  kind, count, and character span are exposed, so callers can safely log
  "national_id x1".

LIMITATIONS OF REGEX-BASED DETECTION
------------------------------------
* It cannot prove a value is a REAL identifier, only that it LOOKS like one
  (no checksum validation is performed for national IDs).
* False positives (a long digit run) and false negatives (an identifier
  split across lines) are possible.
* English-digit oriented; Arabic-Indic digits are not normalised.

This is a guardrail, not a compliance guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# Supported PII kinds.
EGYPTIAN_MOBILE = "egyptian_mobile"
EGYPTIAN_NATIONAL_ID = "egyptian_national_id"
EMAIL = "email"

# Compiled patterns.
_EGYPTIAN_MOBILE_RE = re.compile(
    r"(?<!\d)(?:\+?20[\s-]?0?|0)(?:10|11|12|15)[\s-]?\d[\s-]?\d{3}[\s-]?\d{4}(?!\d)"
)
_EGYPTIAN_NATIONAL_ID_RE = re.compile(
    r"(?<!\d)\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3}(?!\d)"
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

_PATTERNS = {
    EGYPTIAN_NATIONAL_ID: _EGYPTIAN_NATIONAL_ID_RE,
    EGYPTIAN_MOBILE: _EGYPTIAN_MOBILE_RE,
    EMAIL: _EMAIL_RE,
}
_MASK_TOKEN = {
    EGYPTIAN_NATIONAL_ID: "[REDACTED_NATIONAL_ID]",
    EGYPTIAN_MOBILE: "[REDACTED_PHONE]",
    EMAIL: "[REDACTED_EMAIL]",
}


@dataclass
class PIIFinding:
    """A single PII occurrence, described WITHOUT its raw value."""

    kind: str
    start: int
    end: int


@dataclass
class PIIScanResult:
    """Outcome of scanning one piece of text (safe to log)."""

    findings: List[PIIFinding] = field(default_factory=list)

    @property
    def kinds(self) -> List[str]:
        return sorted({f.kind for f in self.findings})

    @property
    def has_pii(self) -> bool:
        return bool(self.findings)

    def counts(self) -> dict:
        """Return ``{kind: count}`` with NO raw values."""
        counts: dict = {}
        for f in self.findings:
            counts[f.kind] = counts.get(f.kind, 0) + 1
        return counts

    def describe(self) -> str:
        """A compact, PII-safe summary such as ``national_id x1``."""
        return ", ".join(f"{k} x{c}" for k, c in sorted(self.counts().items()))


def detect(text: str) -> PIIScanResult:
    """Scan *text* and return every supported PII occurrence.

    Pure detection: nothing is modified, nothing is blocked. A national-ID
    match occupies its span first so the same 14-digit run is not also
    (mis)reported as a mobile number.
    """
    if not text:
        return PIIScanResult()

    result = PIIScanResult()
    consumed: List[tuple] = []

    for match in _EGYPTIAN_NATIONAL_ID_RE.finditer(text):
        result.findings.append(
            PIIFinding(EGYPTIAN_NATIONAL_ID, match.start(), match.end())
        )
        consumed.append((match.start(), match.end()))

    for match in _EGYPTIAN_MOBILE_RE.finditer(text):
        span = (match.start(), match.end())
        if any(span[0] >= s and span[1] <= e for s, e in consumed):
            continue
        result.findings.append(PIIFinding(EGYPTIAN_MOBILE, match.start(), match.end()))
        consumed.append(span)

    for match in _EMAIL_RE.finditer(text):
        result.findings.append(PIIFinding(EMAIL, match.start(), match.end()))

    result.findings.sort(key=lambda f: f.start)
    return result


def mask(text: str, kinds: Optional[List[str]] = None) -> str:
    """Redact supported PII in *text* and return the cleaned string.

    If *kinds* is given, only those kinds are masked; otherwise every
    detected kind is masked with a stable token.
    """
    if not text:
        return text

    allowed = set(kinds) if kinds else set(_PATTERNS)
    spans = []
    for kind, pattern in _PATTERNS.items():
        if kind not in allowed:
            continue
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), kind))

    # De-duplicate overlaps (a longer span wins over a shorter one).
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    cleaned: List[tuple] = []
    for start, end, kind in spans:
        if any(start >= cs and end <= ce for cs, ce, _ in cleaned):
            continue
        cleaned.append((start, end, kind))

    masked = text
    for start, end, kind in sorted(cleaned, key=lambda s: s[0], reverse=True):
        masked = masked[:start] + _MASK_TOKEN[kind] + masked[end:]
    return masked


# Policy actions.
ALLOW = "allow"
MASK = "mask"
BLOCK = "block"


def decide(scan: PIIScanResult, policy: Optional[dict] = None) -> str:
    """Return the policy action (allow / mask / block) for a scan result.

    Default policy: national_id -> block; mobile/email -> mask. A caller may
    override per-kind actions via ``policy``. The strongest action across all
    detected kinds wins (block > mask > allow).
    """
    policy = policy or {}
    default_policy = {
        EGYPTIAN_NATIONAL_ID: BLOCK,
        EGYPTIAN_MOBILE: MASK,
        EMAIL: MASK,
    }
    if not scan.has_pii:
        return ALLOW

    order = {ALLOW: 0, MASK: 1, BLOCK: 2}
    strongest = ALLOW
    for kind in scan.kinds:
        action = policy.get(kind, default_policy.get(kind, MASK))
        if order[action] > order[strongest]:
            strongest = action
    return strongest
