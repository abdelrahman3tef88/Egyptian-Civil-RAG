"""Unit tests for guardrails/pii_detector.py.

Covered without any network, GPU, or model:
  * detection of each supported Egyptian/email PII kind
  * the allow / mask / block policy decision for each kind
  * masking replaces raw values with stable tokens
  * selective masking (only chosen kinds)
  * false-positive cases (ordinary article numbers, short digit runs)
  * the raw value is never exposed in the returned scan structures
"""

import importlib.util
import sys
from pathlib import Path


# The guardrails package lives at the repo root (not under src/), so load it
# directly from its file path rather than relying on sys.path/package layout.
_PII_PATH = Path(__file__).resolve().parent.parent / "guardrails" / "pii_detector.py"
_spec = importlib.util.spec_from_file_location("pii_detector", _PII_PATH)
pii = importlib.util.module_from_spec(_spec)
sys.modules["pii_detector"] = pii
_spec.loader.exec_module(pii)


def test_no_pii_allow():
    """A normal legal question contains no PII -> allow."""
    scan = pii.detect("ما هي شروط بطلان العقد القانون المدني المصري؟")
    assert scan.has_pii is False
    assert pii.decide(scan) == pii.ALLOW


def test_egyptian_mobile_detected_and_masked():
    text = "للتواصل على 01012345678 من فضلك"
    scan = pii.detect(text)
    assert pii.EGYPTIAN_MOBILE in scan.kinds
    # Mobile default policy is mask (not block).
    assert pii.decide(scan) == pii.MASK
    masked = pii.mask(text)
    assert "01012345678" not in masked
    assert "[REDACTED_PHONE]" in masked


def test_egyptian_national_id_blocks_by_default():
    text = "الرقم القومي 29801011234567"
    scan = pii.detect(text)
    assert pii.EGYPTIAN_NATIONAL_ID in scan.kinds
    # A 14-digit national ID must NOT also be reported as a mobile number.
    assert pii.EGYPTIAN_MOBILE not in scan.kinds
    assert pii.decide(scan) == pii.BLOCK


def test_email_masked():
    text = "راسلني على ali.hassan@example.com"
    scan = pii.detect(text)
    assert pii.EMAIL in scan.kinds
    assert pii.decide(scan) == pii.MASK
    assert "ali.hassan@example.com" not in pii.mask(text)


def test_policy_override_mobile_to_block():
    scan = pii.detect("اتصل بـ 01012345678")
    action = pii.decide(scan, policy={pii.EGYPTIAN_MOBILE: "block"})
    assert action == pii.BLOCK


def test_policy_override_national_id_to_allow_but_mask_available():
    # Even if the policy is relaxed to allow, masking still removes the value.
    text = "29801011234567"
    assert "29801011234567" not in pii.mask(text)


def test_false_positive_article_numbers():
    """Ordinary article numbers and short digit runs are NOT PII."""
    assert pii.detect("المادة 121 من القانون المدني").has_pii is False
    assert pii.detect("قيمة الشيك 1234567890").has_pii is False


def test_raw_value_not_exposed():
    """The scan result must never carry the raw matched text."""
    scan = pii.detect("01012345678 و a@b.com")
    # Only kinds/counts/spans are available; the raw digits/email are not.
    serialised = repr(scan)
    assert "01012345678" not in serialised
    assert "a@b.com" not in serialised
    assert scan.describe()  # a safe human-readable summary still exists


def test_selective_masking():
    text = "a@b.com و 01012345678"
    masked = pii.mask(text, kinds=["email"])
    assert "a@b.com" not in masked  # email redacted
    assert "01012345678" in masked  # mobile left untouched


def test_empty_text():
    assert pii.detect("").has_pii is False
    assert pii.mask("") == ""
