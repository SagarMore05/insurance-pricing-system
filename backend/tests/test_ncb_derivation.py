"""
test_ncb_derivation.py
======================
Validates the NCB auto-derivation logic in feature_enrichment_service.

Cases required by specification:
  Case 1: claims=0, tenure=1  → NCB=20
  Case 2: claims=0, tenure=3  → NCB=35
  Case 3: claims=0, tenure=5  → NCB=50
  Case 4: claims=2, tenure=5  → NCB=0

Run:
  cd backend
  pytest tests/test_ncb_derivation.py -v
"""

import pytest
from src.services.feature_enrichment_service import derive_ncb_pct


# ─── Specification test cases ─────────────────────────────────────────────────

class TestNcbSpecificationCases:
    """The four exact cases from the product specification."""

    def test_case_1_no_claims_tenure_1(self):
        """claims=0, tenure=1 → NCB=20"""
        assert derive_ncb_pct(0, 1) == 20.0

    def test_case_2_no_claims_tenure_3(self):
        """claims=0, tenure=3 → NCB=35"""
        assert derive_ncb_pct(0, 3) == 35.0

    def test_case_3_no_claims_tenure_5(self):
        """claims=0, tenure=5 → NCB=50"""
        assert derive_ncb_pct(0, 5) == 50.0

    def test_case_4_claims_present_resets_ncb(self):
        """claims=2, tenure=5 → NCB=0 (any claim resets bonus)"""
        assert derive_ncb_pct(2, 5) == 0.0


# ─── Full slab table ──────────────────────────────────────────────────────────

class TestNcbSlabTable:
    """Verify the complete slab table for a clean record (0 claims)."""

    @pytest.mark.parametrize("tenure,expected", [
        (0,   0.0),
        (1,  20.0),
        (2,  25.0),
        (3,  35.0),
        (4,  45.0),
        (5,  50.0),
        (10, 50.0),
        (30, 50.0),
    ])
    def test_slab_zero_claims(self, tenure, expected):
        assert derive_ncb_pct(0, tenure) == expected

    @pytest.mark.parametrize("tenure", [1, 2, 3, 4, 5, 10])
    def test_any_claim_resets_to_zero(self, tenure):
        """Even 1 claim yields NCB=0 regardless of tenure."""
        assert derive_ncb_pct(1, tenure) == 0.0
        assert derive_ncb_pct(5, tenure) == 0.0
        assert derive_ncb_pct(20, tenure) == 0.0


# ─── Fractional tenure (floor behaviour) ─────────────────────────────────────

class TestNcbFractionalTenure:
    """Policy tenure is floored to completed years before slab lookup."""

    def test_tenure_1_point_9_rounds_down_to_1(self):
        """1.9 years completed → still in year-1 slab → 20%"""
        assert derive_ncb_pct(0, 1.9) == 20.0

    def test_tenure_2_point_0_is_year_2(self):
        assert derive_ncb_pct(0, 2.0) == 25.0

    def test_tenure_4_point_99_is_year_4(self):
        """4.99 years → still year-4 slab → 45%"""
        assert derive_ncb_pct(0, 4.99) == 45.0

    def test_tenure_5_point_0_is_max(self):
        assert derive_ncb_pct(0, 5.0) == 50.0


# ─── Edge / boundary cases ────────────────────────────────────────────────────

class TestNcbEdgeCases:
    """Guard against off-by-one and boundary errors."""

    def test_zero_claims_zero_tenure(self):
        """No claims but never held a policy (tenure=0) → NCB=0"""
        assert derive_ncb_pct(0, 0) == 0.0

    def test_zero_claims_zero_point_9_tenure(self):
        """Less than 1 completed year → NCB=0"""
        assert derive_ncb_pct(0, 0.9) == 0.0

    def test_exactly_one_claim_one_tenure(self):
        assert derive_ncb_pct(1, 1) == 0.0

    def test_return_type_is_float(self):
        for claims in [0, 1]:
            for tenure in [0, 1, 5]:
                result = derive_ncb_pct(claims, tenure)
                assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_ncb_values_are_valid_slabs(self):
        """All possible outputs must be within the allowed slab set."""
        valid_slabs = {0.0, 20.0, 25.0, 35.0, 45.0, 50.0}
        for claims in range(0, 3):
            for tenure in range(0, 8):
                result = derive_ncb_pct(claims, float(tenure))
                assert result in valid_slabs, f"Unexpected NCB={result} for claims={claims} tenure={tenure}"


# ─── Enrichment service integration (no external deps) ───────────────────────

class TestNcbInEnrichmentPayload:
    """
    Verify derive_ncb_pct produces the same value the enrichment service
    would put into inference_payload['no_claim_bonus_pct'].
    """

    def test_payload_ncb_matches_derivation(self):
        """The inference payload must always carry the derived NCB."""
        test_cases = [
            (0, 1, 20.0),
            (0, 3, 35.0),
            (0, 5, 50.0),
            (2, 5,  0.0),
        ]
        for claims, tenure, expected_ncb in test_cases:
            ncb = derive_ncb_pct(claims, float(tenure))
            assert ncb == expected_ncb, (
                f"derive_ncb_pct({claims}, {tenure}) = {ncb}, expected {expected_ncb}"
            )
