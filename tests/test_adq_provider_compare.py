import unittest

from tests.adq_provider_compare import _canonical_id_sha256, _classify, _vram_status


class AdqProviderComparisonTests(unittest.TestCase):
    def test_canonical_token_identity_matches_campaign_contract(self) -> None:
        self.assertEqual(
            _canonical_id_sha256([1, 2, 3]),
            "8a6ae15122001229edb8866f56e342af12ae8187203c3e3b33931743e7c0c48d",
        )

    def test_all_required_classifications(self) -> None:
        self.assertEqual(_classify("PASS", "PASS"), "ALIGNED_PASS")
        self.assertEqual(_classify("FAIL", "PASS"), "KRASIS_DEFICIT")
        self.assertEqual(_classify("PASS", "FAIL"), "KRASIS_ONLY_PASS")
        self.assertEqual(_classify("FAIL", "FAIL"), "ALIGNED_FAIL")
        self.assertEqual(_classify("INCOMPLETE", "PASS"), "INCOMPLETE")
        self.assertEqual(
            _classify("PASS", "PASS", providers_disagree=True),
            "REFERENCE_DISAGREEMENT",
        )

    def test_vram_is_independent_and_fail_closed(self) -> None:
        self.assertEqual(
            _vram_status({"safety_margin_mb": 600, "vram_low_water": [{"min_free_mb": 601}]}),
            ("PASS", 601, 600),
        )
        self.assertEqual(
            _vram_status({"safety_margin_mb": 600, "vram_low_water": [{"min_free_mb": 599}]}),
            ("FAIL", 599, 600),
        )
        self.assertEqual(_vram_status({}), ("INCOMPLETE", None, None))


if __name__ == "__main__":
    unittest.main()
