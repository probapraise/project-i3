import unittest

from profile_lock_rules import merge_field, merge_field_sets


class ProfileLockRulesTests(unittest.TestCase):
    def test_real_field_is_not_overwritten_by_inferred(self) -> None:
        existing = {
            "field": "occupation",
            "value": "taxi driver",
            "source_tag": "REAL",
            "sources": ["src_001"],
        }
        incoming = {
            "field": "occupation",
            "value": "delivery driver",
            "source_tag": "INFERRED",
            "confidence": 0.9,
            "reason": "guess",
        }

        merged, decision = merge_field(existing, incoming)
        self.assertEqual(merged["value"], "taxi driver")
        self.assertEqual(merged["source_tag"], "REAL")
        self.assertEqual(decision.reason, "REAL_LOCKED")

    def test_no_info_is_upgraded_to_inferred(self) -> None:
        existing = {
            "field": "hair_color",
            "value": None,
            "source_tag": "NO_INFO",
        }
        incoming = {
            "field": "hair_color",
            "value": "dark brown",
            "source_tag": "INFERRED",
            "confidence": 0.4,
            "reason": "regional prior",
        }

        merged, decision = merge_field(existing, incoming)
        self.assertEqual(merged["source_tag"], "INFERRED")
        self.assertEqual(decision.reason, "TAG_UPGRADE")

    def test_inferred_prefers_higher_confidence(self) -> None:
        existing = {
            "field": "body_type",
            "value": "average build",
            "source_tag": "INFERRED",
            "confidence": 0.3,
            "reason": "low confidence",
        }
        incoming = {
            "field": "body_type",
            "value": "lean build",
            "source_tag": "INFERRED",
            "confidence": 0.6,
            "reason": "better estimate",
        }

        merged, decision = merge_field(existing, incoming)
        self.assertEqual(merged["value"], "lean build")
        self.assertEqual(decision.reason, "HIGHER_CONFIDENCE_INFERRED")

    def test_merge_field_sets_adds_new_field(self) -> None:
        existing_fields = [{"field": "gender", "value": None, "source_tag": "NO_INFO"}]
        incoming_fields = [
            {
                "field": "ethnicity",
                "value": "Latino/Brazilian appearance",
                "source_tag": "INFERRED",
                "confidence": 0.56,
                "reason": "location prior",
            }
        ]
        merged, decisions = merge_field_sets(existing_fields, incoming_fields)
        merged_names = {f["field"] for f in merged}
        self.assertIn("ethnicity", merged_names)
        self.assertTrue(any(d.reason == "FIELD_ADDED" for d in decisions))


if __name__ == "__main__":
    unittest.main()
