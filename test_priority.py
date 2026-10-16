import unittest

from priority import classify_priority


class ClassifyPriorityTests(unittest.TestCase):
    def test_urgent_outage_is_high(self):
        self.assertEqual(
            classify_priority("Urgent: production outage affecting all users")[0],
            "High",
        )

    def test_system_is_down_is_high(self):
        self.assertEqual(classify_priority("The system is down for all users")[0], "High")

    def test_negated_critical_signal_is_not_high(self):
        self.assertEqual(classify_priority("This is not a critical outage")[0], "Low")

    def test_common_not_working_phrase_is_medium(self):
        self.assertEqual(classify_priority("The application is not working")[0], "Medium")

    def test_cant_access_phrase_is_medium(self):
        self.assertEqual(classify_priority("I can't access my account")[0], "Medium")

    def test_negated_contraction_does_not_trigger_high(self):
        self.assertEqual(classify_priority("It isn't critical anymore")[0], "Low")

    def test_keyword_inside_another_word_does_not_match(self):
        self.assertEqual(classify_priority("The service is a bit subcritical")[0], "Low")

    def test_no_signal_is_low_with_explanation(self):
        priority, reason = classify_priority("Please tell me your support hours")
        self.assertEqual(priority, "Low")
        self.assertIn("No high- or medium-priority signal", reason)


if __name__ == "__main__":
    unittest.main()
