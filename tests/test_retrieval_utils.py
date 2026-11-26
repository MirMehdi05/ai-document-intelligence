import unittest

from app.services.retrieval_utils import (
    classify_query_intent,
    expand_retrieval_query,
    has_current_employment_evidence,
    section_relevance_bonus,
)


class RetrievalIntentTests(unittest.TestCase):
    def test_current_role_questions_are_classified(self):
        self.assertEqual(classify_query_intent("What is Mir's current role?"), "current_role")
        self.assertEqual(classify_query_intent("what does mir currently do?"), "current_role")
        self.assertEqual(classify_query_intent("Where does Mir currently work?"), "current_role")

    def test_project_questions_are_not_misclassified_as_employment(self):
        self.assertEqual(classify_query_intent("What projects has Mir worked on?"), "projects")

    def test_training_is_penalized_for_current_role_search(self):
        self.assertLess(section_relevance_bonus("current_role", "training"), 0)
        self.assertGreater(section_relevance_bonus("current_role", "employment"), 0)

    def test_training_date_range_is_not_current_employment_evidence(self):
        sources = [{
            "section_type": "training",
            "text": "Datamites (2025 – Present)",
        }]
        self.assertFalse(has_current_employment_evidence(sources))

    def test_current_employment_needs_experience_section_and_status_marker(self):
        sources = [{
            "section_type": "employment",
            "text": "Infrastructure Engineer Apprentice, People Inc (2026 – Present)",
        }]
        self.assertTrue(has_current_employment_evidence(sources))
        self.assertFalse(has_current_employment_evidence([{
            "section_type": "employment",
            "text": "Infrastructure Engineer Apprentice, People Inc (2024 – 2025)",
        }]))

    def test_query_expansion_is_intent_specific(self):
        expanded = expand_retrieval_query("what does mir currently do?")
        self.assertIn("professional experience", expanded)
        self.assertIn("employer", expanded)


if __name__ == "__main__":
    unittest.main()
