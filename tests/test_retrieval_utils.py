import unittest

from app.services.retrieval_utils import (
    classify_query_intent,
    expand_retrieval_query,
    expand_with_adjacent_employment_chunks,
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

    def test_adjacent_experience_chunk_restores_split_role_and_date(self):
        ranked = [
            {
                "document_id": "resume-1", "chunk_index": 10, "page_number": 1,
                "section_type": "employment", "text": "Responsibilities: incident response and escalation.",
                "hybrid_score": 0.9, "similarity": 0.8,
            },
            {
                "document_id": "resume-1", "chunk_index": 9, "page_number": 1,
                "section_type": "employment", "text": "Infrastructure Engineer Apprentice, People Inc (2026 – Present)",
                "hybrid_score": 0.7, "similarity": 0.6,
            },
            {
                "document_id": "resume-1", "chunk_index": 11, "page_number": 1,
                "section_type": "employment", "text": "Additional incident management responsibilities.",
                "hybrid_score": 0.5, "similarity": 0.4,
            },
            {
                "document_id": "resume-1", "chunk_index": 12, "page_number": 1,
                "section_type": "training", "text": "Datamites (2025 – Present)",
                "hybrid_score": 0.3, "similarity": 0.2,
            },
        ]

        sources = expand_with_adjacent_employment_chunks(ranked, top_k=1)

        self.assertEqual({source["chunk_index"] for source in sources[:3]}, {9, 10, 11})
        self.assertTrue(has_current_employment_evidence(sources))

    def test_adjacent_context_does_not_cross_section_boundary(self):
        ranked = [
            {
                "document_id": "resume-1", "chunk_index": 10, "page_number": 1,
                "section_type": "employment", "text": "Experience responsibilities.",
                "hybrid_score": 0.9, "similarity": 0.8,
            },
            {
                "document_id": "resume-1", "chunk_index": 9, "page_number": 1,
                "section_type": "training", "text": "Datamites (2025 – Present)",
                "hybrid_score": 0.7, "similarity": 0.6,
            },
        ]

        sources = expand_with_adjacent_employment_chunks(ranked, top_k=1)

        self.assertNotIn(9, {source["chunk_index"] for source in sources})
        self.assertFalse(has_current_employment_evidence(sources))


if __name__ == "__main__":
    unittest.main()
