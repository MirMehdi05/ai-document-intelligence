import unittest

from app.services.chunking_service import chunk_pages_with_sections, detect_section_heading


class SectionAwareChunkingTests(unittest.TestCase):
    def test_recognizes_training_heading(self):
        section = detect_section_heading("CERTIFICATIONS & TRAINING")
        self.assertEqual(section["section_type"], "training")

    def test_training_section_carries_across_pages(self):
        pages = [
            {"page_number": 1, "text": "CERTIFICATIONS & TRAINING\nAI Tools Masterclass"},
            {"page_number": 2, "text": "Datamites (2025 – Present)\nbe10x workshop"},
        ]
        chunks = chunk_pages_with_sections(pages, chunk_size=500, overlap=50)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk["section_type"] == "training" for chunk in chunks))
        self.assertIn("Datamites", chunks[1]["text"])
        self.assertIn("[Document section: Training & Certifications]", chunks[1]["text"])

    def test_experience_heading_switches_section(self):
        pages = [{
            "page_number": 1,
            "text": (
                "CERTIFICATIONS\nDatamites (2025 – Present)\n"
                "PROFESSIONAL EXPERIENCE\nInfrastructure Engineer Apprentice, People Inc (2026 – Present)"
            ),
        }]
        chunks = chunk_pages_with_sections(pages)

        self.assertEqual(chunks[0]["section_type"], "training")
        self.assertEqual(chunks[1]["section_type"], "employment")
        self.assertIn("People Inc", chunks[1]["text"])

    def test_general_documents_keep_page_metadata(self):
        chunks = chunk_pages_with_sections([
            {"page_number": 3, "text": "A paragraph about a technical system."}
        ])
        self.assertEqual(chunks[0]["page_number"], 3)
        self.assertEqual(chunks[0]["section_type"], "other")


if __name__ == "__main__":
    unittest.main()
