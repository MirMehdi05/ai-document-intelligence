from app.services.document_service import extract_text_from_pdf


pdf_path = "data/raw/sample.pdf"

text = extract_text_from_pdf(pdf_path)

print("=" * 60)
print("EXTRACTED TEXT")
print("=" * 60)
print(text)