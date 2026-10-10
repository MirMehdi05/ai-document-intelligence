from reportlab.pdfgen import canvas

pdf_path = "data/raw/sample.pdf"

c = canvas.Canvas(pdf_path)

c.drawString(100, 750, "AI Document Intelligence Test")
c.drawString(100, 720, "This is a clean test PDF.")
c.drawString(100, 690, "Python will extract this text using pypdf.")
c.drawString(100, 660, "Our document intelligence project is working.")

c.save()

print(f"Created PDF: {pdf_path}")