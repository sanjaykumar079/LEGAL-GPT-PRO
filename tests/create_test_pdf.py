# tests/create_test_pdf.py
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Test Legal Document - Page 1", ln=True, align='C')
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Test Legal Document - Page 2", ln=True, align='C')
pdf.output("tests/fixtures/test_legal_doc.pdf")