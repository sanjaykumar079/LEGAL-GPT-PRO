from fpdf import FPDF
import os

# Ensure directory exists
os.makedirs("tests/fixtures", exist_ok=True)

# Create test legal document PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="UNITED STATES DISTRICT COURT", ln=True, align='C')
pdf.cell(200, 10, txt="DISTRICT OF COLUMBIA", ln=True, align='C')
pdf.ln(10)
pdf.cell(200, 10, txt="Case No. 123:456", ln=True)
pdf.multi_cell(0, 10, txt="This is a test legal document created for PDF extraction testing purposes. It contains multiple pages to test page-wise text extraction and proper page numbering for legal citations.")
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="SECTION II - FINDINGS", ln=True)
pdf.multi_cell(0, 10, txt="The court finds that the defendant acted reasonably under the circumstances. This conclusion is supported by evidence presented during the hearing on March 15, 2024.")
pdf.output("tests/fixtures/test_legal_doc.pdf")

# Create test txt file
with open("tests/fixtures/test.txt", "w") as f:
    f.write("This is a test text file")

print("Test files created successfully!")