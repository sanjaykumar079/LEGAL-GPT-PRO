"""Generate a 5-page mock court order for Day 8 testing."""
from fpdf import FPDF
import os

os.makedirs("tests/fixtures", exist_ok=True)

pdf = FPDF()
pdf.set_auto_page_break(True, margin=20)

# ── Page 1: Header + Parties ──────────────────────────────────────────
pdf.add_page()
pdf.set_font("Arial", "B", 14)
pdf.cell(0, 10, "UNITED STATES DISTRICT COURT", align='C', ln=True)
pdf.cell(0, 10, "DISTRICT OF COLUMBIA", align='C', ln=True)
pdf.ln(4)
pdf.set_font("Arial", size=11)
pdf.cell(0, 7, "Case No. 1:24-cv-01234 (JLS)", ln=True)
pdf.cell(0, 7, "Plaintiff: Meridian Insurance Group", ln=True)
pdf.cell(0, 7, "Defendant: Hartwell Logistics Corp.", ln=True)
pdf.ln(4)
pdf.set_font("Arial", "B", size=12)
pdf.cell(0, 8, "ORDER ON MOTION FOR SUMMARY JUDGMENT", ln=True)
pdf.ln(6)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 6, """
The Court has reviewed the Defendant's Motion for Summary Judgment
(Dkt. #47), the Plaintiff's opposition, and the supporting evidence.
Oral argument was heard on March 20, 2024. For the reasons stated
below, the motion is GRANTED in part and DENIED in part.
""")

# ── Page 2: Findings of Fact (Part 1) ─────────────────────────────────
pdf.add_page()
pdf.set_font("Arial", "B", size=12)
pdf.cell(0, 8, "I. FINDINGS OF FACT", ln=True)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 6, """
On March 15, 2023, the Defendant shipped a container of industrial
solvents classified as hazardous materials, Class 3, from the Port
of Baltimore to a warehouse in Alexandria, Virginia.

The Plaintiff, the carrier's insurer, alleges that the solvents
leaked during transit, causing approximately $2.4 million in damage
to adjacent storage facilities at the Alexandria warehouse.

Witness testimony from dock supervisor R. Alvarez established that
the container's external seal was intact upon arrival, but the
internal gasket had degraded. The degradation was attributed to
sub-freezing temperatures experienced during a prior transit in
December 2022.
""")

# ── Page 3: Findings of Fact (Part 2) ─────────────────────────────────
pdf.add_page()
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 6, """
Expert witness Dr. Li Wei Chen, a materials engineer retained by
the Defendant, testified that the gasket polymer (EPDM compound
grade 70) was operated outside its recommended service temperature
range during the December 2022 transit, which accelerated
degradation beyond normal aging.

The Plaintiff's expert, Prof. Adaeze Okafor, reviewed Dr. Chen's
materials analysis and agreed with the degradation mechanism but
disputed the conclusion that the failure was solely due to the
December transit conditions.

The Defendant's internal quality control logs, produced in
discovery, show that the container was flagged for gasket
replacement in November 2022. The replacement was documented
as "pending" in the maintenance log and was not completed
before the December 2022 transit or the March 2023 shipment.
""")

# ── Page 4: Conclusions of Law ────────────────────────────────────────
pdf.add_page()
pdf.set_font("Arial", "B", size=12)
pdf.cell(0, 8, "II. CONCLUSIONS OF LAW", ln=True)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 6, """
Carrier liability for interstate transportation of hazardous
materials is governed by the Hazardous Materials Transportation
Act (HMTA), 49 U.S.C. § 5101 et seq.

Under 49 U.S.C. § 5123, a carrier is liable for damage caused
by the transportation of hazardous materials unless the damage
was caused by an act of God, the act of a public enemy, the act
of the shipper, or the inherent nature of the goods.

The Plaintiff bears the burden of establishing that the
Defendant's breach of the reasonable care standard under
49 C.F.R. § 174.10 was a proximate cause of the damage.

The Court finds that the Defendant's failure to complete the
documented gasket replacement before the December 2022 transit
constituted a breach of the reasonable care standard. This breach
was a proximate cause of the gasket failure that led to the
March 2023 leak.
""")

# ── Page 5: Order ─────────────────────────────────────────────────────
pdf.add_page()
pdf.set_font("Arial", "B", size=12)
pdf.cell(0, 8, "III. ORDER", ln=True)
pdf.set_font("Arial", size=11)
pdf.multi_cell(0, 6, """
For the foregoing reasons, it is hereby ORDERED:

1. The Defendant's Motion for Summary Judgment (Dkt. #47) is
   GRANTED in part and DENIED in part.

2. Summary judgment is GRANTED as to the Plaintiff's claim for
   punitive damages. The punitive damages claim is dismissed
   with prejudice.

3. The claim for compensatory damages in the amount of
   $2,400,000 shall proceed to trial.

4. The Defendant shall file a proposed conforming order within
   fourteen (14) days of this Order.

5. The Plaintiff may file objections to the proposed order
   within seven (7) days of service.

SO ORDERED this 28th day of March, 2024.

                              /s/ Judge L. S. Whitfield
                              United States District Judge
""")

pdf.output("tests/fixtures/court_order_v2.pdf")
print("✅ Created tests/fixtures/court_order_v2.pdf (5 pages)")