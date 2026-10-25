"""
Day 8: Hallucination Control Evaluation
10 answerable + 10 unanswerable questions.
Measures: True Positives, False Positives (hallucinations),
          False Negatives (missed answers), True Negatives.
"""

import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.rag_service import RAGService
from services.pdf_service import PDFService
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService


# ── Question Sets ─────────────────────────────────────────────────────
# Each entry: (question, expected_category)
# 'answerable'   = the answer IS in the document
# 'unanswerable' = the answer is NOT in the document

ANSWERABLE = [
    "What is the case number?",
    "Who are the plaintiff and defendant?",
    "What was shipped on March 15, 2023?",
    "How much damage did the plaintiff allege?",
    "What did the dock supervisor testify about the gasket?",
    "What was the expert witness's analysis of the gasket polymer?",
    "What did the quality control logs show about the gasket replacement?",
    "Which statute governs carrier liability in this case?",
    "What was the ruling on the motion for summary judgment?",
    "What happened to the punitive damages claim?",
]

UNANSWERABLE = [
    "What did the court rule about arbitration clauses in insurance contracts?",
    "What is the statute of limitations for maritime claims in this district?",
    "What was the jury's verdict in this case?",
    "What did the court order regarding attorney's fees?",
    "What is the maximum civil penalty for violating 49 C.F.R. § 174.10?",
    "What did the court say about the plaintiff's bankruptcy filing?",
    "What did the appellate court decide on the defendant's appeal?",
    "What was the total insurance premium paid by the defendant annually?",
    "What did the court order regarding injunctive relief on the warehouse?",
    "What were the specific terms of the shipping contract between the parties?",
]


def ensure_knowledge_base(rag: RAGService, pdf_path: str = "tests/fixtures/court_order_v2.pdf"):
    """Ingest the v2 PDF if the DB is empty."""
    stats = rag.get_knowledge_base_stats()
    if stats['total_chunks'] > 0:
        print(f"   (Using existing DB: {stats['total_chunks']} chunks)")
        return

    print(f"   Ingesting: {pdf_path}")
    pdf_service = PDFService()
    pages = pdf_service.extract_text_from_pdf(pdf_path)

    chunker = ChunkingService(chunk_size=600, overlap=100)
    cleaned = [{'page': p.page_number, 'text': p.text} for p in pages]
    chunks = chunker.chunk_document_pages(cleaned, os.path.basename(pdf_path))

    emb = EmbeddingService()
    chunks_with_emb = [emb.chunk_to_embedding(c) for c in chunks]
    rag.add_document_chunks(chunks_with_emb)
    print(f"   Indexed {len(chunks_with_emb)} chunks")


def run_evaluation(rag: RAGService):
    results = []
    all_questions = (
        [(q, 'answerable') for q in ANSWERABLE] +
        [(q, 'unanswerable') for q in UNANSWERABLE]
    )

    print(f"\n{'='*70}")
    print(f"RUNNING {len(all_questions)} QUESTIONS")
    print(f"{'='*70}")

    for idx, (question, category) in enumerate(all_questions, 1):
        result = rag.answer_question(question, top_k=5)

        status = result['status']  # ANSWERED | REFUSED_RETRIEVAL | REFUSED_LLM

        # Determine correctness
        if category == 'answerable':
            correct = (status == 'ANSWERED')
            outcome = 'TP' if correct else 'FN'
        else:
            correct = (status != 'ANSWERED')
            outcome = 'TN' if correct else 'FP'  # FP = HALLUCINATION

        results.append({
            'num': idx,
            'question': question,
            'category': category,
            'status': status,
            'correct': correct,
            'outcome': outcome,
            'top1': result.get('top1_similarity', 0),
            'mean': result.get('mean_similarity', 0),
        })

        icon = '✅' if correct else '❌'
        print(f"\n[{idx:02d}] {icon} {outcome} | {category:12s} | top1={result.get('top1_similarity',0):.3f}")
        print(f"     Q: {question}")
        if status == 'ANSWERED':
            print(f"     A: {result['answer'][:120]}...")
        else:
            print(f"     R: {result['answer'][:100]}...")
            print(f"     Reason: {result.get('reason', 'N/A')}")

    # ── Summary ─────────────────────────────────────────────────────────
    tp = sum(1 for r in results if r['outcome'] == 'TP')
    fn = sum(1 for r in results if r['outcome'] == 'FN')
    tn = sum(1 for r in results if r['outcome'] == 'TN')
    fp = sum(1 for r in results if r['outcome'] == 'FP')
    total = len(results)
    accuracy = (tp + tn) / total * 100
    # FP is the critical metric: 0 hallucinations = 0 FP
    hallucination_rate = fp / max(1, fp + tn) * 100

    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total questions:       {total}")
    print(f"  Answerable (10):       TP={tp}  FN={fn}")
    print(f"  Unanswerable (10):     TN={tn}  FP={fp}  ← FP = hallucinations")
    print(f"  Overall accuracy:      {accuracy:.0f}%")
    print(f"  Hallucination rate:    {hallucination_rate:.0f}%  (FP / unanswerable)")
    print(f"{'='*70}")

    # Save results
    with open("tests/day8_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to tests/day8_results.json")

    return results


def main():
    print("=== DAY 8: Hallucination Control Evaluation ===\n")

    rag = RAGService(
        chroma_persist_directory="./chroma_db",
        llm_model="llama3:latest",
        min_top1_similarity=0.30,
        min_mean_similarity=0.25,
    )

    ensure_knowledge_base(rag)
    run_evaluation(rag)


if __name__ == "__main__":
    main()