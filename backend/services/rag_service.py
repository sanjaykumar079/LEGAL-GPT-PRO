"""
RAGService: Complete question-answering pipeline with hallucination control.

Pipeline:
    Question → Embed → Retrieve → Gate Check → [Refuse | Continue]
                                           → LLM → Refusal Check → [Refuse | Answer]
"""

from typing import List, Dict, Any, Optional
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from services.llm_service import LLMService
from services.hallucination_guard import HallucinationGuard


class RAGService:

    INSUFFICIENT_MESSAGE = (
        "I could not find sufficient information in the available documents "
        "to answer this question. The retrieved context does not contain "
        "relevant material on this topic."
    )

    def __init__(self,
                 chroma_persist_directory: str = "./chroma_db",
                 llm_model: str = "llama3:latest",
                 min_top1_similarity: float = 0.30,
                 min_mean_similarity: float = 0.25):
        print("Initializing RAG Service...")
        self.embedding_service = EmbeddingService()
        self.chroma_service = ChromaDBService(persist_directory=chroma_persist_directory)
        self.llm_service = LLMService(model_name=llm_model)
        self.guard = HallucinationGuard(
            min_top1_similarity=min_top1_similarity,
            min_mean_similarity=min_mean_similarity,
        )
        print(f"✅ RAG ready (gate: top1>{min_top1_similarity}, mean>{min_mean_similarity})")

    def answer_question(self,
                        question: str,
                        top_k: int = 5,
                        filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Full RAG pipeline with three-layer hallucination control.

        Returns a dict with:
            status: 'ANSWERED' | 'REFUSED_RETRIEVAL' | 'REFUSED_LLM'
            answer: the response text
            top1_similarity, mean_similarity: retrieval quality metrics
            retrieved_chunks: number of chunks found
            sources: list of retrieved chunk dicts
            citations: formatted source list (only if ANSWERED)
            reason: explanation for refusal (if applicable)
            error: False (True only for system errors)
        """
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")

        # ── 1. Embed the question ──────────────────────────────────────
        try:
            q_emb = self.embedding_service.generate_embedding(question)
        except Exception as e:
            return self._error_response(f"Embedding failed: {e}")

        # ── 2. Retrieve top-K chunks ───────────────────────────────────
        try:
            chunks = self.chroma_service.semantic_search(
                query=question,
                query_embedding=q_emb.tolist(),
                n_results=top_k,
                filter_metadata=filter_metadata,
            )
        except Exception as e:
            return self._error_response(f"Retrieval failed: {e}")

        # ── 3. LAYER 1+2: Retrieval quality gate ──────────────────────
        retrieval_eval = self.guard.evaluate_retrieval(chunks)
        print(f"   Gate: top1={retrieval_eval['top1_similarity']:.4f}, "
              f"mean={retrieval_eval['mean_similarity']:.4f} "
              f"({'PASS' if retrieval_eval['should_answer'] else 'FAIL'})")

        if not retrieval_eval['should_answer']:
            print(f"   → REFUSED at retrieval gate: {retrieval_eval['reason']}")
            return {
                'status': 'REFUSED_RETRIEVAL',
                'answer': self.INSUFFICIENT_MESSAGE,
                'reason': retrieval_eval['reason'],
                'top1_similarity': retrieval_eval['top1_similarity'],
                'mean_similarity': retrieval_eval['mean_similarity'],
                'retrieved_chunks': retrieval_eval['n_chunks'],
                'sources': [],
                'citations': [],
                'error': False,
            }

        # ── 4. LLM generation with strict prompt ──────────────────────
        try:
            raw_answer = self.llm_service.answer_question_with_context(question, chunks)
        except Exception as e:
            return self._error_response(f"LLM generation failed: {e}")

        # ── 5. LAYER 3: Post-LLM refusal detection ────────────────────
        refusal_check = self.guard.detect_refusal(raw_answer)

        if refusal_check['is_refusal']:
            print(f"   → REFUSED by LLM: matched {refusal_check['matched_phrases']}")
            return {
                'status': 'REFUSED_LLM',
                'answer': self.INSUFFICIENT_MESSAGE,
                'reason': f"LLM self-reported insufficiency: {refusal_check['matched_phrases']}",
                'top1_similarity': retrieval_eval['top1_similarity'],
                'mean_similarity': retrieval_eval['mean_similarity'],
                'retrieved_chunks': len(chunks),
                'sources': chunks,
                'citations': [],
                'error': False,
            }

        # ── 6. SUCCESS: Build citations from metadata ─────────────────
        citations = self._build_citations(chunks)
        print(f"   → ANSWERED (top1={retrieval_eval['top1_similarity']:.4f}, "
              f"{len(citations)} citations)")

        return {
            'status': 'ANSWERED',
            'answer': raw_answer.strip(),
            'top1_similarity': retrieval_eval['top1_similarity'],
            'mean_similarity': retrieval_eval['mean_similarity'],
            'retrieved_chunks': len(chunks),
            'sources': chunks,
            'citations': citations,
            'error': False,
        }

    def _build_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Extract unique (document, page) pairs from chunk metadata."""
        seen = set()
        citations = []
        for c in chunks:
            doc = c['metadata'].get('document', 'Unknown')
            page = c['metadata'].get('page', 'N/A')
            key = f"{doc}::p{page}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    'document': doc,
                    'page': page,
                    'similarity': c.get('cosine_similarity', 0),
                    'preview': c['text'][:100] + '...' if len(c['text']) > 100 else c['text'],
                })
        return citations

    def _error_response(self, msg: str) -> Dict:
        print(f"   → ERROR: {msg}")
        return {'status': 'ERROR', 'answer': msg, 'error': True, 'sources': []}

    # ── Utility methods ─────────────────────────────────────────────────
    def add_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        return self.chroma_service.add_chunks(chunks)

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        return self.chroma_service.get_collection_stats()