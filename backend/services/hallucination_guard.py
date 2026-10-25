"""
HallucinationGuard: Three-layer defense against fabricated legal answers.

IMPORTANT: Cosine similarity is a GEOMETRIC MEASURE, not a calibrated
probability. The thresholds below are TUNABLE HEURISTICS for a retrieval
quality gate. Do NOT interpret 0.40 as "40% confidence." Calibrate
per-domain using your own labeled test set.
"""

from typing import List, Dict, Any, Optional


class HallucinationGuard:

    # Default refusal phrases (case-insensitive substring match)
    DEFAULT_REFUSAL_PHRASES = [
        "insufficient information",
        "cannot find",
        "could not find",
        "not found in",
        "no information",
        "does not contain",
        "not present in the",
        "i cannot",
        "i'm unable",
        "i am unable",
        "not available in the",
        "no relevant",
        "i do not have",
        "not discussed in",
        "the context does not",
        "i could not locate",
    ]

    def __init__(self,
                 min_top1_similarity: float = 0.30,
                 min_mean_similarity: float = 0.25,
                 refusal_phrases: Optional[List[str]] = None):
        """
        Args:
            min_top1_similarity: If the BEST matching chunk scores below this,
                the retrieval is considered too weak to ground an answer.
                TUNABLE. Start at 0.30 for legal text, adjust per domain.
            min_mean_similarity: If the AVERAGE similarity of retrieved chunks
                is below this, the overall context quality is poor.
                TUNABLE. Start at 0.25.
            refusal_phrases: Phrases that indicate the LLM refused to answer.
        """
        self.min_top1 = min_top1_similarity
        self.min_mean = min_mean_similarity
        self.refusal_phrases = [p.lower() for p in (refusal_phrases or self.DEFAULT_REFUSAL_PHRASES)]

    # ── LAYER 1+2: Pre-LLM retrieval quality gate ──────────────────────
    def evaluate_retrieval(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decide whether retrieved chunks are sufficient to ground an answer.

        Returns a dict with:
            should_answer (bool)
            reason (str)         — human-readable explanation
            top1_similarity (float)
            mean_similarity (float)
            n_chunks (int)
        """
        if not chunks:
            return {
                'should_answer': False,
                'reason': 'no_chunks_retrieved',
                'top1_similarity': 0.0,
                'mean_similarity': 0.0,
                'n_chunks': 0,
            }

        sims = [c.get('cosine_similarity', 0.0) for c in chunks]
        top1 = max(sims)
        mean_sim = sum(sims) / len(sims)

        reasons = []
        should_answer = True

        if top1 < self.min_top1:
            should_answer = False
            reasons.append(
                f"top1={top1:.4f} < gate({self.min_top1})"
            )
        if mean_sim < self.min_mean:
            should_answer = False
            reasons.append(
                f"mean={mean_sim:.4f} < gate({self.min_mean})"
            )

        return {
            'should_answer': should_answer,
            'reason': '; '.join(reasons) if reasons else 'retrieval_ok',
            'top1_similarity': round(top1, 4),
            'mean_similarity': round(mean_sim, 4),
            'n_chunks': len(chunks),
        }

    # ── LAYER 3: Post-LLM refusal detection ────────────────────────────
    def detect_refusal(self, llm_output: str) -> Dict[str, Any]:
        """
        Check whether the LLM itself indicated it could not answer.

        This catches cases where retrieval passed the gate but the
        LLM correctly identified that the context doesn't actually
        answer the specific question.
        """
        text = llm_output.lower().strip()
        matched = [p for p in self.refusal_phrases if p in text]
        return {
            'is_refusal': len(matched) > 0,
            'matched_phrases': matched,
        }

    # ── Combined decision ──────────────────────────────────────────────
    def final_decision(self, retrieval_eval: Dict, refusal_check: Dict) -> str:
        """
        Returns one of: 'ANSWERED', 'REFUSED_RETRIEVAL', 'REFUSED_LLM'
        """
        if not retrieval_eval['should_answer']:
            return 'REFUSED_RETRIEVAL'
        if refusal_check['is_refusal']:
            return 'REFUSED_LLM'
        return 'ANSWERED'