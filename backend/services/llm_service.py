import requests
from typing import List, Dict, Any


class LLMService:
    def __init__(self, model_name: str = "llama3:latest",
                 base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m['name'] for m in r.json().get('models', [])]
            print(f"✅ Ollama connected. Models: {models}")
        except Exception as e:
            raise ConnectionError(f"Ollama not reachable at {self.base_url}: {e}")

    def answer_question_with_context(self, question: str,
                                     context_chunks: List[Dict[str, Any]]) -> str:
        if not context_chunks:
            return "I could not find sufficient information in the available documents."

        context_parts = []
        for i, c in enumerate(context_chunks, 1):
            doc = c['metadata'].get('document', 'Unknown')
            page = c['metadata'].get('page', 'N/A')
            context_parts.append(
                f"[Source {i}] {doc}, Page {page}\n{c['text']}"
            )
        context_block = "\n\n".join(context_parts)

        prompt = (
            "You are a legal research assistant. You answer questions using "
            "ONLY the provided legal context below.\n\n"
            "RULES (strict):\n"
            "1. Use ONLY facts present in the context. Do not add external legal knowledge.\n"
            "2. Do not invent case names, statute numbers, dates, or facts.\n"
            "3. If the context does not contain enough information to answer the "
            "question, respond EXACTLY with:\n"
            "   'The provided context does not contain sufficient information to "
            "answer this question.'\n"
            "4. Do not speculate, extrapolate, or fill gaps.\n\n"
            f"--- LEGAL CONTEXT ---\n{context_block}\n--- END CONTEXT ---\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 800},
                },
                timeout=120,
            )
            if r.status_code == 200:
                return r.json().get('response', '').strip()
            raise Exception(f"Ollama error {r.status_code}: {r.text}")
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot reach Ollama. Is it running?")