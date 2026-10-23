import numpy as np
from typing import List, Dict, Any, Union

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise Exception("Install: pip install sentence-transformers")
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_embedding_dimension()  # renamed method
        print(f"✅ Model loaded. Dimension: {dim}")

    # ---- CORE: normalized embeddings + true cosine similarity ----
    @staticmethod
    def _normalize(vec):
        vec = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    @staticmethod
    def cosine_similarity(v1, v2) -> float:
        """True cosine similarity in [-1, 1]. NOT a probability."""
        v1 = np.asarray(v1, dtype=np.float32)
        v2 = np.asarray(v2, dtype=np.float32)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def generate_embedding(self, text: str) -> np.ndarray:
        """Return a NORMALIZED (unit) embedding."""
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        raw = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return self._normalize(raw)

    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        raw = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [self._normalize(e) for e in raw]

    def chunk_to_embedding(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        if 'text' not in chunk:
            raise ValueError("Chunk must contain 'text'")
        embedding = self.generate_embedding(chunk['text'])
        out = chunk.copy()
        out['embedding'] = embedding.tolist()
        out['embedding_dim'] = len(embedding)
        return out

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_embedding_dimension(),
            "device": str(self.model.device),
        }

class FallbackEmbeddingService:
    """Degraded service if sentence-transformers is missing."""
    def __init__(self):
        print("⚠️  Using fallback embedding (weak, for dev only)")
    def generate_embedding(self, text):
        words = text.lower().split()
        emb = np.zeros(128, dtype=np.float32)
        for w in words:
            emb[hash(w) % 128] += 1
        return self._normalize(emb)
    @staticmethod
    def _normalize(vec):
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    def cosine_similarity(self, a, b):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (n1 * n2)) if n1 and n2 else 0.0
    def chunk_to_embedding(self, chunk):
        out = chunk.copy()
        out['embedding'] = self.generate_embedding(chunk['text']).tolist()
        out['embedding_dim'] = 128
        return out
    def get_model_info(self):
        return {"model_name": "fallback", "embedding_dimension": 128}

def get_embedding_service():
    return EmbeddingService() if SENTENCE_TRANSFORMERS_AVAILABLE else FallbackEmbeddingService()