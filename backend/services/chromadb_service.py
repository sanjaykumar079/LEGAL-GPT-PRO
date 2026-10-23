import chromadb
from chromadb.config import Settings
import uuid, os
from typing import List, Dict, Any, Optional
import numpy as np
from services.embedding_service import EmbeddingService

class ChromaDBService:
    COLLECTION_NAME = "legal_documents"

    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        # Force cosine space so HNSW ranking matches our cosine score
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "description": "Legal chunks (cosine)"
            }
        )
        print(f"✅ ChromaDB ready ({self.collection.count()} items)")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        if not chunks:
            return []
        ids, docs, embs, metas = [], [], [], []
        for c in chunks:
            ids.append(c.get('chunk_id', str(uuid.uuid4())))
            docs.append(c['text'])
            emb = c['embedding']
            embs.append(emb.tolist() if isinstance(emb, np.ndarray) else emb)
            metas.append({
                'document': c.get('document', 'unknown'),
                'page': int(c.get('page', 0)),
                'chunk_index': int(c.get('chunk_index', 0)),
                'char_count': int(c.get('char_count', 0)),
            })
        self.collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        print(f"✅ Added {len(chunks)} chunks")
        return ids

    def semantic_search(self, query: str, query_embedding: List[float],
                        n_results: int = 5,
                        filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)

        res = self.collection.query(
            query_embeddings=[qn.tolist()],
            n_results=n_results,
            where=filter_metadata,
            include=['documents', 'metadatas', 'distances', 'embeddings']
        )

        out = []
        for i in range(len(res['ids'][0])):
            stored_emb = res['embeddings'][0][i] if res['embeddings'][0][i] else qn
            cos = EmbeddingService.cosine_similarity(qn, stored_emb)
            out.append({
                'id': res['ids'][0][i],
                'text': res['documents'][0][i],
                'metadata': res['metadatas'][0][i],
                'cosine_similarity': round(cos, 4),   # REAL cosine, in [-1, 1]
                'distance': res['distances'][0][i],
            })
        return out

    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        docs, pages = set(), set()
        if count:
            sample = self.collection.get(limit=min(50, count),