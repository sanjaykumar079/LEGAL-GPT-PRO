import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service with Sentence Transformers model
        
        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Sentence Transformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            raise Exception(f"Failed to load model {self.model_name}: {str(e)}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            numpy array representing the embedding
        """
        if not self.model:
            raise Exception("Model not loaded")
        
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of numpy arrays representing embeddings
        """
        if not self.model:
            raise Exception("Model not loaded")
        
        if not texts or not all(isinstance(text, str) for text in texts):
            raise ValueError("All items must be non-empty strings")
        
        # Generate embeddings in batch for efficiency
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return [emb for emb in embeddings]
    
    def chunk_to_embedding(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a chunk dictionary to include embedding
        
        Args:
            chunk: Chunk dictionary with 'text' and other metadata
            
        Returns:
            Chunk dictionary with added 'embedding' field
        """
        if 'text' not in chunk:
            raise ValueError("Chunk must contain 'text' field")
        
        # Generate embedding for the chunk text
        embedding = self.generate_embedding(chunk['text'])
        
        # Create a copy and add embedding
        chunk_with_embedding = chunk.copy()
        chunk_with_embedding['embedding'] = embedding.tolist()  # Convert to list for JSON serialization
        chunk_with_embedding['embedding_dim'] = len(embedding)
        
        return chunk_with_embedding
    
    def chunks_to_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert multiple chunks to include embeddings
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunk dictionaries with embeddings
        """
        chunks_with_embeddings = []
        
        for chunk in chunks:
            chunk_with_emb = self.chunk_to_embedding(chunk)
            chunks_with_embeddings.append(chunk_with_emb)
        
        return chunks_with_embeddings
    # Add this function at the end of your embedding_service.py file


    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Generate embeddings
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(emb1, emb2)
        return float(similarity)
    
    def find_most_similar(self, query: str, texts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Find most similar texts to a query
        
        Args:
            query: Query text
            texts: List of texts to search through
            top_k: Number of most similar results to return
            
        Returns:
            List of dictionaries with text and similarity score
        """
        if not texts:
            return []
        
        # Generate embeddings
        query_emb = self.generate_embedding(query)
        text_embs = self.generate_embeddings(texts)
        
        # Calculate similarities
        similarities = []
        for i, text_emb in enumerate(text_embs):
            similarity = self._cosine_similarity(query_emb, text_emb)
            similarities.append({
                'text': texts[i],
                'similarity': float(similarity),
                'index': i
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "max_sequence_length": self.model.max_seq_length,
            "device": str(self.model.device)
        }

# Convenience functions for quick testing
def generate_single_embedding(text: str) -> np.ndarray:
    """Quick function to generate a single embedding"""
    service = EmbeddingService()
    return service.generate_embedding(text)

def test_similarity(text1: str, text2: str) -> float:
    """Quick function to test similarity between two texts"""
    service = EmbeddingService()
    return service.semantic_similarity(text1, text2)
# Replace the get_embedding_service function with this corrected version:

def get_embedding_service():
    """Get the best available embedding service"""
    try:
        # Try to use Sentence Transformers first
        return EmbeddingService()
    except Exception as e:
        print(f"Warning: Sentence Transformers not available: {e}")
        print("Using fallback embedding service")
        return FallbackEmbeddingService()