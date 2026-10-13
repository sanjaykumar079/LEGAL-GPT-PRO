import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional
import numpy as np

class ChromaDBService:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB service
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create client with persistence
            self.client = chromadb.Client(
                Settings(
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="legal_documents",
                metadata={"description": "Legal document chunks with embeddings"}
            )
            
            print(f"✅ ChromaDB initialized. Collection has {self.collection.count()} items.")
            
        except Exception as e:
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add chunks with embeddings to ChromaDB
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'embedding', and metadata
            
        Returns:
            List of IDs for the added chunks
        """
        if not chunks:
            return []
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID if not provided
            chunk_id = chunk.get('chunk_id', str(uuid.uuid4()))
            ids.append(chunk_id)
            
            # Extract data
            documents.append(chunk['text'])
            embeddings.append(chunk['embedding'])
            
            # Prepare metadata (ChromaDB requires flat dictionaries)
            metadata = {
                'document': chunk.get('document', 'unknown'),
                'page': chunk.get('page', 0),
                'chunk_index': chunk.get('chunk_index', 0),
                'char_count': chunk.get('char_count', 0),
                'embedding_dim': chunk.get('embedding_dim', 0)
            }
            metadatas.append(metadata)
        
        try:
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"✅ Added {len(chunks)} chunks to ChromaDB")
            return ids
            
        except Exception as e:
            raise Exception(f"Failed to add chunks to ChromaDB: {str(e)}")
    
    def semantic_search(self, query: str, query_embedding: List[float], 
                       n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using query embedding
        
        Args:
            query: The query text
            query_embedding: Embedding vector for the query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results with text, metadata, and similarity scores
        """
        try:
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            raise Exception(f"Semantic search failed: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Get sample of metadata to analyze
            sample = self.collection.get(
                limit=min(10, count) if count > 0 else 0,
                include=['metadatas']
            )
            
            documents = set()
            pages = set()
            
            if sample['metadatas']:
                for metadata in sample['metadatas']:
                    documents.add(metadata.get('document', 'unknown'))
                    pages.add(metadata.get('page', 0))
            
            return {
                'total_chunks': count,
                'unique_documents': len(documents),
                'sample_pages': sorted(list(pages))[:10],
                'collection_name': self.collection.name
            }
            
        except Exception as e:
            raise Exception(f"Failed to get collection stats: {str(e)}")
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name="legal_documents")
            print("✅ Collection deleted successfully")
        except Exception as e:
            raise Exception(f"Failed to delete collection: {str(e)}")
    
    def clear_collection(self):
        """Clear all items from the collection"""
        try:
            # Get all IDs
            all_items = self.collection.get(include=['metadatas'])
            if all_items['ids']:
                self.collection.delete(ids=all_items['ids'])
                print(f"✅ Cleared {len(all_items['ids'])} items from collection")
            else:
                print("✅ Collection was already empty")
        except Exception as e:
            raise Exception(f"Failed to clear collection: {str(e)}")

# Convenience functions for quick testing
def create_chromadb_service(persist_directory: str = "./chroma_db") -> ChromaDBService:
    """Quick function to create a ChromaDB service"""
    return ChromaDBService(persist_directory)

def test_search(query: str, query_embedding: List[float], n_results: int = 5):
    """Quick function to test search"""
    service = ChromaDBService()
    return service.semantic_search(query, query_embedding, n_results)