from typing import List, Dict, Any, Optional
import os
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from services.llm_service import LLMService

class RAGService:
    def __init__(self, chroma_persist_directory: str = "./chroma_db", 
                 llm_model: str = "llama2"):
        """
        Initialize RAG service that combines retrieval and generation
        
        Args:
            chroma_persist_directory: ChromaDB persistence directory
            llm_model: Ollama model name for generation
        """
        print("Initializing RAG Service...")
        
        # Initialize components
        self.embedding_service = EmbeddingService()
        self.chroma_service = ChromaDBService(persist_directory=chroma_persist_directory)
        self.llm_service = LLMService(model_name=llm_model)
        
        print("✅ RAG Service initialized successfully")
    
    def answer_question(self, question: str, n_results: int = 5, 
                       filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Answer a question using RAG pipeline
        
        Args:
            question: The question to answer
            n_results: Number of context chunks to retrieve
            filter_metadata: Optional metadata filter for search
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print(f"\n🔍 Processing question: '{question}'")
        
        # Step 1: Generate embedding for the question
        print("   Generating question embedding...")
        try:
            question_embedding = self.embedding_service.generate_embedding(question)
        except Exception as e:
            return {
                'answer': f"Error generating embedding: {str(e)}",
                'sources': [],
                'error': True
            }
        
        # Step 2: Search ChromaDB for relevant chunks
        print(f"   Searching for {n_results} relevant chunks...")
        try:
            search_results = self.chroma_service.semantic_search(
                query=question,
                query_embedding=question_embedding.tolist(),
                n_results=n_results,
                filter_metadata=filter_metadata
            )
        except Exception as e:
            return {
                'answer': f"Error searching database: {str(e)}",
                'sources': [],
                'error': True
            }
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant information in the documents to answer your question.",
                'sources': [],
                'retrieved_chunks': 0
            }
        
        print(f"   Found {len(search_results)} relevant chunks")
        
        # Step 3: Generate answer using LLM with context
        print("   Generating answer with LLM...")
        try:
            answer = self.llm_service.answer_question_with_context(question, search_results)
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': search_results,
                'error': True
            }
        
        # Prepare response
        response = {
            'answer': answer,
            'sources': search_results,
            'retrieved_chunks': len(search_results),
            'question': question,
            'error': False
        }
        
        print("✅ Answer generated successfully")
        return response
    
    def add_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add document chunks to the knowledge base
        
        Args:
            chunks: List of chunks with embeddings and metadata
            
        Returns:
            List of chunk IDs added
        """
        return self.chroma_service.add_chunks(chunks)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return self.chroma_service.get_collection_stats()
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks without generating answer
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar chunks
        """
        query_embedding = self.embedding_service.generate_embedding(query)
        
        return self.chroma_service.semantic_search(
            query=query,
            query_embedding=query_embedding.tolist(),
            n_results=n_results
        )