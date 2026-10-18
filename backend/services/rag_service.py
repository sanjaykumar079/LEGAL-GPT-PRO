from typing import List, Dict, Any, Optional
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from services.llm_service import LLMService

class RAGService:
    def __init__(self, chroma_persist_directory: str = "./chroma_db", 
                 llm_model: str = "llama3:latest"):
        """
        Initialize RAG service with configurable components
        """
        print("Initializing RAG Service...")
        self.embedding_service = EmbeddingService()
        self.chroma_service = ChromaDBService(persist_directory=chroma_persist_directory)
        self.llm_service = LLMService(model_name=llm_model)
        print("✅ RAG Service ready")
    
    def answer_question(self, question: str, top_k: int = 5, 
                       filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Answer a question with Top-K retrieval and automatic source citation mapping
        
        Args:
            question: Query text
            top_k: Number of chunks to retrieve (3, 5, 10)
            filter_metadata: Optional dict for ChromaDB filtering
            
        Returns:
            Dict containing answer, formatted sources list, and structured citations
        """
        print(f"\n🔍 Processing Query: '{question}' (Top-K={top_k})")
        
        # 1. Embed Query
        try:
            query_embedding = self.embedding_service.generate_embedding(question)
        except Exception as e:
            return {'answer': f"Embedding Error: {str(e)}", 'sources': [], 'error': True}
        
        # 2. Retrieve Top-K Chunks
        try:
            retrieved_chunks = self.chroma_service.semantic_search(
                query=question,
                query_embedding=query_embedding.tolist(),
                n_results=top_k,
                filter_metadata=filter_metadata
            )
        except Exception as e:
            return {'answer': f"Database Retrieval Error: {str(e)}", 'sources': [], 'error': True}
        
        if not retrieved_chunks:
            return {
                'answer': "No relevant legal context was found in the database.",
                'sources': [],
                'formatted_citations': "No citations available.",
                'retrieved_chunks_count': 0,
                'error': False
            }
        
        # 3. Generate LLM Answer
        try:
            raw_answer = self.llm_service.answer_question_with_context(question, retrieved_chunks)
        except Exception as e:
            return {'answer': f"LLM Generation Error: {str(e)}", 'sources': retrieved_chunks, 'error': True}
        
        # 4. Construct Verified Citations from Chunk Metadata
        citations = self._format_citations(retrieved_chunks)
        
        return {
            'question': question,
            'answer': raw_answer,
            'sources': retrieved_chunks,
            'citations': citations,
            'retrieved_chunks_count': len(retrieved_chunks),
            'top_k_requested': top_k,
            'error': False
        }
    
    def _format_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract unique source citations from metadata to prevent duplicates
        """
        seen = set()
        citations = []
        
        for chunk in chunks:
            doc = chunk['metadata'].get('document', 'Unknown Document')
            page = chunk['metadata'].get('page', 'N/A')
            chunk_id = chunk.get('id', 'N/A')
            similarity = chunk.get('similarity', 0.0)
            
            key = f"{doc}_p{page}"
            if key not in seen:
                seen.add(key)
                citations.append({
                    'document': doc,
                    'page': page,
                    'chunk_id': chunk_id,
                    'relevance_score': round(similarity, 4),
                    'preview': chunk['text'][:120] + "..."
                })
        
        return citations

    def add_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        return self.chroma_service.add_chunks(chunks)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        return self.chroma_service.get_collection_stats()