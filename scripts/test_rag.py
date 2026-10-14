import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.rag_service import RAGService
from services.embedding_service import EmbeddingService
from services.chunking_service import ChunkingService
from services.pdf_service import PDFService
import os

def main():
    print("=== DAY 6: Basic RAG Pipeline Test ===\n")
    
    # Initialize RAG service
    print("1️⃣ Initializing RAG service...")
    
    try:
        rag_service = RAGService()
    except Exception as e:
        print(f"❌ Failed to initialize RAG service: {e}")
        print("Make sure Ollama is running: ollama serve")
        return
    
    # Check if we have data in the database
    print("\n2️⃣ Checking knowledge base...")
    stats = rag_service.get_knowledge_base_stats()
    print(f"   Total chunks in database: {stats.get('total_chunks', 0)}")
    
    if stats.get('total_chunks', 0) == 0:
        print("   ⚠️  No data in knowledge base. Adding test data...")
        
        # Process test PDF and add to database
        test_pdf = "tests/fixtures/test_legal_doc.pdf"
        
        if os.path.exists(test_pdf):
            print(f"   Processing: {test_pdf}")
            
            # Extract text
            pdf_service = PDFService()
            pages = pdf_service.extract_text_from_pdf(test_pdf)
            
            # Chunk document
            chunking_service = ChunkingService(chunk_size=800, overlap=100)
            cleaned_pages = [{'page': page.page_number, 'text': page.text} for page in pages]
            all_chunks = chunking_service.chunk_document_pages(cleaned_pages, os.path.basename(test_pdf))
            
            # Generate embeddings
            embedding_service = EmbeddingService()  
            chunks_with_embeddings = []
            
            for chunk in all_chunks:
                chunk_with_emb = embedding_service.chunk_to_embedding(chunk)
                chunks_with_embeddings.append(chunk_with_emb)
            
            # Add to database
            chunk_ids = rag_service.add_document_chunks(chunks_with_embeddings)
            print(f"   ✅ Added {len(chunk_ids)} chunks to knowledge base")
        else:
            print("   ❌ Test PDF not found. Please run: python scripts/create_test_pdf.py")
            return
    
    # Test questions
    print("\n3️⃣ Testing RAG pipeline...")
    
    test_questions = [
        "What was the decision of the court?",
        "What information is provided about the case?",
        "Find details about legal proceedings"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print('='*60)
        
        try:
            # Get answer
            result = rag_service.answer_question(question, n_results=3)
            
            if result['error']:
                print(f"❌ Error: {result['answer']}")
            else:
                print(f"🤖 Answer:")
                print(f"   {result['answer']}")
                
                print(f"\n📚 Sources used ({result['retrieved_chunks']} chunks):")
                for j, source in enumerate(result['sources'], 1):
                    print(f"   {j}. {source['metadata']['document']} (Page {source['metadata']['page']})")
                    print(f"      Similarity: {source['similarity']:.4f}")
                    print(f"      Text: {source['text'][:100]}...")
                    
        except Exception as e:
            print(f"❌ Failed to answer question: {e}")
    
    # Test search without generation
    print(f"\n{'='*60}")
    print("Testing search functionality (no generation)")
    print('='*60)
    
    try:
        search_results = rag_service.search_similar("court decision", n_results=2)
        print(f"Search results for 'court decision':")
        for i, result in enumerate(search_results, 1):
            print(f"{i}. {result['metadata']['document']} (Page {result['metadata']['page']})")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Text: {result['text'][:80]}...")
    except Exception as e:
        print(f"❌ Search failed: {e}")
    
    print("\n✅ Day 6 RAG test completed!")
    print("Ready for Day 7: Advanced Features & Evaluation")

if __name__ == "__main__":
    main()