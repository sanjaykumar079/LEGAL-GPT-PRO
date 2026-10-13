import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.chromadb_service import ChromaDBService
from services.embedding_service import EmbeddingService
from services.chunking_service import ChunkingService
from services.pdf_service import PDFService
import os

def main():
    print("=== DAY 5: ChromaDB Vector Database Test ===\n")
    
    # Initialize services
    print("1️⃣ Initializing services...")
    embedding_service = EmbeddingService()
    chunking_service = ChunkingService(chunk_size=800, overlap=100)  # Smaller for testing
    chroma_service = ChromaDBService(persist_directory="./test_chroma_db")
    
    # Clear any existing data
    print("\n2️⃣ Clearing existing data...")
    chroma_service.clear_collection()
    
    # Test with sample PDF
    test_pdf = "tests/fixtures/test_legal_doc.pdf"
    
    if not os.path.exists(test_pdf):
        print("❌ Test PDF not found!")
        print("Please run: python scripts/create_test_pdf.py")
        return
    
    print(f"\n3️⃣ Processing: {test_pdf}")
    
    # Extract text from PDF
    print("\n4️⃣ Extracting text from PDF...")
    pdf_service = PDFService()
    pages = pdf_service.extract_text_from_pdf(test_pdf)
    print(f"   ✅ Extracted {len(pages)} pages")
    
    # Chunk the document
    print("\n5️⃣ Chunking document...")
    cleaned_pages = []
    for page in pages:
        cleaned_pages.append({
            'page': page.page_number,
            'text': page.text
        })
    
    all_chunks = chunking_service.chunk_document_pages(cleaned_pages, os.path.basename(test_pdf))
    print(f"   ✅ Created {len(all_chunks)} chunks")
    
    # Generate embeddings for chunks
    print("\n6️⃣ Generating embeddings for chunks...")
    chunks_with_embeddings = []
    
    for i, chunk in enumerate(all_chunks):
        if i >= 20:  # Limit to 20 chunks for testing
            break
            
        chunk_with_emb = embedding_service.chunk_to_embedding(chunk)
        chunks_with_embeddings.append(chunk_with_emb)
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1}/{min(20, len(all_chunks))} chunks...")
    
    print(f"   ✅ Generated embeddings for {len(chunks_with_embeddings)} chunks")
    
    # Add to ChromaDB
    print("\n7️⃣ Adding chunks to ChromaDB...")
    chunk_ids = chroma_service.add_chunks(chunks_with_embeddings)
    print(f"   ✅ Added {len(chunk_ids)} chunks with IDs: {chunk_ids[:3]}...")
    
    # Show collection stats
    print("\n8️⃣ Collection Statistics:")
    stats = chroma_service.get_collection_stats()
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Test semantic search
    print("\n9️⃣ Testing semantic search...")
    
    # Test queries
    test_queries = [
        "What did the court say about the appeal?",
        "Find information about legal proceedings",
        "Contract law and agreements"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Generate query embedding
        query_embedding = embedding_service.generate_embedding(query)
        
        # Search ChromaDB
        results = chroma_service.semantic_search(
            query=query,
            query_embedding=query_embedding.tolist(),
            n_results=3
        )
        
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Similarity: {result['similarity']:.4f}")
            print(f"      Document: {result['metadata']['document']}")
            print(f"      Page: {result['metadata']['page']}")
            print(f"      Text: {result['text'][:100]}...")
    
    # Test metadata filtering
    print("\n10️⃣ Testing metadata filtering...")
    
    # Filter by page number
    filter_results = chroma_service.semantic_search(
        query="court findings",
        query_embedding=embedding_service.generate_embedding("court findings").tolist(),
        n_results=5,
        filter_metadata={"page": 2}  # Only page 2
    )
    
    print(f"   Filtered results (page=2): {len(filter_results)} found")
    for result in filter_results:
        print(f"   - Page {result['metadata']['page']}: {result['text'][:80]}...")
    
    print("\n✅ Day 5 test completed successfully!")
    print("Ready for Day 6: RAG Pipeline Integration")

if __name__ == "__main__":
    main()