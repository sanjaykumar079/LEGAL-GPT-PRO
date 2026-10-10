import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.pdf_service import PDFService
from services.chunking_service import ChunkingService
from services.text_cleaning_service import TextCleaningService

def main():
    print("=== DAY 3: Text Cleaning + Chunking Test ===\n")
    
    # Initialize services
    pdf_service = PDFService()
    chunking_service = ChunkingService(chunk_size=1200, overlap=200)
    cleaning_service = TextCleaningService()
    
    # Test with sample PDF
    test_pdf = "tests/fixtures/test_legal_doc.pdf"
    
    if not os.path.exists(test_pdf):
        print("❌ Test PDF not found!")
        print("Please run: python scripts/create_test_pdf.py")
        return
    
    print(f"📄 Processing: {test_pdf}")
    
    # Step 1: Extract text from PDF
    print("\n1️⃣ Extracting text from PDF...")
    pages = pdf_service.extract_text_from_pdf(test_pdf)
    print(f"   ✅ Extracted {len(pages)} pages")
    
    # Step 2: Clean text
    print("\n2️⃣ Cleaning text...")
    cleaned_pages = []
    for page in pages:
        cleaned_text = cleaning_service.clean_text(page.text)
        cleaned_pages.append({
            'page': page.page_number,
            'text': cleaned_text,
            'original_word_count': page.word_count,
            'cleaned_word_count': len(cleaned_text.split())
        })
    print(f"   ✅ Cleaned text for {len(cleaned_pages)} pages")
    
    # Step 3: Chunk the document
    print("\n3️⃣ Chunking document...")
    all_chunks = chunking_service.chunk_document_pages(cleaned_pages, os.path.basename(test_pdf))
    print(f"   ✅ Created {len(all_chunks)} chunks")
    
    # Step 4: Display statistics
    print("\n📊 STATISTICS:")
    stats = chunking_service.get_chunk_stats(all_chunks)
    for key, value in stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Step 5: Show first 5 chunks
    print("\n📋 FIRST 5 CHUNKS:")
    for i, chunk in enumerate(all_chunks[:5], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Document: {chunk['document']}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Characters: {chunk['char_count']}")
        print(f"Text Preview: {chunk['text'][:150]}...")
    
    # Step 6: Show metadata example
    print("\n🏷️  METADATA EXAMPLE:")
    if all_chunks:
        example_chunk = all_chunks[0]
        print("Complete chunk metadata:")
        for key, value in example_chunk.items():
            if key != 'text':  # Skip full text for display
                print(f"   {key}: {value}")
    
    print("\n✅ Day 3 test completed successfully!")
    print("Ready for Day 4: Embedding Service")

if __name__ == "__main__":
    main()