import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from services.embedding_service import EmbeddingService
import numpy as np

def main():
    print("=== DAY 4: Embeddings Service Test ===\n")
    
    # Initialize embedding service
    print("1️⃣ Initializing embedding service...")
    embedding_service = EmbeddingService()
    
    # Show model info
    print("\n📊 MODEL INFO:")
    model_info = embedding_service.get_model_info()
    for key, value in model_info.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Test single embedding
    print("\n2️⃣ Testing single text embedding...")
    test_text = "The court rejected the appeal."
    embedding = embedding_service.generate_embedding(test_text)
    
    print(f"   Text: '{test_text}'")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5].tolist()}")
    print(f"   Embedding type: {type(embedding)}")
    
    # Test multiple embeddings
    print("\n3️⃣ Testing multiple text embeddings...")
    test_texts = [
        "The court rejected the appeal.",
        "The appeal was dismissed by the court.",
        "The weather is good today.",
        "Contract law governs agreements between parties.",
        "The defendant was found guilty."
    ]
    
    embeddings = embedding_service.generate_embeddings(test_texts)
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Each embedding has {len(embeddings[0])} dimensions")
    
    # Test semantic similarity
    print("\n4️⃣ Testing semantic similarity...")
    
    # Test cases
    sentence_a = "The court rejected the appeal."
    sentence_b = "The appeal was dismissed by the court."
    sentence_c = "The weather is good today."
    
    print(f"   Sentence A: '{sentence_a}'")
    print(f"   Sentence B: '{sentence_b}'")
    print(f"   Sentence C: '{sentence_c}'")
    
    # Calculate similarities
    sim_ab = embedding_service.semantic_similarity(sentence_a, sentence_b)
    sim_ac = embedding_service.semantic_similarity(sentence_a, sentence_c)
    sim_bc = embedding_service.semantic_similarity(sentence_b, sentence_c)
    
    print(f"\n   Similarity A-B: {sim_ab:.4f}")
    print(f"   Similarity A-C: {sim_ac:.4f}")
    print(f"   Similarity B-C: {sim_bc:.4f}")
    
    # Verify semantic relationships
    print("\n5️⃣ Verifying semantic relationships...")
    if sim_ab > sim_ac:
        print("   ✅ A and B are more similar than A and C (as expected)")
    else:
        print("   ❌ Unexpected: A and C are more similar than A and B")
    
    if sim_ab > 0.7:
        print("   ✅ A and B have high semantic similarity")
    else:
        print("   ❌ A and B similarity is lower than expected")
    
    # Test find most similar
    print("\n6️⃣ Testing 'find most similar' functionality...")
    query = "Legal appeal decision"
    texts_to_search = [
        "The court made a decision on the appeal.",
        "I went to the store to buy milk.",
        "The appeal was denied by the appellate court.",
        "Criminal law deals with crimes and punishments.",
        "The judge ruled on the motion."
    ]
    
    similar_results = embedding_service.find_most_similar(query, texts_to_search, top_k=3)
    
    print(f"   Query: '{query}'")
    print(f"   Top 3 most similar texts:")
    for i, result in enumerate(similar_results, 1):
        print(f"   {i}. '{result['text']}' (similarity: {result['similarity']:.4f})")
    
    # Test chunk embedding
    print("\n7️⃣ Testing chunk embedding...")
    sample_chunk = {
        'text': 'The contract was terminated due to breach of terms.',
        'document': 'contract_case.pdf',
        'page': 5,
        'chunk_id': 'contract_case_5_001'
    }
    
    chunk_with_embedding = embedding_service.chunk_to_embedding(sample_chunk)
    
    print(f"   Original chunk keys: {list(sample_chunk.keys())}")
    print(f"   Chunk with embedding keys: {list(chunk_with_embedding.keys())}")
    print(f"   Embedding dimension: {chunk_with_embedding['embedding_dim']}")
    print(f"   Embedding is list: {isinstance(chunk_with_embedding['embedding'], list)}")
    
    print("\n✅ Day 4 test completed successfully!")
    print("Ready for Day 5: Vector Database (ChromaDB)")

if __name__ == "__main__":
    main()