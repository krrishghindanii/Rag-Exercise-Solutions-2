from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.retriever import DocumentRetriever

# Initialize components
dp = DocumentProcessor()
em = EmbeddingManager()
dr = DocumentRetriever()

# First, populate the database if it's empty
if dr.collection.count() == 0:
    print("Populating ChromaDB...")
    docs = dp.load_documents('documents')
    
    # Process all documents
    all_chunks = []
    for doc in docs:
        chunks = dp.chunk_text(doc['content'], doc['metadata'])
        all_chunks.extend(chunks)
    
    print(f"Total chunks to process: {len(all_chunks)}")
    
    # Generate embeddings
    texts = [chunk['content'] for chunk in all_chunks]
    embeddings = em.generate_embeddings(texts)
    
    # Store in ChromaDB
    dr.add_documents(all_chunks, embeddings.tolist())
    print(f"Stored {len(all_chunks)} chunks in ChromaDB")
else:
    print(f"ChromaDB already contains {dr.collection.count()} documents")

# Test search functionality
test_queries = [
    "What are the vacation days policy?",
    "How do I book a meeting room?",
    "What is the API rate limit?",
    "Who is the Engineering Manager?"
]

print("\n" + "="*50)
print("TESTING SEARCH FUNCTIONALITY")
print("="*50)

for query in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 40)
    
    # Generate query embedding
    query_embedding = em.embed_query(query)
    
    # Search for similar documents
    results = dr.search(query_embedding.tolist(), top_k=3)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
        print(f"  Chunk: {result['metadata'].get('chunk_index', 'Unknown')}")
        print(f"  Similarity: {result['similarity_score']:.3f}")
        print(f"  Preview: {result['content'][:150]}...")