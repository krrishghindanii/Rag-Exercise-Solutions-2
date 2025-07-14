import os
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator
from src.config import Config

# Load environment variables
load_dotenv()

# Initialize components
dp = DocumentProcessor()
em = EmbeddingManager()
dr = DocumentRetriever()

# Check if OpenAI API key is set
if not Config.OPENAI_API_KEY:
    print("ERROR: Please set OPENAI_API_KEY in your .env file")
    print("Copy .env.example to .env and add your OpenAI API key")
    exit(1)

rg = ResponseGenerator(Config.OPENAI_API_KEY)

# Ensure ChromaDB is populated
if dr.collection.count() == 0:
    print("Populating ChromaDB...")
    docs = dp.load_documents('documents')
    
    all_chunks = []
    for doc in docs:
        chunks = dp.chunk_text(doc['content'], {'source': doc['source'], **doc['metadata']})
        all_chunks.extend(chunks)
    
    texts = [chunk['content'] for chunk in all_chunks]
    embeddings = em.generate_embeddings(texts)
    dr.add_documents(all_chunks, embeddings.tolist())
    print(f"Stored {len(all_chunks)} chunks in ChromaDB\n")

# Test queries
test_queries = [
    "What are the vacation days policy for employees with 3 years of experience?",
    "How do I book the Golden Gate conference room?",
    "What is the API rate limit for enterprise customers?",
    "Who is the Engineering Manager and what's their email?"
]

print("="*60)
print("COMPLETE RAG SYSTEM TEST")
print("="*60)

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print("="*60)
    
    # Generate query embedding
    query_embedding = em.embed_query(query)
    
    # Search for relevant documents
    results = dr.search(query_embedding.tolist(), top_k=5)
    
    print(f"\nRetrieved {len(results)} relevant chunks:")
    for i, result in enumerate(results[:3]):  # Show top 3
        print(f"  {i+1}. {result['metadata'].get('source')} (similarity: {result['similarity_score']:.3f})")
    
    # Generate response
    print("\nGenerating response...")
    response = rg.generate_response(query, results)
    
    print(f"\nANSWER:\n{response}")
    print("\n" + "-"*60)

print("\n\nRAG system test complete!")