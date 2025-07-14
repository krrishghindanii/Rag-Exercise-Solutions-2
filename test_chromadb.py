from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingManager
from src.retriever import DocumentRetriever

# Initialize components
dp = DocumentProcessor()
em = EmbeddingManager()
dr = DocumentRetriever()

# Load and process documents
print("Loading documents...")
docs = dp.load_documents('documents')

# Process first document as a test
if docs:
    print(f"\nProcessing {docs[0]['source']}...")
    
    # Chunk the document
    chunks = dp.chunk_text(docs[0]['content'], docs[0]['metadata'])
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings
    print("Generating embeddings...")
    texts = [chunk['content'] for chunk in chunks]
    embeddings = em.generate_embeddings(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Store in ChromaDB
    print("Storing in ChromaDB...")
    dr.add_documents(chunks, embeddings.tolist())
    
    # Verify storage
    print(f"\nTotal documents in ChromaDB: {dr.collection.count()}")
    
    # Now check the directory
    import os
    if os.path.exists('./chroma_db'):
        print("\nchroma_db directory created successfully!")
