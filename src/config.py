"""
Configuration settings for the RAG system
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Retrieval settings
    TOP_K_DOCUMENTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Generation settings
    MAX_TOKENS = 500
    TEMPERATURE = 0.1
    
    # Paths
    DOCUMENTS_DIR = "documents"
    VECTOR_STORE_DIR = "data/vector_store"
    
    # Embedding model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"