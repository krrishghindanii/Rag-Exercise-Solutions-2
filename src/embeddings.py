"""
This file manages embedding generation for text and queries. It wraps a sentence transformer model and provides easy-to-use functions for getting embeddings.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # This function loads the sentence transformer model for generating embeddings.
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        This function takes a list of text strings and returns their embeddings as a NumPy array.
        It uses the loaded model to encode all texts at once.
        
        Args:
            texts: A list of text strings to embed.
        Returns:
            A NumPy array containing the embeddings for each text.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        This function generates an embedding for a single query string.
        It returns the embedding as a NumPy array.
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding
