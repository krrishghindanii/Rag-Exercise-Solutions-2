"""
This file provides document retrieval using vector similarity and keyword search. It manages adding documents, searching by embedding, and hybrid search for best results.
"""
import chromadb
import uuid
from typing import List, Dict, Tuple
import numpy as np

class DocumentRetriever:
    def __init__(self, collection_name: str = "company_docs"):
        # This function initializes the ChromaDB client and gets or creates the document collection.
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path="./chroma_db")
    
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"Initialized collection: {collection_name}")

    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """
        This function adds a list of documents and their embeddings to the vector store.
        Each document gets a unique ID and its metadata is prepared for storage.
        """
        ids = []
        metadatas = []
        texts = []

        for i, doc in enumerate(documents):
            # This line creates a unique ID for each document chunk.
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            
            # This line extracts the text content from the document.
            texts.append(doc['content'])
            
            # This block prepares and updates the metadata for each document.
            source = 'unknown'
            if 'metadata' in doc and isinstance(doc['metadata'], dict):
                source = doc['metadata'].get('source', doc['metadata'].get('file_name', 'unknown'))
            
            metadata = {
                'source': source,
                'chunk_index': doc.get('chunk_index', i),
                'start_char': doc.get('start_char', 0),
                'end_char': doc.get('end_char', 0)
            }
            if 'metadata' in doc and isinstance(doc['metadata'], dict):
                metadata.update(doc['metadata'])
            metadatas.append(metadata)
        
        # This function adds all documents, embeddings, and metadata to ChromaDB.
        self.collection.add(
            ids=ids,
            embeddings=np.array(embeddings),
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to ChromaDB")

    def search(self, query_embedding: List[float], top_k: int = 5, file_type: str = None, date_from: float = None, date_to: float = None) -> List[Dict]:
        """
        This function searches for documents most similar to the given query embedding.
        It returns the top_k results with similarity scores and metadata.
        Optionally filters by file_type and created_date.
        """
        # Build metadata filter
        where = {}
        if file_type:
            where['file_type'] = file_type
        if date_from is not None or date_to is not None:
            date_filter = {}
            if date_from is not None:
                date_filter['$gte'] = date_from
            if date_to is not None:
                date_filter['$lte'] = date_to
            where['created_date'] = date_filter
        
        # This line performs a similarity search in ChromaDB using the query embedding.
        search_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
            where=where if where else None
        )
        
        # This block formats the search results into a list of dictionaries.
        results = []
        if (
            search_results['ids'] and len(search_results['ids'][0]) > 0
            and search_results['documents'] is not None
            and search_results['metadatas'] is not None
            and search_results['distances'] is not None
        ):
            for i in range(len(search_results['ids'][0])):
                result = {
                    'id': search_results['ids'][0][i],
                    'content': search_results['documents'][0][i],
                    'metadata': search_results['metadatas'][0][i],
                    'distance': search_results['distances'][0][i],
                    'similarity_score': 1 - search_results['distances'][0][i]  # This converts distance to similarity.
                }
                results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        This function performs a hybrid search by combining semantic (vector) and keyword (lexical) search using ChromaDB.
        It returns the top_k unique results, prioritizing those that appear in both searches.
        """
        # This block performs a semantic search using the query embedding.
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # This gets more results to allow for overlap.
            include=["documents", "metadatas", "distances"]
        )
        semantic_docs = {}
        if (
            semantic_results['ids'] and len(semantic_results['ids'][0]) > 0
            and semantic_results['documents'] is not None
            and semantic_results['metadatas'] is not None
            and semantic_results['distances'] is not None
        ):
            for i in range(len(semantic_results['ids'][0])):
                doc_id = semantic_results['ids'][0][i]
                semantic_docs[doc_id] = {
                    'id': doc_id,
                    'content': semantic_results['documents'][0][i],
                    'metadata': semantic_results['metadatas'][0][i],
                    'distance': semantic_results['distances'][0][i],
                    'similarity_score': 1 - semantic_results['distances'][0][i],
                    'source': 'semantic'
                }

        # This block performs a keyword (lexical) search using the query string.
        keyword_results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,  # This gets more results to allow for overlap.
            where_document={"$contains": query},
            include=["documents", "metadatas"]
        )
        keyword_docs = {}
        if (
            keyword_results['ids'] and len(keyword_results['ids'][0]) > 0
            and keyword_results['documents'] is not None
            and keyword_results['metadatas'] is not None
        ):
            for i in range(len(keyword_results['ids'][0])):
                doc_id = keyword_results['ids'][0][i]
                keyword_docs[doc_id] = {
                    'id': doc_id,
                    'content': keyword_results['documents'][0][i],
                    'metadata': keyword_results['metadatas'][0][i],
                    'distance': None,
                    'similarity_score': None,
                    'source': 'keyword'
                }

        # This block merges the results, prioritizing documents found in both searches.
        merged = {}
        for doc_id, doc in semantic_docs.items():
            merged[doc_id] = doc
        for doc_id, doc in keyword_docs.items():
            if doc_id in merged:
                merged[doc_id]['source'] = 'hybrid'  # This marks documents found in both searches.
            else:
                merged[doc_id] = doc

        # This function sorts the results: hybrid > semantic > keyword, then by similarity score if available.
        def sort_key(doc):
            source_priority = {'hybrid': 0, 'semantic': 1, 'keyword': 2}
            return (
                source_priority.get(doc['source'], 3),
                -(doc['similarity_score'] if doc['similarity_score'] is not None else 0)
            )
        sorted_results = sorted(merged.values(), key=sort_key)
        return sorted_results[:top_k]
