import json
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
import hashlib


class SimpleVectorDB:
    def __init__(self, persist_directory: str = "./vector_db"):
        self.persist_directory = persist_directory
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        
        os.makedirs(persist_directory, exist_ok=True)
        self.db_file = os.path.join(persist_directory, "vectordb.pkl")
        self.load()
    
    def save(self):
        """Save the database to disk"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadatas': self.metadatas,
            'ids': self.ids
        }
        with open(self.db_file, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self):
        """Load the database from disk"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'rb') as f:
                    data = pickle.load(f)
                self.documents = data.get('documents', [])
                self.embeddings = data.get('embeddings', [])
                self.metadatas = data.get('metadatas', [])
                self.ids = data.get('ids', [])
            except Exception as e:
                print(f"Warning: Could not load existing database: {e}")
                self.documents = []
                self.embeddings = []
                self.metadatas = []
                self.ids = []
    
    def add(self, documents: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict], ids: List[str]):
        """Add documents to the database"""
        # Remove existing documents with same IDs
        for doc_id in ids:
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.documents.pop(idx)
                self.embeddings.pop(idx)
                self.metadatas.pop(idx)
                self.ids.pop(idx)
        
        # Add new documents
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        self.save()
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, List]:
        """Query the database for similar documents"""
        if not self.embeddings:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]],
                'ids': [[]]
            }
        
        # Calculate similarities
        similarities = []
        for embedding in self.embeddings:
            sim = self.cosine_similarity(query_embedding, embedding)
            similarities.append(sim)
        
        # Get top n results
        indices = np.argsort(similarities)[::-1][:n_results]
        
        results = {
            'documents': [[self.documents[i] for i in indices]],
            'metadatas': [[self.metadatas[i] for i in indices]],
            'distances': [[1 - similarities[i] for i in indices]],  # Convert similarity to distance
            'ids': [[self.ids[i] for i in indices]]
        }
        
        return results
    
    def count(self) -> int:
        """Return the number of documents in the database"""
        return len(self.documents)
    
    def remove_by_source(self, source_path: str) -> int:
        """Remove all chunks from a specific document source"""
        removed_count = 0
        indices_to_remove = []
        
        # Find all indices with matching source
        for i, metadata in enumerate(self.metadatas):
            if metadata.get('source') == source_path:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain correct indices
        for i in reversed(indices_to_remove):
            self.documents.pop(i)
            self.embeddings.pop(i)
            self.metadatas.pop(i)
            self.ids.pop(i)
            removed_count += 1
        
        if removed_count > 0:
            self.save()
        
        return removed_count
    
    def list_sources(self) -> List[str]:
        """Get list of unique document sources"""
        sources = set()
        for metadata in self.metadatas:
            if 'source' in metadata:
                sources.add(metadata['source'])
        return sorted(list(sources))
    
    def get_source_info(self) -> Dict[str, int]:
        """Get document sources with chunk counts"""
        source_counts = {}
        for metadata in self.metadatas:
            source = metadata.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        return source_counts
    
    def clear_all(self) -> int:
        """Remove all documents from the database"""
        count = len(self.documents)
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
        self.save()
        return count