import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone

class VectorStore:
    def __init__(self, use_pinecone: bool = False):
        self.use_pinecone = use_pinecone
        # Use multilingual model compatible with 384 dimensions
        try:
            self.local_embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # 384 dims
            print("Using multilingual MiniLM embeddings (384 dimensions)")
        except:
            self.local_embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fallback 384 dims
            print("Using MiniLM embeddings (fallback)")
        
        if use_pinecone:
            try:
                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                self.index = pc.Index("course-qa-index")
                print("Connected to Pinecone index: course-qa-index")
            except Exception as e:
                print(f"Pinecone not available: {e}, using local storage")
                self.use_pinecone = False
                self._init_local_storage()
        else:
            self._init_local_storage()
    
    def _init_local_storage(self):
        """Initialize local vector storage"""
        self.vectors = {}
        self.metadata = {}
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        for doc in documents:
            # Get embedding (fallback to local model if AWS fails)
            try:
                embedding = doc.get('embedding')
                if not embedding:
                    embedding = self.local_embedder.encode(doc['text']).tolist()
            except:
                embedding = self.local_embedder.encode(doc['text']).tolist()
            
            doc_id = doc['chunk_id']
            
            if self.use_pinecone:
                # Store in Pinecone with metadata
                self.index.upsert([
                    {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": {
                            "text": str(doc['text'][:1000]),  # Ensure string
                            "chunk_id": str(doc['chunk_id']),
                            "doc_id": str(doc.get('doc_id', '')),
                            "page_number": int(doc.get('page_number') or 1)  # Ensure integer
                        }
                    }
                ])
            else:
                # Store locally
                self.vectors[doc_id] = embedding
                self.metadata[doc_id] = doc
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        try:
            # Get query embedding (fallback to local)
            query_embedding = self.local_embedder.encode(query).tolist()
        except:
            return []
        
        if self.use_pinecone:
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            pinecone_results = []
            for match in results.get('matches', []):
                metadata = match.get('metadata', {})
                metadata['vector_score'] = match.get('score', 0)
                metadata['chunk_id'] = metadata.get('chunk_id', match['id'])
                pinecone_results.append(metadata)
            
            return pinecone_results
        else:
            # Local similarity search
            similarities = []
            for doc_id, doc_embedding in self.vectors.items():
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append((similarity, doc_id))
            
            # Sort by similarity and return top results
            similarities.sort(reverse=True)
            results = []
            for similarity, doc_id in similarities[:top_k]:
                doc = self.metadata[doc_id].copy()
                doc['vector_score'] = similarity
                results.append(doc)
            
            return results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))