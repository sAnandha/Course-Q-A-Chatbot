from typing import List, Dict, Any
from app.services.vector_store import VectorStore
from app.services.bm25_search import BM25Search
from app.services.cross_encoder import CrossEncoderReranker

class HybridRetriever:
    def __init__(self, use_pinecone: bool = False):
        self.vector_store = VectorStore(use_pinecone)
        self.bm25_search = BM25Search()
        self.cross_encoder = CrossEncoderReranker()
        self.documents = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to all search indices"""
        self.documents = documents
        self.vector_store.add_documents(documents)
        self.bm25_search.add_documents(documents)
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Hybrid search: Vector + BM25 + Cross-encoder reranking"""
        
        # 1. Vector search
        vector_results = self.vector_store.search(query, top_k)
        
        # 2. BM25 search  
        bm25_results = self.bm25_search.search(query, top_k)
        
        # 3. Combine results
        combined_results = self._combine_results(vector_results, bm25_results)
        
        # 4. Cross-encoder reranking
        if combined_results:
            reranked_results = self.cross_encoder.rerank(query, combined_results, top_k)
            return reranked_results
        
        return combined_results[:top_k]
    
    def _combine_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate vector and BM25 results"""
        seen_chunks = {}
        combined = []
        
        # Add vector results
        for doc in vector_results:
            chunk_id = doc.get('chunk_id')
            if chunk_id not in seen_chunks:
                doc['retrieval_method'] = 'vector'
                seen_chunks[chunk_id] = doc
                combined.append(doc)
        
        # Add BM25 results (merge scores if duplicate)
        for doc in bm25_results:
            chunk_id = doc.get('chunk_id')
            if chunk_id in seen_chunks:
                # Merge scores for documents found by both methods
                existing_doc = seen_chunks[chunk_id]
                existing_doc['bm25_score'] = doc.get('bm25_score', 0)
                existing_doc['retrieval_method'] = 'hybrid'
                # Combined score: vector + BM25
                vector_score = existing_doc.get('vector_score', 0)
                bm25_score = doc.get('bm25_score', 0)
                existing_doc['score'] = vector_score * 0.6 + bm25_score * 0.4
            else:
                # New document from BM25 only
                doc['retrieval_method'] = 'bm25'
                doc['score'] = doc.get('bm25_score', 0)
                seen_chunks[chunk_id] = doc
                combined.append(doc)
        
        # Sort by combined score
        combined.sort(key=lambda x: x.get('score', 0), reverse=True)
        return combined