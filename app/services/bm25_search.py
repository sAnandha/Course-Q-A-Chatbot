from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re

class BM25Search:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_metadata = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to BM25 index"""
        self.documents = []
        self.doc_metadata = []
        
        for doc in documents:
            # Tokenize document text
            tokens = self._tokenize(doc['text'])
            self.documents.append(tokens)
            self.doc_metadata.append(doc)
        
        # Build BM25 index
        if self.documents:
            self.bm25 = BM25Okapi(self.documents)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search"""
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                doc = self.doc_metadata[idx].copy()
                doc['bm25_score'] = float(scores[idx])
                results.append(doc)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens