from typing import List, Dict, Any, Tuple
import requests
import json

class CrossEncoderReranker:
    def __init__(self):
        # Mock cross-encoder for reranking
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder scoring"""
        
        # Simple reranking based on query-document similarity
        scored_docs = []
        
        for doc in documents:
            score = self._calculate_relevance_score(query, doc['text'])
            doc['rerank_score'] = score
            scored_docs.append(doc)
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return scored_docs[:top_k]
    
    def _calculate_relevance_score(self, query: str, text: str) -> float:
        """Enhanced relevance scoring with multiple factors"""
        query_lower = query.lower()
        text_lower = text.lower()
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        
        # 1. Exact phrase matching (highest weight)
        phrase_score = 1.0 if query_lower in text_lower else 0.0
        
        # 2. Word overlap (Jaccard similarity)
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        jaccard = intersection / union if union > 0 else 0
        
        # 3. Partial word matching
        partial_matches = 0
        for q_word in query_words:
            for t_word in text_words:
                if q_word in t_word or t_word in q_word:
                    partial_matches += 1
                    break
        partial_score = partial_matches / len(query_words) if query_words else 0
        
        # 4. Position bonus (earlier mentions are better)
        position_score = 0.0
        first_match_pos = text_lower.find(query_lower)
        if first_match_pos >= 0:
            position_score = max(0, 1 - (first_match_pos / len(text_lower)))
        
        # 5. Length bonus (prefer detailed content)
        length_bonus = min(1.0, len(text) / 800)
        
        # Weighted combination
        final_score = (
            phrase_score * 0.4 +
            jaccard * 0.25 +
            partial_score * 0.15 +
            position_score * 0.1 +
            length_bonus * 0.1
        )
        
        return min(1.0, final_score)