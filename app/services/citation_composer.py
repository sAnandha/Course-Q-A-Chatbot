import re
from typing import List, Dict, Any
from app.models.schemas import Citation

class CitationComposer:
    def __init__(self):
        pass
    
    def compose_answer_with_citations(self, answer: str, sources: List[Dict[str, Any]]) -> tuple[str, List[Citation]]:
        """Compose answer with inline citations and return citation objects"""
        
        # Create citation mapping
        citations = []
        citation_map = {}
        
        for i, source in enumerate(sources):
            source_id = f"S{i+1}"
            page_info = f":pp{source.get('page_number', '')}" if source.get('page_number') else ""
            citation_key = f"[{source_id}{page_info}]"
            
            citation_map[source_id] = citation_key
            
            # Create citation object
            citation = Citation(
                source_id=source.get('chunk_id', source_id),
                span=source.get('text', '')[:200] + "...",
                confidence=source.get('score', 0.8),
                page_number=source.get('page_number')
            )
            citations.append(citation)
        
        # Insert citations into answer
        enhanced_answer = self._insert_citations(answer, sources, citation_map)
        
        return enhanced_answer, citations
    
    def _insert_citations(self, answer: str, sources: List[Dict[str, Any]], citation_map: Dict[str, str]) -> str:
        """Insert citation markers into answer text"""
        
        # Split answer into sentences
        sentences = re.split(r'(?<=[.!?])\\s+', answer)
        enhanced_sentences = []
        
        for sentence in sentences:
            # Find best matching source for this sentence
            best_source_idx = self._find_best_source_for_sentence(sentence, sources)
            
            if best_source_idx is not None:
                source_id = f"S{best_source_idx + 1}"
                citation_key = citation_map.get(source_id, f"[{source_id}]")
                sentence = sentence.rstrip() + f" {citation_key}"
            
            enhanced_sentences.append(sentence)
        
        return " ".join(enhanced_sentences)
    
    def _find_best_source_for_sentence(self, sentence: str, sources: List[Dict[str, Any]]) -> int:
        """Find the best matching source for a sentence"""
        
        sentence_words = set(sentence.lower().split())
        best_score = 0
        best_idx = None
        
        for i, source in enumerate(sources):
            source_words = set(source.get('text', '').lower().split())
            
            # Calculate overlap
            overlap = len(sentence_words.intersection(source_words))
            score = overlap / len(sentence_words) if sentence_words else 0
            
            if score > best_score and score > 0.2:  # Minimum threshold
                best_score = score
                best_idx = i
        
        return best_idx