import re
from typing import List

class QueryEnhancer:
    def __init__(self):
        # Common synonyms and expansions for technical terms
        self.synonyms = {
            'ml': 'machine learning',
            'ai': 'artificial intelligence',
            'nn': 'neural network',
            'dl': 'deep learning',
            'nlp': 'natural language processing',
            'cv': 'computer vision'
        }
        
        # Question type patterns
        self.question_patterns = {
            'definition': ['what is', 'define', 'meaning of'],
            'process': ['how does', 'how to', 'process of'],
            'comparison': ['difference between', 'compare', 'vs'],
            'examples': ['examples of', 'types of', 'kinds of']
        }
    
    def enhance_query(self, query: str) -> List[str]:
        """Generate multiple enhanced versions of the query"""
        enhanced_queries = [query]  # Original query
        
        # Expand abbreviations
        expanded_query = self._expand_abbreviations(query)
        if expanded_query != query:
            enhanced_queries.append(expanded_query)
        
        # Add synonyms
        synonym_query = self._add_synonyms(query)
        if synonym_query != query:
            enhanced_queries.append(synonym_query)
        
        # Generate related questions
        related_queries = self._generate_related_queries(query)
        enhanced_queries.extend(related_queries)
        
        return enhanced_queries[:5]  # Limit to 5 variations
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand common abbreviations"""
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            if word in self.synonyms:
                expanded_words.append(self.synonyms[word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def _add_synonyms(self, query: str) -> str:
        """Add synonyms to key terms"""
        enhanced = query
        for abbrev, full_term in self.synonyms.items():
            if full_term in query.lower():
                enhanced += f" {abbrev}"
        return enhanced
    
    def _generate_related_queries(self, query: str) -> List[str]:
        """Generate related queries based on question type"""
        related = []
        query_lower = query.lower()
        
        # If asking "what is X", also search for "X definition", "X explanation"
        if any(pattern in query_lower for pattern in self.question_patterns['definition']):
            topic = self._extract_topic(query)
            if topic:
                related.extend([
                    f"{topic} definition",
                    f"{topic} explanation",
                    f"understanding {topic}"
                ])
        
        # If asking "how does X work", also search for "X process", "X mechanism"
        elif any(pattern in query_lower for pattern in self.question_patterns['process']):
            topic = self._extract_topic(query)
            if topic:
                related.extend([
                    f"{topic} process",
                    f"{topic} mechanism",
                    f"{topic} algorithm"
                ])
        
        return related[:3]  # Limit related queries
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query"""
        # Remove question words and extract key terms
        stop_words = ['what', 'is', 'how', 'does', 'work', 'the', 'a', 'an']
        words = [w for w in query.lower().split() if w not in stop_words]
        
        # Return the longest meaningful phrase
        if len(words) >= 2:
            return ' '.join(words[:2])
        elif len(words) == 1:
            return words[0]
        
        return ""