from typing import Dict, Any
import requests
import json

class TranslationService:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi'
        }
        
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English for retrieval"""
        if source_lang == 'en':
            return text
            
        # Simple translation mapping for common Hindi terms
        hindi_to_english = {
            'डेटा साइंस': 'data science',
            'मशीन लर्निंग': 'machine learning',
            'कृत्रिम बुद्धिमत्ता': 'artificial intelligence',
            'डेटा': 'data',
            'विश्लेषण': 'analysis',
            'एल्गोरिदम': 'algorithm',
            'न्यूरल नेटवर्क': 'neural network',
            'डीप लर्निंग': 'deep learning',
            'पायथन': 'python',
            'प्रोग्रामिंग': 'programming',
            'क्या है': 'what is',
            'कैसे काम करता है': 'how does it work',
            'फायदे': 'benefits',
            'उपयोग': 'applications'
        }
        
        # Translate common terms
        translated_text = text
        for hindi_term, english_term in hindi_to_english.items():
            translated_text = translated_text.replace(hindi_term, english_term)
        
        return translated_text
    
    def translate_to_target_language(self, text: str, target_lang: str) -> str:
        """Translate English response to target language"""
        if target_lang == 'en':
            return text
            
        # Simple translation for Hindi responses
        english_to_hindi = {
            'Data Science': 'डेटा साइंस',
            'Machine Learning': 'मशीन लर्निंग',
            'Artificial Intelligence': 'कृत्रिम बुद्धिमत्ता',
            'Data': 'डेटा',
            'Analysis': 'विश्लेषण',
            'Algorithm': 'एल्गोरिदम',
            'Neural Network': 'न्यूरल नेटवर्क',
            'Deep Learning': 'डीप लर्निंग',
            'Python': 'पायथन',
            'Programming': 'प्रोग्रामिंग',
            'Benefits': 'फायदे',
            'Applications': 'उपयोग',
            'Key Points': 'मुख्य बिंदु',
            'Details': 'विवरण',
            'Summary': 'सारांश'
        }
        
        # Translate key terms while preserving structure
        translated_text = text
        for english_term, hindi_term in english_to_hindi.items():
            translated_text = translated_text.replace(english_term, hindi_term)
        
        return translated_text
    
    def get_language_prompt(self, target_lang: str) -> str:
        """Get language-specific prompt instructions"""
        if target_lang == 'hi':
            return """
अतिरिक्त निर्देश:
- मुख्य तकनीकी शब्दों को अंग्रेजी में रखें लेकिन हिंदी में समझाएं
- उदाहरण: "Data Science (डेटा साइंस) एक बहुविषयक क्षेत्र है"
- संरचना को बनाए रखें लेकिन हिंदी में उत्तर दें
"""
        return ""