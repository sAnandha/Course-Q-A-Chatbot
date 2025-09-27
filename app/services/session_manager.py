import uuid
from typing import Dict, List, Any
from datetime import datetime, timedelta
from langchain_core.documents import Document

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)  # 2 hour timeout
    
    def create_session(self) -> str:
        """Create new session and return session ID"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            'documents': [],
            'created_at': datetime.now(),
            'last_accessed': datetime.now(),
            'upload_count': 0,
            'qa_history': []
        }
        return session_id
    
    def get_session_documents(self, session_id: str) -> List[Document]:
        """Get documents for a specific session"""
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        session['last_accessed'] = datetime.now()
        return session['documents']
    
    def add_documents_to_session(self, session_id: str, documents: List[Document]):
        """Add documents to a specific session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'documents': [],
                'created_at': datetime.now(),
                'last_accessed': datetime.now(),
                'upload_count': 0,
                'qa_history': []
            }
        
        self.sessions[session_id]['documents'].extend(documents)
        self.sessions[session_id]['last_accessed'] = datetime.now()
        self.sessions[session_id]['upload_count'] += 1
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session information"""
        if session_id not in self.sessions:
            return {'exists': False}
        
        session = self.sessions[session_id]
        return {
            'exists': True,
            'document_count': len(session['documents']),
            'upload_count': session['upload_count'],
            'created_at': session['created_at'].isoformat(),
            'last_accessed': session['last_accessed'].isoformat(),
            'documents': [
                {
                    'doc_id': doc.metadata.get('doc_id', 'unknown'),
                    'chunk_id': doc.metadata.get('chunk_id', 'unknown'),
                    'page_number': doc.metadata.get('page_number'),
                    'source_type': doc.metadata.get('source_type', 'unknown')
                }
                for doc in session['documents'][:10]  # Show first 10
            ]
        }
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session['last_accessed'] > self.session_timeout
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)
    
    def add_qa_to_session(self, session_id: str, question: str, answer: str):
        """Add Q&A pair to session history"""
        if session_id not in self.sessions:
            return
        
        # DEBUG: Print original text
        print(f"🔍 DEBUG - Original Question: {repr(question)}")
        print(f"🔍 DEBUG - Original Answer: {repr(answer[:100])}...")
        
        # Check if text contains Hindi characters
        import re
        has_hindi_q = bool(re.search(r'[\u0900-\u097F]', str(question)))
        has_hindi_a = bool(re.search(r'[\u0900-\u097F]', str(answer)))
        print(f"🔍 DEBUG - Question has Hindi: {has_hindi_q}")
        print(f"🔍 DEBUG - Answer has Hindi: {has_hindi_a}")
        
        # Store as-is without encoding/decoding
        clean_question = str(question) if question else ""
        clean_answer = str(answer) if answer else ""
        
        # DEBUG: Print stored text
        print(f"🔍 DEBUG - Stored Question: {repr(clean_question)}")
        print(f"🔍 DEBUG - Stored Answer: {repr(clean_answer[:100])}...")
        
        self.sessions[session_id]['qa_history'].append({
            'question': clean_question,
            'answer': clean_answer,
            'timestamp': datetime.now().isoformat()
        })
        self.sessions[session_id]['last_accessed'] = datetime.now()
    
    def get_session_qa_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get Q&A history for a session"""
        if session_id not in self.sessions:
            return []
        
        qa_history = self.sessions[session_id].get('qa_history', [])
        
        # DEBUG: Print retrieved text
        print(f"🔍 DEBUG - Retrieved {len(qa_history)} Q&A pairs")
        for i, qa in enumerate(qa_history[:2]):  # Show first 2
            print(f"🔍 DEBUG - Retrieved Q{i+1}: {repr(qa.get('question', ''))}")
            print(f"🔍 DEBUG - Retrieved A{i+1}: {repr(qa.get('answer', '')[:100])}...")
        
        return qa_history

# Global session manager instance
session_manager = SessionManager()