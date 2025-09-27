from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import time
from typing import Dict, Any
from app.models.schemas import QueryRequest, AnswerResponse, FeedbackRequest, Citation
# from app.services.rag_service import RAGService  # Removed - using hybrid retriever
from app.services.langchain_rag import LangChainRAGService
from app.services.cross_encoder import CrossEncoderReranker
from app.services.citation_composer import CitationComposer
from app.services.document_processor import DocumentProcessor
from app.services.hybrid_retriever import HybridRetriever
from app.services.metrics import metrics_collector
from app.services.session_manager import session_manager
# Safety filter removed - using built-in validation
from langchain_core.documents import Document
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
from fastapi.responses import FileResponse
import tempfile
import os
from datetime import datetime

app = FastAPI(title="Course Q&A Chatbot", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services with hybrid retrieval
hybrid_retriever = HybridRetriever(use_pinecone=True)  # Using your Pinecone index
langchain_rag = LangChainRAGService()  # LangChain-based RAG
cross_encoder = CrossEncoderReranker()  # Cross-encoder reranker
citation_composer = CitationComposer()  # Citation composer
doc_processor = DocumentProcessor()

# @app.on_event("startup")
# async def startup_event():
#     """Initialize OpenSearch index on startup"""
#     retriever.create_index()

@app.post("/answer", response_model=AnswerResponse)
async def get_answer(request: QueryRequest, x_session_id: str = Header(None)):
    """Get answer for a query with citations"""
    import uuid
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Basic input validation
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get session documents
        if x_session_id:
            session_docs = session_manager.get_session_documents(x_session_id)
            # Temporarily set session documents for this query
            original_docs = langchain_rag.documents
            langchain_rag.documents = session_docs
        
        # Generate answer using enhanced RAG with reranking
        print(f"ℹ️ QUERY PROCESSING: '{request.query}' (lang: {request.lang.value})")
        response = langchain_rag.generate_answer(request)
        print(f"✓ RAG RESPONSE: {len(response.answer)} chars, {len(response.citations)} citations")
        
        # Restore original documents
        if x_session_id:
            langchain_rag.documents = original_docs
        
        # Debug: Log response data
        print(f"ℹ️ Generated answer preview: {response.answer[:100]}...")
        print(f"ℹ️ Citations count: {len(response.citations)}")
        for i, citation in enumerate(response.citations[:3]):
            print(f"  Citation {i+1}: {citation.source_id} (confidence: {citation.confidence:.3f})")
        
        # Apply cross-encoder reranking if citations exist
        if response.citations:
            print(f"Applying reranking to {len(response.citations)} citations")
            
            # Convert citations to documents for reranking
            docs_for_rerank = [{
                'text': citation.span,
                'score': citation.confidence,
                'chunk_id': citation.source_id,
                'page_number': citation.page_number
            } for citation in response.citations]
            
            # Rerank documents
            reranked_docs = cross_encoder.rerank(request.query, docs_for_rerank, len(docs_for_rerank))
            
            # Update citations with reranked data
            response.citations = [Citation(
                source_id=doc['chunk_id'],
                span=doc['text'],
                confidence=doc.get('rerank_score', doc['score']),
                page_number=doc['page_number'],
                document_name=doc.get('document_name') or doc.get('metadata', {}).get('document_name')
            ) for doc in reranked_docs]
            
            print(f"✓ RERANKING COMPLETE: {len(response.citations)} final citations")
        else:
            print("⚠️ No citations found in response")
        
        # Response is already processed
        
        # Log metrics
        metrics_collector.log_query(
            trace_id, request.query, request.lang.value,
            response.latency_ms, response.usage.get("tokens", 0),
            len(response.citations), "success"
        )
        
        # Track Q&A in session with Unicode preservation
        if x_session_id:
            # Ensure Unicode text is preserved
            clean_query = request.query if isinstance(request.query, str) else str(request.query)
            clean_answer = response.answer if isinstance(response.answer, str) else str(response.answer)
            
            print(f"🔍 DEBUG - Before storing in session:")
            print(f"  Query type: {type(clean_query)}")
            print(f"  Answer type: {type(clean_answer)}")
            
            session_manager.add_qa_to_session(x_session_id, clean_query, clean_answer)
        
        print(f"✓ API RESPONSE READY: {len(response.answer)} chars, {len(response.citations)} citations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.log_query(
            trace_id, request.query, request.lang.value,
            int((time.time() - start_time) * 1000), 0, 0, "error", str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/source/{source_id}")
async def get_source(source_id: str, x_session_id: str = Header(None)) -> Dict[str, Any]:
    """Get original chunk and document metadata"""
    try:
        # Parse source_id format: S1:filename.pdf:pp3 -> extract chunk index
        import re
        match = re.match(r'S(\d+):([^:]+):pp(\d+)', source_id)
        if match:
            chunk_index = int(match.group(1)) - 1  # Convert to 0-based index
            doc_name = match.group(2)
            page_num = int(match.group(3))
        else:
            # Fallback for old format S1:pp3
            old_match = re.match(r'S(\d+):pp(\d+)', source_id)
            if old_match:
                chunk_index = int(old_match.group(1)) - 1
                page_num = int(old_match.group(2))
                doc_name = 'Unknown'
            else:
                chunk_index = 0
                page_num = 1
                doc_name = 'Unknown'
        
        # Get documents from session or global
        if x_session_id:
            documents = session_manager.get_session_documents(x_session_id)
        else:
            documents = langchain_rag.documents
        
        # Get document by index
        if 0 <= chunk_index < len(documents):
            doc = documents[chunk_index]
            return {
                "chunk_id": source_id,
                "text": doc.page_content,
                "metadata": doc.metadata,
                "page_number": page_num,
                "document_name": doc_name or doc.metadata.get('document_name', 'Unknown')
            }
        
        # Fallback: search by chunk_id in metadata
        for doc in documents:
            if doc.metadata.get('chunk_id') == source_id:
                return {
                    "chunk_id": source_id,
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
        
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")
    except Exception as e:
        print(f"Error finding source {source_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Source {source_id} not found")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for an answer"""
    # Store feedback (implement based on your needs)
    return {"status": "feedback received", "feedback_id": "fb_123"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), x_session_id: str = Header(None)):
    """Upload and process a document"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process document based on file type - sanitize for Pinecone
        from app.services.document_processor import sanitize_id
        doc_id = sanitize_id(file.filename.replace(".", "_"))
        print(f"✓ UPLOAD START: {file.filename} ({file.content_type}, {len(content)} bytes)")
        
        try:
            if file.filename.endswith('.pdf'):
                print(f"✓ Processing PDF: {file.filename}")
                chunks = doc_processor.process_pdf(tmp_file_path, doc_id, file.filename)
            elif file.filename.endswith('.md'):
                print(f"✓ Processing Markdown: {file.filename}")
                chunks = doc_processor.process_markdown(tmp_file_path, doc_id, file.filename)
            elif file.filename.endswith('.csv'):
                print(f"✓ Processing CSV: {file.filename}")
                chunks = doc_processor.process_csv(tmp_file_path, doc_id, file.filename)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            
            print(f"✓ PROCESSING COMPLETE: {len(chunks)} chunks created from {file.filename}")
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
        
        # Index chunks in LangChain
        langchain_docs = []
        for chunk in chunks:
            # Add to LangChain documents
            doc = Document(
                page_content=chunk.text,
                metadata={
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'page_number': chunk.page_number,
                    'document_name': chunk.document_name,
                    **chunk.metadata
                }
            )
            langchain_docs.append(doc)
        
        # Always add to LangChain service for search functionality
        langchain_rag.add_documents(langchain_docs)
        
        # Also add to session if session ID provided
        if x_session_id:
            session_manager.add_documents_to_session(x_session_id, langchain_docs)
        
        # Also add to hybrid retriever
        hybrid_docs = [{
            'text': chunk.text,
            'chunk_id': chunk.chunk_id,
            'doc_id': chunk.doc_id,
            'page_number': chunk.page_number,
            'document_name': chunk.document_name,
            'embedding': chunk.embedding
        } for chunk in chunks]
        
        hybrid_retriever.add_documents(hybrid_docs)
        
        # Debug: Check document counts
        print(f"✓ INDEXING COMPLETE:")
        print(f"  • Hybrid retriever: {len(chunks)} chunks")
        print(f"  • LangChain docs: {len(langchain_docs)} added")
        print(f"  • Total LangChain docs: {len(langchain_rag.documents)}")
        if chunks:
            print(f"  • Sample chunk: {chunks[0].text[:100]}...")
        if x_session_id:
            session_docs = session_manager.get_session_documents(x_session_id)
            print(f"  • Session {x_session_id[:8]}: {len(session_docs)} docs")
        
        # Clean up
        os.unlink(tmp_file_path)
        
        print(f"✓ UPLOAD SUCCESS: {file.filename} → {len(chunks)} chunks indexed")
        return {"status": "success", "chunks_processed": len(chunks), "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    print("✓ Health check requested")
    return {
        "status": "healthy", 
        "documents_loaded": len(langchain_rag.documents),
        "services": {
            "bedrock": langchain_rag.bedrock_available,
            "pinecone": True,
            "embeddings": "multilingual-MiniLM"
        }
    }

@app.get("/debug/documents")
async def debug_documents(x_session_id: str = Header(None)):
    """Check uploaded documents for session or global"""
    if x_session_id:
        session_docs = session_manager.get_session_documents(x_session_id)
        # Group by doc_id to show original documents
        doc_groups = {}
        for doc in session_docs:
            doc_id = doc.metadata.get("doc_id", "unknown")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    "doc_id": doc_id,
                    "text_preview": doc.page_content[:200] + "...",
                    "chunk_count": 0
                }
            doc_groups[doc_id]["chunk_count"] += 1
        
        return {
            "session_id": x_session_id,
            "total_documents": len(doc_groups),
            "documents": list(doc_groups.values())
        }
    else:
        # Group by doc_id to show original documents
        doc_groups = {}
        for doc in langchain_rag.documents:
            doc_id = doc.metadata.get("doc_id", "unknown")
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {
                    "doc_id": doc_id,
                    "text_preview": doc.page_content[:200] + "...",
                    "chunk_count": 0
                }
            doc_groups[doc_id]["chunk_count"] += 1
        
        return {
            "session_id": "global",
            "total_documents": len(doc_groups),
            "documents": list(doc_groups.values())
        }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics and performance stats"""
    return metrics_collector.get_metrics_summary()

@app.post("/session/create")
async def create_session():
    """Create new session"""
    session_id = session_manager.create_session()
    print(f"Created new session: {session_id}")
    return {"session_id": session_id}

@app.get("/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get session information and uploaded sources"""
    return session_manager.get_session_info(session_id)

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session and clear all documents"""
    try:
        # Clear session documents
        if session_id in session_manager.sessions:
            del session_manager.sessions[session_id]
        
        # Clear all documents from LangChain service
        langchain_rag.documents = []
        
        # Clear hybrid retriever documents
        hybrid_retriever.documents = []
        
        # Clear vector store if using local storage
        if hasattr(hybrid_retriever.vector_store, 'vectors'):
            hybrid_retriever.vector_store.vectors = {}
            hybrid_retriever.vector_store.metadata = {}
        
        # Clear BM25 search documents
        if hasattr(hybrid_retriever.bm25_search, 'documents'):
            hybrid_retriever.bm25_search.documents = []
        
        print(f"Session {session_id} cleared - all documents removed")
        return {"status": "session deleted", "documents_cleared": True}
    except Exception as e:
        print(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear session: {str(e)}")

def convert_to_hinglish(text):
    """Convert Hindi text to proper English"""
    hindi_to_english = {
        # Basic particles and conjunctions
        'का': 'of', 'की': 'of', 'के': 'of', 'को': 'to', 'में': 'in', 'से': 'from', 'पर': 'on', 'ने': '',
        'और': 'and', 'या': 'or', 'तथा': 'and', 'एवं': 'and', 'व': 'and', 'अथवा': 'or',
        
        # Verbs
        'है': 'is', 'हैं': 'are', 'था': 'was', 'थी': 'was', 'थे': 'were', 'होना': 'to be',
        'होगा': 'will be', 'होगी': 'will be', 'होंगे': 'will be', 'करना': 'to do', 'करने': 'to do', 'किया': 'did',
        'करता': 'does', 'करती': 'does', 'करते': 'do', 'जाता': 'goes', 'जाती': 'goes', 'जाते': 'go',
        'आता': 'comes', 'आती': 'comes', 'आते': 'come', 'देना': 'to give', 'देने': 'to give', 'दिया': 'gave',
        'लेना': 'to take', 'लेने': 'to take', 'लिया': 'took', 'बनाना': 'to make', 'बनाने': 'to make', 'बनाया': 'made',
        
        # Common words
        'यह': 'this', 'वह': 'that', 'ये': 'these', 'वे': 'those', 'इस': 'this', 'उस': 'that', 'इन': 'these', 'उन': 'those',
        'कि': 'that', 'जो': 'who', 'जिस': 'which', 'जिन': 'which', 'कोई': 'any', 'कुछ': 'some', 'सब': 'all', 'सभी': 'all',
        'एक': 'one', 'दो': 'two', 'तीन': 'three', 'चार': 'four', 'पांच': 'five', 'अधिक': 'more', 'कम': 'less',
        'बड़ा': 'big', 'छोटा': 'small', 'अच्छा': 'good', 'बुरा': 'bad', 'नया': 'new', 'पुराना': 'old',
        
        # Numbers and age
        'उम्र': 'age', 'वर्ष': 'years', 'साल': 'years', 'महीना': 'month', 'दिन': 'day', 'समय': 'time',
        '२१': '21', '२३': '23', '२५': '25', '३०': '30', '३५': '35', '४०': '40', '४५': '45', '५०': '50',
        
        # Names (common examples)
        'जूलिया': 'Julia', 'बॉब': 'Bob', 'राम': 'Ram', 'श्याम': 'Shyam', 'गीता': 'Geeta', 'सीता': 'Sita',
        
        # Technical terms
        'उपयोग': 'use', 'प्रयोग': 'experiment', 'इस्तेमाल': 'usage', 'विकास': 'development', 'निर्माण': 'construction',
        'प्रणाली': 'system', 'सिस्टम': 'system', 'तकनीक': 'technology', 'विधि': 'method', 'पद्धति': 'methodology',
        'आवश्यकता': 'requirement', 'आवश्यकताओं': 'requirements', 'जरूरत': 'need', 'आवश्यक': 'necessary',
        'परीक्षण': 'testing', 'टेस्ट': 'test', 'जांच': 'examination', 'मूल्यांकन': 'evaluation',
        'विश्लेषण': 'analysis', 'अध्ययन': 'study', 'समीक्षा': 'review', 'अनुसंधान': 'research',
        'प्रबंधन': 'management', 'व्यवस्थापन': 'administration', 'संचालन': 'operation', 'नियंत्रण': 'control',
        'डेटा': 'data', 'आंकड़े': 'statistics', 'जानकारी': 'information', 'सूचना': 'information', 'तथ्य': 'facts',
        'दस्तावेज': 'document', 'फाइल': 'file', 'रिपोर्ट': 'report', 'प्रतिवेदन': 'report',
        
        # Project management terms
        'परियोजना': 'project', 'प्रोजेक्ट': 'project', 'योजना': 'plan', 'कार्य': 'work', 'काम': 'work',
        'गतिविधि': 'activity', 'प्रक्रिया': 'process', 'चरण': 'phase', 'स्तर': 'level', 'भाग': 'part',
        'हिस्सा': 'part', 'अंश': 'portion', 'टुकड़ा': 'piece', 'खंड': 'section', 'विभाग': 'department',
        
        # Quality and standards
        'गुणवत्ता': 'quality', 'मानक': 'standard', 'स्तर': 'level', 'श्रेणी': 'category', 'दर्जा': 'grade',
        'सुनिश्चित': 'ensure', 'निश्चित': 'certain', 'पक्का': 'sure', 'तय': 'fixed', 'स्थिर': 'stable',
        'आश्वासन': 'assurance', 'गारंटी': 'guarantee', 'भरोसा': 'trust', 'विश्वास': 'confidence',
        
        # Common phrases
        'के लिए': 'for', 'के साथ': 'with', 'के बाद': 'after', 'के पहले': 'before',
        'के अनुसार': 'according to', 'के द्वारा': 'by', 'के बिना': 'without', 'के अलावा': 'besides',
        'इसके अलावा': 'besides this', 'इसके साथ': 'with this', 'इसके बाद': 'after this',
        
        # Negations and questions
        'नहीं': 'not', 'न': 'no', 'मत': 'don\'t', 'कभी नहीं': 'never', 'बिल्कुल नहीं': 'not at all',
        'क्या': 'what', 'कैसे': 'how', 'कब': 'when', 'कहां': 'where', 'क्यों': 'why', 'कौन': 'who',
        
        # Specific technical terms from the context
        'ट्रैक': 'track', 'ट्रेसेबिलिटी': 'traceability', 'मैट्रिक्स': 'matrix', 'केसों': 'cases',
        'वेयरहाउस': 'warehouse', 'केंद्रीकृत': 'centralized', 'भंडारण': 'storage',
        'विभिन्न': 'various', 'स्रोतों': 'sources', 'एकत्रित': 'collected', 'संग्रहीत': 'stored',
        'व्यवस्थित': 'organized', 'निर्णय': 'decision', 'व्यावसायिक': 'business',
        'बुद्धिमत्ता': 'intelligence', 'रिपोर्टिंग': 'reporting', 'समर्थन': 'support', 'प्रदान': 'provide'
    }
    
    # Replace Hindi words with English equivalents (longest first to avoid partial matches)
    sorted_items = sorted(hindi_to_english.items(), key=lambda x: len(x[0]), reverse=True)
    for hindi, english in sorted_items:
        text = text.replace(hindi, english)
    
    # Clean up any remaining Devanagari characters with placeholder
    import re
    text = re.sub(r'[\u0900-\u097F]+', '[Hindi text]', text)
    
    # Clean up extra spaces and grammar
    text = ' '.join(text.split())  # Remove extra whitespace
    text = text.replace(' of of ', ' of ')  # Fix duplicate prepositions
    text = text.replace(' is is ', ' is ')  # Fix duplicate verbs
    
    return text

@app.get("/session/{session_id}/export")
async def export_session_qa(session_id: str):
    """Export session Q&A as PDF (English questions only)"""
    try:
        # Get Q&A history
        print(f"🔍 DEBUG - PDF Export for session: {session_id}")
        qa_pairs = session_manager.get_session_qa_history(session_id)
        print(f"🔍 DEBUG - Got {len(qa_pairs)} Q&A pairs for PDF")
        
        # Filter for English questions only
        import re
        english_qa_pairs = []
        for qa in qa_pairs:
            question = qa.get('question', '')
            # Check if question contains Hindi characters
            has_hindi = bool(re.search(r'[\u0900-\u097F]', question))
            if not has_hindi:
                english_qa_pairs.append(qa)
        
        print(f"🔍 DEBUG - Filtered to {len(english_qa_pairs)} English Q&A pairs")
        
        # If no English questions, return error message
        if not english_qa_pairs:
            raise HTTPException(
                status_code=400, 
                detail="PDF export is only available for English questions. Hindi PDF conversion is under development."
            )
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_path = tmp_file.name
        
        # Use FPDF for proper Unicode support
        if FPDF_AVAILABLE:
            try:
                class UnicodePDF(FPDF):
                    def header(self):
                        self.set_font('Arial', 'B', 15)
                        self.cell(0, 10, 'Course Q&A Session Report', 0, 1, 'C')
                        self.ln(10)
                    
                    def __init__(self):
                        super().__init__()
                        self.set_margins(20, 20, 20)  # Set proper margins
                        self.add_page()
                        # Load Noto Sans Devanagari font
                        try:
                            current_dir = os.path.dirname(os.path.abspath(__file__))
                            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
                            font_path = os.path.join(project_root, 'static', 'NotoSansDevanagari-Regular.ttf')
                            
                            if os.path.exists(font_path):
                                self.add_font('NotoSans', '', font_path, uni=True)
                                self.unicode_font = 'NotoSans'
                            else:
                                self.unicode_font = 'Helvetica'
                        except:
                            self.unicode_font = 'Helvetica'
                    
                    def header(self):
                        self.set_font('Helvetica', 'B', 15)
                        self.cell(0, 10, 'Course Q&A Session Report', 0, 1, 'C')
                        self.ln(10)
                    
                    def add_unicode_text(self, text, size=12, style=''):
                        self.set_font(self.unicode_font, style, size)
                        
                        lines = text.split('\n')
                        for line in lines:
                            if line.strip():
                                if self.unicode_font == 'NotoSans':
                                    self.multi_cell(0, 10, line)
                                    self.ln(2)
                                else:
                                    safe_line = line.encode('ascii', 'replace').decode('ascii')
                                    self.multi_cell(0, 10, safe_line)
                                    self.ln(2)
                            else:
                                self.ln(5)
                
                pdf = UnicodePDF()
                pdf.add_page()
                
                # Add session info
                pdf.add_unicode_text(f"Session: {session_id[:8]}", 10)
                pdf.add_unicode_text(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 10)
                pdf.ln(10)
                
                # Add Q&A content
                if english_qa_pairs:
                    pdf.add_unicode_text("Questions & Answers (English Only):", 14, 'B')
                    pdf.ln(5)
                    
                    for i, qa in enumerate(english_qa_pairs, 1):
                        # Keep original formatting for English questions
                        import re
                        has_hindi_q = bool(re.search(r'[\u0900-\u097F]', qa['question']))
                        question_text = convert_to_hinglish(qa['question']) if has_hindi_q else qa['question']
                        pdf.add_unicode_text(f"Q{i}: {question_text}", 12, 'B')
                        pdf.ln(3)
                        
                        # Keep UI formatting for English answers
                        has_hindi_a = bool(re.search(r'[\u0900-\u097F]', qa['answer']))
                        answer_text = convert_to_hinglish(qa['answer']) if has_hindi_a else qa['answer']
                        answer_text = answer_text.replace('**', '').replace('##', '')
                        pdf.add_unicode_text(answer_text, 10)
                        pdf.ln(8)
                else:
                    pdf.add_unicode_text("No questions found in this session.", 12)
                
                # Save PDF
                pdf.output(pdf_path)
                
                return FileResponse(
                    path=pdf_path,
                    filename=f"session_{session_id[:8]}_qa.pdf",
                    media_type="application/pdf"
                )
                
            except Exception as e:
                print(f"FPDF Unicode generation failed: {e}")
        
        # Use ReportLab Canvas with Hindi font (like hindi_to_pdf.py)
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        
        # Get font path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        font_path = os.path.join(project_root, 'static', 'NotoSansDevanagari-Regular.ttf')
        
        try:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('NotoDeva', font_path))
                
                # Create PDF with proper formatting using canvas
                c = canvas.Canvas(pdf_path, pagesize=A4)
                page_width, page_height = A4
                margin = 40
                leading = 14
                
                # Start first page
                y = page_height - 60
                
                # Title
                c.setFont('Helvetica-Bold', 16)
                c.drawString(margin, y, 'Course Q&A Session Report')
                y -= 30
                
                # Session info
                c.setFont('Helvetica', 10)
                c.drawString(margin, y, f'Session: {session_id[:8]} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
                y -= 40
                
                # Q&A content (English only)
                if english_qa_pairs:
                    c.setFont('Helvetica-Bold', 12)
                    c.drawString(margin, y, 'Questions & Answers (English Only):')
                    y -= 30
                    
                    for i, qa in enumerate(english_qa_pairs, 1):
                        # Check if we need a new page
                        if y < 150:
                            c.showPage()
                            y = page_height - 60
                        # Question
                        c.setFont('Helvetica-Bold', 11)
                        import re
                        has_hindi_q = bool(re.search(r'[\u0900-\u097F]', qa['question']))
                        question_display = convert_to_hinglish(qa['question']) if has_hindi_q else qa['question']
                        question_text = f"Q{i}: {question_display}"
                        
                        c.drawString(margin, y, question_text)
                        y -= 20
                        
                        # Answer
                        has_hindi_a = bool(re.search(r'[\u0900-\u097F]', qa['answer']))
                        answer_text = convert_to_hinglish(qa['answer']) if has_hindi_a else qa['answer']
                        answer_text = answer_text.replace('**', '').replace('##', '')
                        
                        c.setFont('Helvetica', 10)
                        lines = answer_text.split('\n')
                        
                        for line in lines:
                            if line.strip():
                                # Simple word wrapping
                                words = line.split()
                                current_line = ""
                                for word in words:
                                    test_line = (current_line + " " + word).strip()
                                    if c.stringWidth(test_line, 'Helvetica', 10) <= page_width - 2*margin:
                                        current_line = test_line
                                    else:
                                        if current_line:
                                            c.drawString(margin + 20, y, current_line)
                                            y -= 12
                                            if y < 60:
                                                c.showPage()
                                                y = page_height - 60
                                        current_line = word
                                if current_line:
                                    c.drawString(margin + 20, y, current_line)
                                    y -= 12
                                    if y < 60:
                                        c.showPage()
                                        y = page_height - 60
                            else:
                                y -= 6
                        
                        y -= 20  # Space between Q&A pairs
                else:
                    c.setFont('NotoDeva', 12)
                    c.drawString(x, y, 'No questions found in this session.')
                
                c.save()
                
                return FileResponse(
                    path=pdf_path,
                    filename=f"session_{session_id[:8]}_qa.pdf",
                    media_type="application/pdf"
                )
            else:
                raise FileNotFoundError(f"Font not found: {font_path}")
                
        except Exception as e:
            print(f"Hindi PDF generation failed: {e}")
            # Fallback to simple text
            pass
        
        # Fallback: Simple ReportLab without Hindi
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        story.append(Paragraph("Course Q&A Session Report", styles['Title']))
        story.append(Paragraph(f"Session: {session_id[:8]}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        if english_qa_pairs:
            story.append(Paragraph("Questions & Answers (English Only):", styles['Heading2']))
            for i, qa in enumerate(english_qa_pairs, 1):
                # Keep original formatting for English content
                import re
                has_hindi_q = bool(re.search(r'[\u0900-\u097F]', qa['question']))
                question_text = convert_to_hinglish(qa['question']) if has_hindi_q else qa['question']
                
                has_hindi_a = bool(re.search(r'[\u0900-\u097F]', qa['answer']))
                answer_text = convert_to_hinglish(qa['answer']) if has_hindi_a else qa['answer']
                answer_text = answer_text.replace('**', '').replace('##', '')
                
                story.append(Paragraph(f"Q{i}: {question_text}", styles['Heading3']))
                story.append(Paragraph(answer_text, styles['Normal']))
                story.append(Spacer(1, 0.2*inch))
        else:
            story.append(Paragraph("No English questions found in this session.", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        # Return PDF file
        return FileResponse(
            path=pdf_path,
            filename=f"session_{session_id[:8]}_qa.pdf",
            media_type="application/pdf"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/metrics/export")
async def export_metrics():
    """Export metrics as CSV"""
    from fastapi.responses import FileResponse
    return FileResponse(
        path="metrics.csv",
        filename="chatbot_metrics.csv",
        media_type="text/csv"
    )

@app.get("/test-hindi-pdf")
async def test_hindi_pdf():
    """Test Hindi font in PDF generation"""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        
        # Create test PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf_path = tmp_file.name
        
        # Get font path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        font_path = os.path.join(project_root, 'static', 'NotoSansDevanagari-Regular.ttf')
        
        print(f"🔍 DEBUG - Font path: {font_path}")
        print(f"🔍 DEBUG - Font exists: {os.path.exists(font_path)}")
        
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('NotoDeva', font_path))
            
            c = canvas.Canvas(pdf_path, pagesize=A4)
            c.setFont('NotoDeva', 16)
            
            # Test Hindi text
            test_text = "बॉब की उम्र 21 वर्ष है"
            print(f"🔍 DEBUG - Test text: {repr(test_text)}")
            
            c.drawString(100, 750, "Test Hindi Text:")
            c.drawString(100, 720, test_text)
            c.drawString(100, 690, "Bob ki umra 21 varsh hai")
            
            c.save()
            
            return FileResponse(
                path=pdf_path,
                filename="hindi_test.pdf",
                media_type="application/pdf"
            )
        else:
            return {"error": "Font file not found", "path": font_path}
            
    except Exception as e:
        return {"error": str(e)}

@app.post("/add_sample_docs")
async def add_sample_documents(request: Dict[str, Any], x_session_id: str = Header(None)):
    """Add sample documents for testing"""
    try:
        documents = request.get("documents", [])
        langchain_docs = []
        
        for i, doc_data in enumerate(documents):
            doc = Document(
                page_content=doc_data["text"],
                metadata={
                    'chunk_id': f'sample_chunk_{i}',
                    'doc_id': f'sample_doc_{i}',
                    'page_number': doc_data["metadata"].get("page_number", 1),
                    'document_name': doc_data["metadata"].get("document_name", f"sample_{i}.pdf"),
                }
            )
            langchain_docs.append(doc)
        
        # Add to LangChain service
        langchain_rag.add_documents(langchain_docs)
        
        # Add to session if provided
        if x_session_id:
            session_manager.add_documents_to_session(x_session_id, langchain_docs)
        
        print(f"Added {len(langchain_docs)} sample documents")
        return {"status": "success", "documents_added": len(langchain_docs)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)