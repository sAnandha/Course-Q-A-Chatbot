from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class Language(str, Enum):
    EN = "en"
    HI = "hi"

class QueryRequest(BaseModel):
    query: str
    lang: Optional[Language] = Language.EN
    top_k: Optional[int] = 5

class Citation(BaseModel):
    source_id: str
    span: str
    confidence: float
    page_number: Optional[int] = None
    document_name: Optional[str] = None

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    usage: dict
    latency_ms: int

class FeedbackRequest(BaseModel):
    query: str
    answer_id: str
    label: str
    note: Optional[str] = None

class DocumentChunk(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    embedding: List[float]
    metadata: dict
    page_number: Optional[int] = None
    document_name: Optional[str] = None