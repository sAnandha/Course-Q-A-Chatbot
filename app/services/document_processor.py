import PyPDF2
import csv
import re
import os
import base64
from typing import List, Dict, Any
from app.models.schemas import DocumentChunk
from app.services.local_llm import LocalLLMService
try:
    import fitz  # PyMuPDF for PDF to image conversion
    from PIL import Image
    import io
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Vision libraries not available. Install: pip install PyMuPDF Pillow")

def sanitize_id(text: str) -> str:
    """Sanitize text to be ASCII-only for Pinecone IDs"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text)

class DocumentProcessor:
    def __init__(self):
        self.llm_service = LocalLLMService()
    
    def chunk_by_semantic_boundaries(self, text: str, doc_id: str, page_num: int = 1, document_name: str = None, source_type: str = "pdf") -> List[DocumentChunk]:
        """Enhanced semantic chunking with content-type optimization"""
        chunks = []
        
        # Content-type specific chunking strategies
        if source_type == "csv":
            # CSV already chunked by rows, return as single chunk
            chunk_id = sanitize_id(f"{doc_id}_row_{page_num}")
            embedding = self.llm_service.get_embedding(text)
            chunk = DocumentChunk(
                doc_id=doc_id, chunk_id=chunk_id, text=text, embedding=embedding,
                metadata={"source_type": "csv", "document_name": document_name or doc_id},
                page_number=page_num, document_name=document_name or doc_id
            )
            return [chunk]
        
        # Enhanced semantic boundaries for PDF/Markdown
        patterns = [
            r'\n#{1,6}\s+[^\n]+',  # Markdown headers
            r'\n\d+\.\s+[^\n]+',   # Numbered lists
            r'\n[A-Z][^\n]*:',     # Section headers
            r'\n\s*\n\s*\n',       # Double line breaks
            r'\n\s*[-*+]\s+',      # Bullet points
        ]
        
        # Split by semantic boundaries with priority
        sections = [text]  # Start with full text
        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend([p.strip() for p in parts if p.strip()])
            sections = new_sections
        
        # Process sections with intelligent sizing
        for i, section in enumerate(sections):
            if len(section) < 50:  # Skip very short sections
                continue
            
            # Adaptive chunking based on content length
            if len(section) > 1500:  # Large sections
                words = section.split()
                chunk_size = 400
                overlap = 75
                
                for j in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[j:j + chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    
                    if len(chunk_text.strip()) < 100:
                        continue
                    
                    chunk_id = sanitize_id(f"{doc_id}_chunk_{i}_{j}")
                    embedding = self.llm_service.get_embedding(chunk_text)
                    
                    chunk = DocumentChunk(
                        doc_id=doc_id, chunk_id=chunk_id, text=chunk_text, embedding=embedding,
                        metadata={"source_type": source_type, "section_id": i, "chunk_type": "overlapping", "document_name": document_name or doc_id},
                        page_number=page_num, document_name=document_name or doc_id
                    )
                    chunks.append(chunk)
            else:
                # Optimal size sections - keep as single chunk
                chunk_id = sanitize_id(f"{doc_id}_chunk_{i}")
                embedding = self.llm_service.get_embedding(section)
                
                chunk = DocumentChunk(
                    doc_id=doc_id, chunk_id=chunk_id, text=section, embedding=embedding,
                    metadata={"source_type": source_type, "section_id": i, "chunk_type": "semantic", "document_name": document_name or doc_id},
                    page_number=page_num, document_name=document_name or doc_id
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_pdf(self, file_path: str, doc_id: str, document_name: str = None) -> List[DocumentChunk]:
        """Process PDF file with vision support for images and text"""
        chunks = []
        use_vision = False  # Disabled for faster processing
        
        print(f"Processing PDF: {file_path} (Vision: {use_vision})")
        
        try:
            # Try vision-based processing first
            if use_vision:
                chunks = self._process_pdf_with_vision(file_path, doc_id, document_name)
                if chunks:
                    print(f"✓ Vision processing successful: {len(chunks)} chunks")
                    return chunks
                else:
                    print("Vision processing failed, falling back to text extraction")
            
            # Fallback to text-only processing
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    print(f"Page {page_num + 1}: {len(text)} characters")
                    
                    # Skip pages with no meaningful text
                    if len(text.strip()) < 50:
                        print(f"⚠️ Page {page_num + 1}: Minimal text ({len(text.strip())} chars), skipping")
                        continue
                    
                    # Clean and validate text
                    text = text.strip()
                    if text.startswith('Page') and 'from document' in text and len(text) < 100:
                        print(f"⚠️ Page {page_num + 1}: Generic fallback text detected, skipping")
                        continue
                    
                    page_chunks = self.chunk_by_semantic_boundaries(text, doc_id, page_num + 1, document_name, "pdf")
                    
                    # Only create fallback if we have some actual content
                    if not page_chunks and len(text) > 100:
                        chunk_id = sanitize_id(f"{doc_id}_page_{page_num + 1}")
                        embedding = self.llm_service.get_embedding(text)
                        chunk = DocumentChunk(
                            doc_id=doc_id,
                            chunk_id=chunk_id,
                            text=text,
                            embedding=embedding,
                            metadata={'source_type': 'pdf', 'document_name': document_name or doc_id},
                            page_number=page_num + 1,
                            document_name=document_name or doc_id
                        )
                        page_chunks = [chunk]
                    
                    chunks.extend(page_chunks)
                    print(f"✓ PDF Page {page_num + 1}: Created {len(page_chunks)} chunks")
        
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            raise
        
        # Only create fallback if absolutely no content was extracted
        if not chunks:
            print(f"⚠️ No content extracted from PDF {doc_id}")
            # Try alternative PDF processing
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    all_text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 50:
                            all_text += page_text + "\n\n"
                    
                    if all_text.strip():
                        print(f"✓ Alternative extraction successful: {len(all_text)} chars")
                        alt_chunks = self.chunk_by_semantic_boundaries(all_text, doc_id, 1, document_name)
                        chunks.extend(alt_chunks)
            except ImportError:
                print("pdfplumber not available for alternative extraction")
            except Exception as e:
                print(f"Alternative extraction failed: {e}")
            
            # Final fallback only if still no content
            if not chunks:
                fallback_text = f"Unable to extract readable content from {document_name or doc_id}. Please ensure the PDF contains extractable text."
                embedding = self.llm_service.get_embedding(fallback_text)
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=sanitize_id(f"{doc_id}_fallback"),
                    text=fallback_text,
                    embedding=embedding,
                    metadata={'source_type': 'pdf', 'document_name': document_name or doc_id, 'extraction_failed': True},
                    page_number=1,
                    document_name=document_name or doc_id
                )
                chunks.append(chunk)
                print(f"⚠️ Created extraction failure notice for {doc_id}")
        
        return chunks
    
    def process_markdown(self, file_path: str, doc_id: str, document_name: str = None) -> List[DocumentChunk]:
        """Process Markdown file and extract chunks"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return self.chunk_by_semantic_boundaries(content, doc_id, 1, document_name, "markdown")
    
    def process_csv(self, file_path: str, doc_id: str, document_name: str = None) -> List[DocumentChunk]:
        """Process CSV file and extract chunks"""
        chunks = []
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            file_content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        file_content = file.read()
                    print(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if not file_content:
                raise Exception("Could not read CSV file with any encoding")
            
            # Parse CSV content
            import io
            csv_reader = csv.DictReader(io.StringIO(file_content))
            
            # Get column names
            fieldnames = csv_reader.fieldnames or []
            print(f"CSV columns: {fieldnames}")
            
            if not fieldnames:
                raise Exception("CSV file has no columns")
            
            row_count = 0
            for i, row in enumerate(csv_reader):
                row_count += 1
                
                # Try different column name combinations
                question = (row.get('question') or row.get('Question') or 
                           row.get('q') or row.get('Q') or '')
                answer = (row.get('answer') or row.get('Answer') or 
                         row.get('a') or row.get('A') or '')
                
                # If no question/answer columns, use all row data
                if not question and not answer:
                    # Create text from all non-empty values
                    values = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
                    text = "\n".join(values) if values else f"Row {i+1}: {str(row)}"
                else:
                    text = f"Q: {question}\nA: {answer}"
                
                # Skip empty rows
                if not text.strip() or text.strip() in ['Q: \nA: ', 'Q:\nA:', 'Row : {}']:
                    print(f"Skipping empty row {i+1}")
                    continue
                
                print(f"✓ Processing CSV row {i+1}: {text[:100]}...")
                
                # Use optimized CSV chunking
                csv_chunks = self.chunk_by_semantic_boundaries(text, doc_id, i+1, document_name, "csv")
                for chunk in csv_chunks:
                    chunk.metadata["row_number"] = i+1
                chunks.extend(csv_chunks)
            
            print(f"✓ CSV Processing Complete: {row_count} rows → {len(chunks)} chunks")
            
            if not chunks:
                # Create a fallback chunk if no data was processed
                fallback_text = f"CSV file {document_name or doc_id} with columns: {', '.join(fieldnames)}"
                embedding = self.llm_service.get_embedding(fallback_text)
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=sanitize_id(f"{doc_id}_fallback"),
                    text=fallback_text,
                    embedding=embedding,
                    metadata={"source_type": "csv", "document_name": document_name or doc_id},
                    page_number=1,
                    document_name=document_name or doc_id
                )
                chunks.append(chunk)
                print(f"✓ Created fallback chunk for empty CSV: {document_name or doc_id}")
            
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")
            # Create error chunk
            error_text = f"Error processing CSV file {document_name or doc_id}: {str(e)}"
            embedding = self.llm_service.get_embedding(error_text)
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=sanitize_id(f"{doc_id}_error"),
                text=error_text,
                embedding=embedding,
                metadata={"source_type": "csv", "error": True, "document_name": document_name or doc_id},
                page_number=1,
                document_name=document_name or doc_id
            )
            chunks.append(chunk)
        
        return chunks
    
    def _process_pdf_with_vision(self, file_path: str, doc_id: str, document_name: str = None) -> List[DocumentChunk]:
        """Process PDF using Claude 3 Sonnet vision capabilities"""
        if not VISION_AVAILABLE:
            return []
        
        chunks = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_doc = fitz.open(file_path)
            print(f"Vision processing PDF with {len(pdf_doc)} pages")
            
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                img_data = pix.tobytes("png")
                
                # Encode image to base64
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Use Claude 3 Sonnet vision to extract content
                vision_text = self._extract_with_claude_vision(img_base64, page_num + 1)
                
                if vision_text and len(vision_text.strip()) > 50:
                    # Create chunks from vision-extracted text
                    page_chunks = self.chunk_by_semantic_boundaries(
                        vision_text, doc_id, page_num + 1, document_name
                    )
                    
                    # Mark chunks as vision-processed
                    for chunk in page_chunks:
                        chunk.metadata['extraction_method'] = 'claude_vision'
                        chunk.metadata['has_images'] = True
                    
                    chunks.extend(page_chunks)
                    print(f"Vision extracted {len(page_chunks)} chunks from page {page_num + 1}")
                else:
                    print(f"Vision extraction failed for page {page_num + 1}")
            
            pdf_doc.close()
            
        except Exception as e:
            print(f"Vision processing error: {e}")
            return []
        
        return chunks
    
    def _extract_with_claude_vision(self, img_base64: str, page_num: int) -> str:
        """Extract text and describe images using Claude 3 Sonnet vision"""
        try:
            # Check if Bedrock LLM service is available
            if not hasattr(self.llm_service, 'bedrock') or not self.llm_service.use_bedrock:
                return ""
            
            import json
            
            # Claude 3 Sonnet vision prompt
            prompt = f"""Analyze this PDF page image and extract ALL content including:

1. **Text Content**: Extract all readable text exactly as it appears
2. **Visual Elements**: Describe charts, diagrams, tables, images
3. **Structure**: Note headings, bullet points, formatting
4. **Data**: Extract data from tables, charts, graphs

Page {page_num} Content:"""
            
            # Prepare message for Claude with image
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": img_base64
                                }
                            }
                        ]
                    }
                ]
            })
            
            # Call Bedrock with vision
            response = self.llm_service.bedrock.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=body
            )
            
            response_body = json.loads(response['body'].read())
            extracted_text = response_body['content'][0]['text']
            
            print(f"✓ Claude vision extracted {len(extracted_text)} characters from page {page_num}")
            return extracted_text
            
        except Exception as e:
            print(f"Claude vision extraction failed: {e}")
            return ""