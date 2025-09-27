import time
import uuid
import os
from typing import List, Dict, Any
from langchain_core.documents import Document
# Only import what we actually use
try:
    from langchain_aws import ChatBedrock
except ImportError:
    ChatBedrock = None
from app.models.schemas import QueryRequest, AnswerResponse, Citation, Language
from app.services.local_llm import LocalLLMService
from app.services.query_enhancer import QueryEnhancer
from app.services.translation_service import TranslationService

class LangChainRAGService:
    def __init__(self):
        self.llm_service = LocalLLMService()
        self.query_enhancer = QueryEnhancer()
        self.translation_service = TranslationService()
        
        # Try AWS Bedrock first, fallback to local embeddings
        use_bedrock = os.getenv('USE_BEDROCK', 'true').lower() == 'true'
        
        if use_bedrock and ChatBedrock:
            try:
                self.llm = ChatBedrock(
                    model_id=os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
                    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
                )
                print("✓ AWS Bedrock Claude 3 Sonnet INITIALIZED")
                # Test connection immediately
                from langchain_core.messages import HumanMessage
                test_msg = [HumanMessage(content="Say 'Bedrock connected successfully'")]
                test_response = self.llm.invoke(test_msg)
                print(f"✓ Bedrock test: {test_response.content[:50]}...")
                self.bedrock_available = True
            except Exception as e:
                print(f"Bedrock initialization failed: {e}")
                print("✓ Falling back to local LLM with MiniLM embeddings")
                self.llm = None
                self.bedrock_available = False
        else:
            print("✓ Using local LLM with MiniLM embeddings (Bedrock disabled)")
            self.llm = None
            self.bedrock_available = False
        
        # Initialize local embedding model as fallback
        try:
            from sentence_transformers import SentenceTransformer
            self.local_embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ Local MiniLM embeddings initialized as fallback")
        except Exception as e:
            print(f"Local embeddings failed: {e}")
            self.local_embedder = None
        
        # We use local embeddings only
        
        # Enhanced prompt for comprehensive, structured answers
        self.prompt_template = """You are a document-based Q&A assistant. Provide a comprehensive answer about the specific topic using ONLY the provided context.

IMPORTANT RULES:
1. Use ONLY information from the provided context
2. Use ONLY the exact source IDs provided in the context
3. Structure the answer with clear sections
4. Include ALL relevant information about the topic from the context
5. Add citations after each point using exact source IDs

REQUIRED FORMAT:
**[Topic Name]**

## Definition:
• Clear definition and explanation of what it is [source ID]
• Additional context or background [source ID]

## Key Concepts:
• Concept 1: Detailed explanation [source ID]
• Concept 2: Detailed explanation [source ID]
• Concept 3: Detailed explanation [source ID]

## Methods/Techniques:
• Method 1: How it works or is implemented [source ID]
• Method 2: Specific techniques or approaches [source ID]
• Method 3: Tools or processes involved [source ID]

## Applications/Examples:
• Application 1: Real-world use case [source ID]
• Application 2: Industry application [source ID]
• Application 3: Practical examples [source ID]

## Related Information:
• Additional relevant details [source ID]
• Important considerations [source ID]

Context:
{context}

Question: {question}

Provide a comprehensive, well-structured answer:"""
        
        # Store documents for retrieval
        self.documents = []
    
    def add_documents(self, docs: List[Document]):
        """Add documents to the knowledge base"""
        self.documents.extend(docs)
    
    def generate_answer(self, request: QueryRequest) -> AnswerResponse:
        """Generate answer using LangChain RAG pipeline"""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        
        # Handle greetings and simple conversational inputs
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if request.query.lower().strip() in greetings:
            return AnswerResponse(
                answer="Hello! I'm here to help you with questions about your uploaded documents. Please ask me anything about the content you've shared.",
                citations=[],
                usage={"tokens": 0, "trace_id": trace_id},
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        # Use the original query
        query = request.query
        
        # Translate query to English for retrieval if needed
        english_query = self.translation_service.translate_to_english(query, request.lang.value)
        print(f"Original query: {query}")
        print(f"English query: {english_query}")
        
        # Enhance query for better retrieval
        enhanced_queries = self.query_enhancer.enhance_query(english_query)
        print(f"Enhanced queries: {enhanced_queries}")
        
        # Detect query intent for better matching
        query_intent = self._detect_query_intent(english_query.lower())
        print(f"Query intent detected: {query_intent}")
        self._current_query_intent = query_intent
        
        # Optimized retrieval: First find relevant document, then search within it
        relevant_docs = self._smart_document_retrieval(query, request.top_k)
        
        if not relevant_docs:
            print(f"No relevant documents found for query: {query}")
            print(f"Total documents available: {len(self.documents)}")
            if self.documents:
                print(f"Sample document: {self.documents[0].page_content[:100]}...")
            
            return AnswerResponse(
                answer="Please ask questions directly related to the content of your uploaded files.",
                citations=[],
                usage={"tokens": 0, "trace_id": trace_id},
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        # Build context and generate answer
        context = self._build_context(relevant_docs)
        prompt = self.prompt_template.format(context=context, question=query)
        
        # Add language-specific instructions
        language_prompt = self.translation_service.get_language_prompt(request.lang.value)
        full_prompt = prompt + language_prompt
        
        # Generate answer with Bedrock first, fallback to local LLM
        answer = None
        
        print(f"ℹ️ Generating answer for: {query[:50]}...")
        print(f"ℹ️ Context length: {len(context)} chars")
        
        if self.llm is not None and self.bedrock_available:
            # Try AWS Bedrock Claude 3 Sonnet first
            try:
                from langchain_core.messages import HumanMessage
                messages = [HumanMessage(content=full_prompt)]
                response = self.llm.invoke(messages)
                answer = response.content
                print(f"✓ Bedrock generated answer: {len(answer)} chars")
                print(f"ℹ️ Answer preview: {answer[:100]}...")
            except Exception as e:
                print(f"⚠️ Bedrock failed during generation: {e}")
                print("✓ Falling back to local LLM with MiniLM")
                answer = None
        
        # Fallback to local LLM if Bedrock unavailable or failed
        if answer is None:
            answer = self.llm_service.generate_answer(full_prompt)
            print(f"✓ Local LLM generated answer: {len(answer)} chars")
            print(f"ℹ️ Answer preview: {answer[:100]}...")
        
        # Post-process answer to ensure document grounding
        print(f"ℹ️ Pre-grounding answer: {answer[:100]}...")
        answer = self._enforce_document_grounding(answer, relevant_docs, request)
        print(f"✓ Post-grounding answer: {answer[:100]}...")
        
        # Enhanced citation integration - update inline citations to match retrieved docs
        if relevant_docs and "Please ask questions directly related to the content" not in answer:
            # Create proper citation references for retrieved docs
            citation_refs = []
            for i, doc in enumerate(relevant_docs):
                doc_name = doc.get('document_name', f'Doc{i+1}')
                page_num = doc.get('page_number', i+1)
                citation_refs.append(f"[S{i+1}:{doc_name}:pp{page_num}]")
            
            # Replace any existing generic citations with actual ones
            import re
            # Replace patterns like [S1:filename.pdf:pp3] with actual retrieved document citations
            citation_pattern = r'\[S\d+:[^\]]+\]'
            existing_citations = re.findall(citation_pattern, answer)
            
            if existing_citations:
                # Replace existing citations with actual retrieved document citations
                for i, old_citation in enumerate(existing_citations):
                    if i < len(citation_refs):
                        answer = answer.replace(old_citation, citation_refs[i])
                print(f"✓ Updated {len(existing_citations)} inline citations with actual sources")
            elif '[S' not in answer:
                # Add citations if none exist
                answer += f"\n\n**Sources:** {', '.join(citation_refs)}"
                print(f"✓ Added {len(citation_refs)} source citations to answer")
        
        # Translate answer to target language if needed
        if request.lang.value != 'en':
            print(f"ℹ️ Translating answer to {request.lang.value}")
            answer = self.translation_service.translate_to_target_language(answer, request.lang.value)
            print(f"✓ Translation complete: {answer[:100]}...")
        
        # Extract citations from relevant docs with unique IDs
        citations = self._extract_citations(answer, relevant_docs)
        print(f"✓ Extracted {len(citations)} citations")
        
        # Ensure we always have citations if we have docs
        if not citations and relevant_docs:
            citations = []
            for i, doc in enumerate(relevant_docs):
                page_num = doc.get('page_number', i+1)
                doc_name = doc.get('document_name') or doc.get('metadata', {}).get('document_name') or f'Doc{i+1}'
                source_id = f"S{i+1}:{doc_name}:pp{page_num}"
                
                citation = Citation(
                    source_id=source_id,
                    span=doc['text'][:200] + "...",
                    confidence=doc['score'],
                    page_number=page_num,
                    document_name=doc_name
                )
                citations.append(citation)
            print(f"✓ Created {len(citations)} fresh citations for this query")
            for citation in citations:
                print(f"  • {citation.source_id} → {citation.document_name}")
        
        # Answer already translated by translation service above
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        print(f"✓ ANSWER GENERATION COMPLETE:")
        print(f"  • Answer length: {len(answer)} chars")
        print(f"  • Citations: {len(citations)}")
        print(f"  • Latency: {latency_ms}ms")
        print(f"  • Final answer: {answer[:150]}...")
        
        return AnswerResponse(
            answer=answer,
            citations=citations,
            usage={"tokens": len(answer.split()), "trace_id": trace_id},
            latency_ms=latency_ms
        )
    
    def _smart_document_retrieval(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Smart retrieval: First identify relevant document, then search within it"""
        if not self.documents:
            return []
        
        # Step 1: Group documents by document name
        doc_groups = {}
        for doc in self.documents:
            doc_name = doc.metadata.get('document_name', 'Unknown')
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(doc)
        
        print(f"Found {len(doc_groups)} documents: {list(doc_groups.keys())}")
        query_words = query.lower().split()
        name_indicators = ['age', 'name', 'student', 'julia', 'alice', 'bob', 'carol', 'david', 'emma']
        is_name_query = any(indicator in query_words for indicator in name_indicators)
        print(f"Query: '{query}' - Content-based document selection")
        
        # Step 2: Score each document group in parallel
        doc_scores = []
        query_lower = query.lower()
        
        # Parallel processing for multiple documents
        if len(doc_groups) > 1:
            from concurrent.futures import ThreadPoolExecutor
            
            def score_document(doc_item):
                doc_name, doc_list = doc_item
                combined_text = ' '.join([doc.page_content for doc in doc_list[:5]])
                doc_score = self._calculate_document_relevance(query_lower, combined_text)
                
                sample_text = combined_text[:100].replace('\n', ' ')
                doc_type = "CSV" if '.csv' in doc_name.lower() else "PDF"
                query_terms_found = [word for word in query.lower().split() if word in sample_text.lower()]
                
                return {
                    'doc_name': doc_name,
                    'score': doc_score,
                    'chunks': doc_list,
                    'sample': sample_text,
                    'type': doc_type,
                    'terms_found': query_terms_found
                }
            
            # Process documents in parallel
            with ThreadPoolExecutor(max_workers=min(4, len(doc_groups))) as executor:
                results = list(executor.map(score_document, doc_groups.items()))
            
            doc_scores = results
            
            # Log results
            for result in doc_scores:
                print(f"Document '{result['doc_name']}': relevance score {result['score']:.3f}")
                print(f"  Type: {result['type']} | Terms found: {result['terms_found']} | Sample: {result['sample']}...")
        else:
            # Single document - no need for parallel processing
            for doc_name, doc_list in doc_groups.items():
                combined_text = ' '.join([doc.page_content for doc in doc_list[:5]])
                doc_score = self._calculate_document_relevance(query_lower, combined_text)
                
                sample_text = combined_text[:100].replace('\n', ' ')
                doc_type = "CSV" if '.csv' in doc_name.lower() else "PDF"
                query_terms_found = [word for word in query.lower().split() if word in sample_text.lower()]
                
                doc_scores.append({
                    'doc_name': doc_name,
                    'score': doc_score,
                    'chunks': doc_list
                })
                print(f"Document '{doc_name}': relevance score {doc_score:.3f}")
                print(f"  Type: {doc_type} | Terms found: {query_terms_found} | Sample: {sample_text}...")
        
        # Step 3: Sort documents by relevance (parallel results already computed)
        doc_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Show parallel processing results
        if len(doc_scores) > 1:
            print(f"\u2699\ufe0f Parallel processing complete - best match: '{doc_scores[0]['doc_name']}' ({doc_scores[0]['score']:.3f})")
        
        # Step 4: Search within most relevant documents (prioritize high-scoring ones)
        relevant_docs = []
        
        # Use content-based selection with lower thresholds
        if doc_scores:
            # Parallel chunk search in top documents
            searched_any = False
            
            # If clear winner, search only that document
            if len(doc_scores) > 1 and doc_scores[0]['score'] > doc_scores[1]['score'] * 1.5 and doc_scores[0]['score'] > 0.2:
                print(f"\u2713 Clear winner - searching only in '{doc_scores[0]['doc_name']}' (score: {doc_scores[0]['score']:.3f})")
                chunk_results = self._retrieve_from_chunks(query, doc_scores[0]['chunks'], top_k)
                relevant_docs.extend(chunk_results)
                searched_any = True
            else:
                # Parallel search in multiple relevant documents
                search_docs = [doc for doc in doc_scores if doc['score'] > 0.05]
                
                if len(search_docs) > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    
                    def search_in_doc(doc_group):
                        return self._retrieve_from_chunks(query, doc_group['chunks'], top_k // len(search_docs) + 1)
                    
                    print(f"\u2699\ufe0f Parallel search in {len(search_docs)} documents")
                    with ThreadPoolExecutor(max_workers=min(3, len(search_docs))) as executor:
                        chunk_results_list = list(executor.map(search_in_doc, search_docs))
                    
                    # Combine results from all documents
                    for i, chunk_results in enumerate(chunk_results_list):
                        print(f"  \u2022 {search_docs[i]['doc_name']}: {len(chunk_results)} chunks found")
                        relevant_docs.extend(chunk_results)
                    searched_any = True
                else:
                    # Single document search
                    for doc_group in search_docs:
                        print(f"Searching within '{doc_group['doc_name']}' (score: {doc_group['score']:.3f})")
                        chunk_results = self._retrieve_from_chunks(query, doc_group['chunks'], top_k)
                        relevant_docs.extend(chunk_results)
                        searched_any = True
            
            # Fallback
            if not searched_any and doc_scores:
                print(f"\u26a0\ufe0f Fallback - searching in '{doc_scores[0]['doc_name']}' despite low score ({doc_scores[0]['score']:.3f})")
                chunk_results = self._retrieve_from_chunks(query, doc_scores[0]['chunks'], top_k)
                relevant_docs.extend(chunk_results)
        
        # Step 5: Sort final results and return top_k
        relevant_docs.sort(key=lambda x: x['score'], reverse=True)
        
        if relevant_docs:
            print(f"Final results from documents:")
            for doc in relevant_docs[:3]:
                print(f"  {doc['document_name']}: score {doc['score']:.3f}")
        
        return relevant_docs[:top_k]
    
    def _calculate_document_relevance(self, query_lower: str, doc_text: str) -> float:
        """Calculate how relevant a document is to the query"""
        doc_text_lower = doc_text.lower()
        query_words = [word for word in query_lower.split() if len(word) > 2]  # Filter meaningful words
        
        if not query_words:
            return 0.0
        
        scores = []
        
        # 1. Exact phrase match (highest priority)
        if query_lower in doc_text_lower:
            scores.append(1.0)
        
        # 2. Individual word matches with fuzzy matching
        exact_matches = 0
        partial_matches = 0
        
        for word in query_words:
            # Exact word match
            import re
            if re.search(r'\b' + re.escape(word) + r'\b', doc_text_lower):
                exact_matches += 1
            # Partial/fuzzy match
            elif word in doc_text_lower:
                partial_matches += 1
            # Stem matching for common variations
            elif word.endswith('ology') and word[:-5] in doc_text_lower:  # methodology -> method
                partial_matches += 1
            elif word == 'environmental' and ('environment' in doc_text_lower or 'ecology' in doc_text_lower):
                partial_matches += 1
            elif word == 'assessment' and ('assess' in doc_text_lower or 'evaluation' in doc_text_lower):
                partial_matches += 1
            elif word == 'impact' and ('effect' in doc_text_lower or 'influence' in doc_text_lower):
                partial_matches += 1
        
        # Calculate scores
        if exact_matches > 0:
            exact_score = exact_matches / len(query_words)
            scores.append(exact_score)
        
        if partial_matches > 0:
            partial_score = (partial_matches / len(query_words)) * 0.6
            scores.append(partial_score)
        
        # 3. Topic-based matching
        topic_keywords = {
            'environmental': ['environment', 'ecology', 'nature', 'pollution', 'conservation'],
            'assessment': ['assessment', 'evaluation', 'analysis', 'study', 'review'],
            'impact': ['impact', 'effect', 'influence', 'consequence', 'result'],
            'methodology': ['method', 'approach', 'technique', 'procedure', 'process']
        }
        
        topic_score = 0
        for query_word in query_words:
            if query_word in topic_keywords:
                related_words = topic_keywords[query_word]
                matches = sum(1 for word in related_words if word in doc_text_lower)
                if matches > 0:
                    topic_score += matches / len(related_words)
        
        if topic_score > 0:
            scores.append(topic_score * 0.4)
        
        # 4. Content-based topic matching (prioritize actual content)
        content_topics = {
            'project_management': ['project', 'management', 'methodology', 'agile', 'scrum', 'waterfall', 'planning'],
            'environmental': ['environmental', 'impact', 'assessment', 'ecology', 'pollution', 'conservation'],
            'data_science': ['data', 'analysis', 'science', 'statistics', 'mining', 'visualization']
        }
        
        content_score = 0
        for topic, keywords in content_topics.items():
            topic_matches = sum(1 for keyword in keywords if keyword in doc_text_lower)
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            
            if topic_matches > 0 and query_matches > 0:
                relevance = (topic_matches / len(keywords)) * (query_matches / len(query_words))
                content_score = max(content_score, relevance * 0.7)
        
        if content_score > 0:
            scores.append(content_score)
        
        # 5. Filename hint (only if content supports it)
        if content_score > 0.3:
            if 'spm' in doc_text_lower and any(word in query_lower for word in ['project', 'management']):
                scores.append(0.2)
            elif 'eia' in doc_text_lower and any(word in query_lower for word in ['environmental', 'impact']):
                scores.append(0.2)
        
        # 6. Fallback
        if not scores:
            basic_terms = ['project', 'management', 'environmental', 'impact', 'data', 'analysis']
            basic_matches = sum(1 for term in basic_terms if term in doc_text_lower and term in query_lower)
            if basic_matches > 0:
                scores.append(0.1 * (basic_matches / len(query_words)))
        
        return max(scores) if scores else 0.0
    
    def _retrieve_from_chunks(self, query: str, chunks: List[Document], top_k: int) -> List[Dict[str, Any]]:
        """Search within specific document chunks with semantic focus"""
        results = []
        query_lower = query.lower()
        query_words = [word.strip('?.,!') for word in query_lower.split() if len(word) > 2]
        
        print(f"Searching in {len(chunks)} chunks for query: {query}")
        
        # Extract main topic from query
        main_topic = self._extract_main_topic(query_lower)
        print(f"Main topic identified: {main_topic}")
        
        # Try semantic search with local embeddings first
        query_embedding = None
        if self.local_embedder is not None:
            try:
                query_embedding = self.local_embedder.encode(query)
            except Exception as e:
                pass
        
        # Intent-based search through chunks
        query_intent = getattr(self, '_current_query_intent', 'general')
        
        for i, doc in enumerate(chunks):
            doc_text_lower = doc.page_content.lower()
            
            # Skip chunks that don't contain the main topic
            if main_topic and main_topic not in doc_text_lower:
                continue
            
            scores = []
            aspect_type = "general"
            
            # 1. Intent-specific scoring
            if query_intent == 'applications':
                app_indicators = [
                    f"{main_topic} application", f"{main_topic} applications", f"{main_topic} use", f"{main_topic} uses",
                    f"application of {main_topic}", f"applications of {main_topic}", f"uses of {main_topic}",
                    "industry", "business", "real-world", "example", "case study", "implemented"
                ]
                if any(indicator in doc_text_lower for indicator in app_indicators):
                    scores.append(1.5)
                    aspect_type = "applications"
            
            elif query_intent == 'methods':
                method_indicators = [
                    f"{main_topic} method", f"{main_topic} technique", f"{main_topic} process",
                    "algorithm", "steps", "procedure", "approach", "methodology"
                ]
                if any(indicator in doc_text_lower for indicator in method_indicators):
                    scores.append(1.5)
                    aspect_type = "methods"
            
            # 2. General topic relevance
            if main_topic and main_topic in doc_text_lower:
                scores.append(0.8)
            
            # 3. Semantic similarity
            if query_embedding is not None:
                try:
                    doc_embedding = self.local_embedder.encode(doc.page_content)
                    from numpy import dot
                    from numpy.linalg import norm
                    similarity = dot(query_embedding, doc_embedding) / (norm(query_embedding) * norm(doc_embedding))
                    scores.append(similarity * 0.9)
                except:
                    pass
            
            # 4. Query word matching
            topic_matches = sum(1 for word in query_words if word in doc_text_lower)
            if topic_matches > 0:
                scores.append((topic_matches / len(query_words)) * 0.7)
            
            # Final score
            final_score = max(scores) if scores else 0
            
            if final_score > 0.3:
                # Use correct page number from metadata
                page_num = doc.metadata.get('page_number', 1)
                doc_name = doc.metadata.get('document_name') or f'Doc{i+1}'
                results.append({
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': final_score,
                    'source_id': f"S{len(results)+1}",
                    'chunk_id': doc.metadata.get('chunk_id', f'chunk_{i}'),
                    'page_number': page_num,
                    'document_name': doc_name,
                    'aspect_type': aspect_type
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        print(f"Found {len(results)} chunks for '{main_topic}' with intent '{getattr(self, '_current_query_intent', 'general')}'")
        for result in results[:3]:
            print(f"  Page {result['page_number']}: {result.get('aspect_type', 'general')} - {result['text'][:50]}...")
        
        return results
    
    def _extract_main_topic(self, query_lower: str) -> str:
        """Extract the main topic from the query"""
        # Remove question words
        query_clean = query_lower.replace('what is', '').replace('explain', '').replace('define', '')
        query_clean = query_clean.replace('?', '').strip()
        
        # Common topics in data science/analytics
        topics = {
            'data mining': 'data mining',
            'data warehouse': 'data warehouse', 
            'data warehousing': 'data warehouse',
            'etl': 'etl',
            'olap': 'olap',
            'oltp': 'oltp',
            'data analytics': 'data analytics',
            'machine learning': 'machine learning',
            'artificial intelligence': 'artificial intelligence',
            'big data': 'big data',
            'business intelligence': 'business intelligence'
        }
        
        # Find the best matching topic
        for topic_phrase, canonical_topic in topics.items():
            if topic_phrase in query_clean:
                return canonical_topic
        
        # Fallback to first meaningful word
        words = [w for w in query_clean.split() if len(w) > 3]
        return words[0] if words else query_clean
    
    def _detect_query_intent(self, query_lower: str) -> str:
        """Detect what specific aspect the user is asking about"""
        if any(word in query_lower for word in ['application', 'applications', 'use', 'uses', 'example', 'examples', 'applied']):
            return 'applications'
        elif any(word in query_lower for word in ['method', 'methods', 'technique', 'techniques', 'how', 'process', 'algorithm']):
            return 'methods'
        elif any(word in query_lower for word in ['concept', 'concepts', 'principle', 'principles', 'feature', 'features']):
            return 'concepts'
        elif any(word in query_lower for word in ['definition', 'define', 'what is', 'meaning', 'means']):
            return 'definition'
        else:
            return 'general'
    
    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents with proper citation format"""
        context_parts = []
        for i, doc in enumerate(docs):
            page_num = doc.get('page_number', i+1)
            doc_name = doc.get('document_name') or doc.get('metadata', {}).get('document_name') or f'Doc{i+1}'
            source_id = f"S{i+1}:{doc_name}:pp{page_num}"
            # Include source mapping for LLM to use correct citations
            context_parts.append(f"[{source_id}] {doc['text']}")
        
        # Add instruction for LLM to use these exact source IDs
        context_header = "Use these exact source IDs in your citations:\n"
        for i, doc in enumerate(docs):
            doc_name = doc.get('document_name', f'Doc{i+1}')
            page_num = doc.get('page_number', i+1)
            context_header += f"- Source {i+1}: [S{i+1}:{doc_name}:pp{page_num}]\n"
        
        return context_header + "\n" + "\n\n".join(context_parts)
    
    def _extract_citations(self, answer: str, docs: List[Dict[str, Any]]) -> List[Citation]:
        """Extract citations from answer with page numbers and document names"""
        citations = []
        
        # Create fresh citations for all retrieved docs
        for i, doc in enumerate(docs):
            page_num = doc.get('page_number', i+1)
            doc_name = doc.get('document_name') or doc.get('metadata', {}).get('document_name') or f'Doc{i+1}'
            source_id = f"S{i+1}:{doc_name}:pp{page_num}"
            
            print(f"📎 Creating citation {i+1}: {source_id} for doc: {doc_name}")
            
            citation = Citation(
                source_id=source_id,
                span=doc['text'],
                confidence=doc['score'],
                page_number=page_num,
                document_name=doc_name
            )
            citations.append(citation)
        
        return citations
    
    def _enforce_document_grounding(self, answer: str, relevant_docs: List[Dict[str, Any]], request: QueryRequest = None) -> str:
        """Ensure answer is grounded in documents and add safety checks"""
        # If we have relevant docs, the answer should be valid
        if relevant_docs:
            # Check for refusal messages when we have valid documents
            refusal_phrases = [
                "I can help answer questions based on your uploaded documents",
                "Please upload some documents first",
                "I cannot answer", "I don't have information", 
                "ask questions directly related"
            ]
            
            # If answer contains refusal phrases but we have docs, generate proper answer
            if any(phrase.lower() in answer.lower() for phrase in refusal_phrases):
                print("⚠️ LLM gave refusal message despite having relevant docs - generating proper answer")
                
                # Generate comprehensive structured answer
                query_lower = request.query.lower() if hasattr(request, 'query') else ""
                topic = query_lower.replace('what is', '').replace('explain', '').strip().title()
                if not topic:
                    topic = "Information"
                
                # Build comprehensive answer from all relevant docs
                answer_parts = [f"**{topic}**\n"]
                
                # Definition section
                answer_parts.append("## Definition:")
                for i, doc in enumerate(relevant_docs[:2]):
                    content = doc['text'][:300].strip()
                    doc_name = doc.get('document_name', f'Doc{i+1}')
                    page_num = doc.get('page_number', i+1)
                    answer_parts.append(f"• {content} [S{i+1}:{doc_name}:pp{page_num}]")
                answer_parts.append("")
                
                # Key Concepts section
                if len(relevant_docs) > 2:
                    answer_parts.append("## Key Concepts:")
                    for i, doc in enumerate(relevant_docs[2:5], 3):
                        content = doc['text'][:250].strip()
                        doc_name = doc.get('document_name', f'Doc{i}')
                        page_num = doc.get('page_number', i)
                        answer_parts.append(f"• {content} [S{i}:{doc_name}:pp{page_num}]")
                    answer_parts.append("")
                
                # Add source summary
                citations = []
                for i, doc in enumerate(relevant_docs[:5]):
                    doc_name = doc.get('document_name', f'Doc{i+1}')
                    page_num = doc.get('page_number', i+1)
                    citations.append(f"[S{i+1}:{doc_name}:pp{page_num}]")
                
                answer_parts.append(f"**Sources:** {', '.join(citations)}")
                
                return "\n".join(answer_parts)
    
    def _detect_query_intent(self, query_lower: str) -> str:
        """Detect what specific aspect the user is asking about"""
        if any(word in query_lower for word in ['application', 'applications', 'use', 'uses', 'example', 'examples', 'applied']):
            return 'applications'
        elif any(word in query_lower for word in ['method', 'methods', 'technique', 'techniques', 'how', 'process', 'algorithm']):
            return 'methods'
        elif any(word in query_lower for word in ['concept', 'concepts', 'principle', 'principles', 'feature', 'features']):
            return 'concepts'
        elif any(word in query_lower for word in ['definition', 'define', 'what is', 'meaning', 'means']):
            return 'definition'
        else:
            return 'general'
        
        return answer