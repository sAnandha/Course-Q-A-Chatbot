##  System Architecture – Course Q&A Chatbot

This architecture represents a **Hybrid RAG (Retrieval-Augmented Generation) system** for a Course Q&A Chatbot.
The solution integrates **document ingestion**, **hybrid search**, **LLM response generation**, and **metrics collection** to deliver high-quality answers with citations.

![System Architecture](./Architecture%20Diagram.png)

###  Components

1. **Frontend Layer**

   * **React Web App** – Modern, multilingual UI for document upload and question input.
   * Provides **inline citations** and displays formatted answers to users.

2. **FastAPI Backend**

   * Handles REST API requests from the frontend.
   * Validates uploaded documents and routes user queries for processing.

3. **Document Processing Pipeline**

   * Extracts, cleans, and preprocesses text from uploaded documents.
   * Generates **local embeddings** for semantic search.
   * Indexes keywords for BM25-based keyword search.

4. **Vector Database (Pinecone)**

   * Stores embeddings for fast similarity search.
   * Supports scalable, low-latency vector retrieval.

5. **Hybrid Retrieval System**

   * Combines **BM25 keyword search** + **vector search**.
   * Uses **result fusion** and **cross-encoder re-ranking** to return the most relevant context.

6. **LLM & Query Enhancement**

   * Query enhancement layer rewrites or clarifies user questions.
   * AWS Bedrock (Claude / Titan / other model) generates context-aware answers.

7. **Citation & Response Processing**

   * Adds inline citations to retrieved passages.
   * Formats the final answer for display.

8. **Evaluation & Monitoring**

   * Collects metrics like latency, hit ratio, and relevancy scores.
   * Supports A/B testing and continuous improvement of retrieval quality.

---

###  Workflow

#### **1. Document Ingestion**

1. User uploads a document via the React UI.
2. FastAPI backend validates and forwards it to the document processor.
3. Text is extracted, cleaned, and converted into embeddings.
4. BM25 keyword index and Pinecone vector database are updated.

#### **2. Query Handling**

1. User enters a question in the UI.
2. FastAPI enhances the query (spelling corrections, paraphrasing).
3. Hybrid Retrieval performs:

   * **BM25 search** for keyword relevance.
   * **Vector search** for semantic relevance.
   * **Result fusion** merges results.
   * **Cross-encoder reranker** ranks passages.

#### **3. LLM Generation**

1. Most relevant passages are fed to the LLM.
2. The LLM generates a contextually correct answer.

#### **4. Citation & Formatting**

1. Retrieved sources are cited inline.
2. Response formatter prepares user-friendly output.

#### **5. Display & Monitoring**

1. Final answer with citations is displayed in the UI.
2. Logs and metrics are captured for performance monitoring and evaluation.




