# Course Q&A RAG Chatbot

A production-ready multilingual course Q&A chatbot with hybrid retrieval, inline citations, and comprehensive document analysis capabilities.This Course Q&A Chatbot uses RAG (Retrieval-Augmented Generation) to answer questions based on course materials with source citations. It combines semantic search and LLMs to deliver accurate, transparent, and context-aware responses, helping students learn efficiently and reducing instructor workload.

##  Architecture

**Tech Stack:**
- **Backend**: FastAPI + LangChain + Python 3.10
- **Frontend**: React + Material-UI
- **Vector DB**: Pinecone + Local embeddings
- **LLM**: AWS Bedrock (Claude 3 Sonnet)
- **Deployment**: Docker + Docker Compose

**AWS Services:**
- Amazon Bedrock (Claude 3 Sonnet)
**Pinecone Services**
- Pinecone (Vector database)

## 📁 Project Structure

```

├── app/                               # FastAPI backend
│   ├── api/
│   │   └── main.py                    # API endpoints
│   ├── models/
│   │   └── schemas.py                 # Pydantic models
│   ├── services/
│   │   ├── langchain_rag.py           # Main RAG service
│   │   ├── document_processor.py      # Document processing
│   │   ├── hybrid_retriever.py        # Hybrid search
│   │   ├── cross_encoder.py           # Reranking
│   │   ├── session_manager.py         # Session management
│   │   ├── metrics.py                 # Performance tracking
│   │   └── translation_service.py     # Hindi-English translation
│   └── __init__.py
├── frontend/                          # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   └── SourceViewer.jsx       # Citation viewer
│   │   ├── App.js                     # Main component
│   │   ├── App.css                    # Styling
│   │   └── index.js                   # Entry point
│   ├── public/
│   │   └── index.html                 # HTML template
│   └── package.json                   # Dependencies
├── static/
│   └── NotoSansDevanagari-Regular.ttf # Hindi font
├── .env.example                       # Environment template
├── docker-compose.yml                 # Docker configuration
├── Dockerfile.backend                 # Backend container
├── Dockerfile.frontend                # Frontend container
├── requirements.txt                   # Python dependencies
├── run-docker.bat                     # Windows Docker script
├── stop-docker.bat                    # Windows stop script
└── README.md                          # This file
```

##  Prerequisites

### Required Services
1. **AWS Account** with Bedrock access
2. **Pinecone Account** with vector index
3. **Docker** installed locally

### API Keys Needed
- AWS Access Key & Secret Key  ( or ) AWS API Key for BedRock 
- Pinecone API Key & Index Name

##  Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git

# Copy environment Variables
cp .env
```

### 2. Configure Credentials
Edit `.env` with your API keys:
```bash
# Required: Add your actual credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_DEFAULT_REGION=us-east-1           #Ensure all the services are available in the Region 
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
```

### 3. Manual and Docker Setup


### Backend
```bash
# Install Python dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Run FastAPI server
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## 🐳 Docker Commands

```bash
# Start services
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild specific service
docker-compose up --build backend
```


### 4. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs



##  API Endpoints

### Core Endpoints
- `POST /answer` - Query chatbot with citations
- `POST /upload` - Upload documents (PDF/MD/CSV)
- `GET /source/{source_id}` - Retrieve source content
- `POST /session/create` - Create new session
- `GET /session/{id}/export` - Export session as PDF
- `POST /feedback` - Submit feedback
- `GET /metrics` - System performance metrics

### Example Usage
```bash
# Query the chatbot
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is data mining?", "lang": "en", "top_k": 5}'

# Upload document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"

# Get source details
curl "http://localhost:8000/source/S1:document.pdf:pp3"
```

## 🎯 How It Works

### 1. Document Upload Process
1. **File Upload**: PDF/CSV/MD files via drag & drop or button
2. **Text Extraction**: Content extracted with metadata preservation
3. **Semantic Chunking**: 500-1000 token chunks with 100-token overlap
4. **Embedding Generation**: Local multilingual model (384 dimensions)
5. **Vector Storage**: Stored in Pinecone with BM25 indexing

### 2. Query Processing Flow
1. **Intent Detection**: Identifies query type (definition, application, method)
2. **Translation**: Hindi queries translated to English for processing
3. **Hybrid Retrieval**: 
   - Vector search in Pinecone
   - BM25 keyword matching
   - Cross-encoder reranking
4. **LLM Generation**: Claude 3 Sonnet generates structured answers
5. **Citation Integration**: Inline citations with source mapping
6. **Response Translation**: Answers translated back to target language

### 3. Answer Structure
```
**Topic Name**

## Definition:
• Clear explanation [S1:doc.pdf:pp3]

## Key Concepts:
• Concept 1: Details [S2:doc.pdf:pp5]
• Concept 2: Details [S3:doc.pdf:pp7]

## Methods/Techniques:
• Method 1: Implementation [S4:doc.pdf:pp9]

## Applications/Examples:
• Application 1: Use case [S5:doc.pdf:pp11]

Sources: [S1:doc.pdf:pp3], [S2:doc.pdf:pp5], ...
```

##  Features

 - **Hybrid Retrieval**      : Vector search + BM25 + Cross-encoder reranking  
 - **Inline Citations**      : [S1:filename.pdf:pp3] format with clickable source viewer  
 - **Multilingual Support**  : English + Hindi with automatic translation  
 - **Document Processing**   : PDF, Markdown, CSV upload & processing  
 - **Session Management**    : Isolated document sessions with export  
 - **PDF Export**            : Export Q&A sessions as PDF reports (English only)  
 - **Real-time UI**          : Live document upload with drag & drop  
 - **Intent Detection**      : Automatically detects query intent (definitions, applications, methods)  
 - **Comprehensive Answers** : Structured responses with multiple sections  

##  Deployment Options

### Local Development
```bash
docker-compose up --build
```

## 🔧 Configuration

### Environment Variables
```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
```

##  Performance Optimization

### Retrieval Optimization
- **Parallel Processing**: Vector and BM25 search run concurrently
- **Smart Document Selection**: Content-based document prioritization
- **Intent-based Filtering**: Query intent drives search strategy
- **Cross-encoder Reranking**: Improves precision of results

### Response Time Optimization
- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Reuse database connections
- **Caching**: Frequent queries cached for performance
- **Batch Processing**: Multiple documents processed simultaneously

##  Troubleshooting

### Common Issues

**Port conflicts:**
```bash
# Change ports in docker-compose.yml
ports:
  - "3001:3000"  # Frontend
  - "8001:8000"  # Backend
```

**Permission issues (Linux/Mac):**
```bash
sudo chown -R $USER:$USER ./
```

**Clean restart:**
```bash
docker-compose down -v
docker system prune -f
docker-compose up --build
```

**Hindi PDF Export:**
- Currently only English Q&A sessions can be exported to PDF
- Hindi PDF conversion is under development

### Support
- Check logs: `docker-compose logs -f`
- API docs: http://localhost:8000/docs
- Issues: Create GitHub issue with logs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request


## 🎯 Key Features Explained

### Hybrid Retrieval System
Combines three retrieval methods for optimal accuracy:
1. **Vector Search**: Semantic similarity using embeddings
2. **BM25 Search**: Keyword-based lexical matching
3. **Cross-encoder Reranking**: Context-aware result refinement


### Citation System
- **Inline Citations**: [S1:filename.pdf:pp3] format with Click to view source content
- **Source Viewer**: Right panel shows detailed source information

### Multilingual Support
- **Query Translation**: Hindi queries translated to English
- **Response Translation**: Answers translated back to Hindi

## 📄 License

This project is licensed under the MIT License.

---

**⚠️ Important**: Always configure your `.env` file with real credentials before running the application!

---

