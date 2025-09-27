@echo off
echo Starting RAG Chatbot with Docker...

echo Building backend image...
docker build -t rag-backend .

echo Building frontend image...
cd frontend
docker build -t rag-frontend .
cd ..

echo Creating network...
docker network create rag-network 2>nul

echo Starting backend container...
docker run -d --name rag-backend-container --network rag-network -p 8000:8000 --env-file .env rag-backend

echo Starting frontend container...
docker run -d --name rag-frontend-container --network rag-network -p 3000:3000 rag-frontend

echo Application started!
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
echo API Docs: http://localhost:8000/docs

pause