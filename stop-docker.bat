@echo off
echo Stopping RAG Chatbot containers...

docker stop rag-backend-container rag-frontend-container 2>nul
docker rm rag-backend-container rag-frontend-container 2>nul

echo Containers stopped and removed.
pause