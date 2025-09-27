# Manual Docker Setup Guide

## Prerequisites
- Docker Desktop installed and running
- Git (optional, for cloning)

## Step 1: Prepare Environment
```bash
# Navigate to project directory
cd c:\Users\Anandha Nivas\Desktop\eagle

# Ensure .env file exists with your credentials
# Copy from .env.example if needed
copy .env.example .env
```

## Step 2: Build Backend Docker Image
```bash
# Build backend image
docker build -t rag-backend -f Dockerfile.backend .
```

## Step 3: Build Frontend Docker Image
```bash
# Build frontend image
docker build -t rag-frontend -f Dockerfile.frontend .
```

## Step 4: Create Docker Network
```bash
# Create network for containers to communicate
docker network create rag-network
```

## Step 5: Run Backend Container
```bash
# Run backend container
docker run -d \
  --name rag-backend \
  --network rag-network \
  -p 8000:8000 \
  --env-file .env \
  rag-backend
```

## Step 6: Run Frontend Container
```bash
# Run frontend container
docker run -d \
  --name rag-frontend \
  --network rag-network \
  -p 3000:3000 \
  rag-frontend
```

## Step 7: Verify Setup
```bash
# Check running containers
docker ps

# Check backend logs
docker logs rag-backend

# Check frontend logs
docker logs rag-frontend
```

## Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Troubleshooting Commands
```bash
# Stop containers
docker stop rag-frontend rag-backend

# Remove containers
docker rm rag-frontend rag-backend

# Remove images
docker rmi rag-frontend rag-backend

# Remove network
docker network rm rag-network

# View container logs
docker logs -f rag-backend
docker logs -f rag-frontend
```

## Environment Variables Required
Ensure your `.env` file contains:
```
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
```