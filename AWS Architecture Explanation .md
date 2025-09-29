## **AWS Service Mapping for This Architecture**

### **1. Frontend**

* **React Web App** →

  * Host on **Amazon S3 (Static Website Hosting)**
  * Distribute via **Amazon CloudFront (CDN)**
  * Use **Amazon Route 53** for domain + SSL/TLS with **AWS Certificate Manager**

---

### **2. API & Backend**

* **FastAPI Backend** →

  * Deploy as **AWS Lambda** (serverless) + **Amazon API Gateway** for HTTP endpoints
  * OR use **Amazon ECS (Fargate)** if you prefer container-based deployment

---

### **3. Document Processing**

* **Document Processor (Extract & Chunk)** →

  * **Amazon Textract** for text extraction from PDFs/images
  * **AWS Lambda** for chunking & pre-processing
  * **Amazon SQS** for async processing queue
  * **Amazon S3** for raw + processed document storage

---

### **4. Embedding & Vector Database**

* **Local Embedding Service** →

  * Use **Amazon Bedrock** (Titan Embeddings Model) or **SageMaker Endpoint** for embeddings
* **Vector DB (Pinecone in diagram)** →

  * Replace with **Amazon OpenSearch Service** (supports KNN vector search)
  * OR use **Aurora PostgreSQL + pgvector** if relational + vector search needed

---

### **5. Hybrid Retrieval**

* **BM25 Search** →

  * Use **Amazon OpenSearch Service** for keyword search (BM25 is default scoring)
* **Fusion Layer (Combining Results)** →

  * Implement inside **Lambda** or **ECS microservice**

---

### **6. LLM & Query Enhancement**

* **AWS Bedrock (Claude/Sonnet)** →

  * Query enhancement and generation with **Anthropic Claude**, **Llama 3**, or **Amazon Titan Text**
* Can be orchestrated with **Step Functions** for managing multi-step workflow

---

### **7. Citation & Response Processing**

* **Citation Composer & Response Formatter** →

  * AWS Lambda for response formatting
  * Use **Amazon DynamoDB** to store logs & citations for traceability

---

### **8. Monitoring & Evaluation**

* **Metrics Collection & Reporting** →

  * **Amazon CloudWatch** for logs, metrics, custom dashboards
  * **AWS X-Ray** for tracing requests
  * **Amazon QuickSight** for visualization of performance metrics
  * **AWS CloudTrail** for auditing API calls

---

### **9. Security**

* **Authentication** →

  * **Amazon Cognito** (User Pool + Identity Pool) for login/auth
* **IAM Roles** →

  * Fine-grained access control for Lambda, S3, OpenSearch

