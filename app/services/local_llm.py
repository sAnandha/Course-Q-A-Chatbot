import os
from typing import Dict, Any
import boto3
import json
from sentence_transformers import SentenceTransformer

class LocalLLMService:
    def __init__(self):
        # Initialize multilingual embedding model compatible with 384 dimensions
        try:
            self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            print("Using multilingual MiniLM embeddings (384 dimensions)")
        except:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("Using MiniLM embeddings (fallback)")
        
        # Try AWS Bedrock, fallback to local processing
        use_bedrock = os.getenv('USE_BEDROCK', 'false').lower() == 'true'
        
        if use_bedrock:
            try:
                self.bedrock = boto3.client(
                    'bedrock-runtime',
                    region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
                )
                self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
                print(f"✓ AWS Bedrock initialized: {self.model_id}")
                self.use_bedrock = True
            except Exception as e:
                print(f"Bedrock initialization failed: {e}")
                print("Using local LLM fallback")
                self.use_bedrock = False
        else:
            print("Using local LLM (Bedrock disabled)")
            self.use_bedrock = False
    
    def get_embedding(self, text: str) -> list:
        """Get embedding using local model"""
        try:
            embedding = self.embedder.encode(text).tolist()
            return embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 384  # Fallback
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using Claude 3 Sonnet"""
        if self.use_bedrock:
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1
                })
                
                response = self.bedrock.invoke_model(
                    modelId=self.model_id,
                    body=body
                )
                
                response_body = json.loads(response['body'].read())
                return response_body['content'][0]['text']
                
            except Exception as e:
                print(f"Bedrock error: {e}")
                return self._fallback_answer()
        else:
            return self._fallback_answer()
    
    def _fallback_answer(self) -> str:
        """Fallback answer when LLM is unavailable"""
        return "I can help answer questions based on your uploaded documents. Please upload some documents first, then ask me questions about them."