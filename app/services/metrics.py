import time
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import csv
import os

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.metrics = []
        self.metrics_file = "metrics.csv"
        self._init_metrics_file()
    
    def _init_metrics_file(self):
        """Initialize metrics CSV file with headers"""
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'trace_id', 'query', 'language', 'latency_ms', 
                    'tokens_used', 'citations_count', 'status', 'error'
                ])
    
    def log_query(self, trace_id: str, query: str, language: str, 
                  latency_ms: int, tokens_used: int, citations_count: int, 
                  status: str = "success", error: str = None):
        """Log query metrics"""
        
        # Structured logging
        log_data = {
            "trace_id": trace_id,
            "query": query[:100],  # Truncate for privacy
            "language": language,
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "citations_count": citations_count,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            log_data["error"] = error
            logger.error(f"Query failed: {json.dumps(log_data)}")
        else:
            logger.info(f"Query processed: {json.dumps(log_data)}")
        
        # Store metrics
        self.metrics.append(log_data)
        
        # Write to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log_data['timestamp'], trace_id, query[:50], language,
                latency_ms, tokens_used, citations_count, status, error or ""
            ])
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics:
            return {"message": "No metrics available"}
        
        latencies = [m['latency_ms'] for m in self.metrics if m['status'] == 'success']
        
        if not latencies:
            return {"message": "No successful queries"}
        
        return {
            "total_queries": len(self.metrics),
            "successful_queries": len(latencies),
            "error_rate": (len(self.metrics) - len(latencies)) / len(self.metrics),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0,
            "avg_tokens": sum(m['tokens_used'] for m in self.metrics if m['status'] == 'success') / len(latencies) if latencies else 0,
            "avg_citations": sum(m['citations_count'] for m in self.metrics if m['status'] == 'success') / len(latencies) if latencies else 0
        }

# Global metrics collector
metrics_collector = MetricsCollector()