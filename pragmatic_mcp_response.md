# Pragmatic Memory MCP System Design

## Core Decision: Vector-First, Graph-Ready

**Recommendation: Build a vector-only MVP and defer the graph component to Month 2.**

### Justification:
1. **Time-to-Value:** 2-3 person team can ship vector search in 4 weeks vs doubling complexity with hybrid
2. **Sufficient for MVP:** Vector search handles 80% of initial use cases for memory retrieval
3. **Graceful Evolution:** Design schema to be "graph-aware" from day one for easy migration
4. **Operational Simplicity:** Single database is vastly simpler to debug and maintain

## 1. MVP Architecture

### Components (Only 3!):
1. **API Gateway** - Manages ingress, auth, rate limiting
2. **Orchestration Service** - Python/FastAPI with LangChain on AWS Fargate
3. **Qdrant Cloud** - Managed vector database

### Data Flow:
```
Ingestion:
External System → API Gateway → Orchestration Service → Embedding API → Qdrant Cloud

Retrieval:
User Query → API Gateway → Orchestration Service → Embedding API → Qdrant Search → Format → LLM
```

### Failure Points & Degradation:
- **Qdrant timeout (150ms)**: Log error, serve stateless response
- **Service failure**: Fargate auto-restarts containers
- **Embedding API failure**: Retry for ingestion, fail individual requests for retrieval

## 2. Technology Stack

- **Orchestration:** LangChain (mature, broad Qdrant support)
- **Vector DB:** Qdrant Cloud v1.7+ (4x better RPS than alternatives)
- **Deployment:** AWS Fargate (serverless, auto-scaling)

### Why Not Alternatives:
- **Neo4j w/ vectors:** Impedance mismatch - paying graph complexity for vector workload
- **FalkorDB:** Main advantage is graph ops which we're deferring

## 3. Data Schema (Graph-Ready)

### Ingestion Pipeline:
1. Receive JSON memory event
2. Chunk with RecursiveCharacterTextSplitter (~512 tokens)
3. Generate embeddings (text-embedding-3-small)
4. Batch upsert to Qdrant

### JSON Schema:
```json
{
  "text": "The user confirmed the final budget for Project Phoenix is $250,000.",
  "metadata": {
    "memory_id": "mem_uuid_123",
    "conversation_id": "conv_uuid_abc",
    "user_id": "user_456",
    "timestamp": "2023-10-27T10:00:00Z",
    "entities": [
      { "id": "proj_phoenix", "label": "Project", "name": "Project Phoenix" },
      { "id": "budget_250k", "label": "Budget", "value": 250000 }
    ]
  }
}
```

### Updates & Contradictions:
- **Updates:** Last write wins (simple for MVP)
- **Contradictions:** Pass both to LLM, let it synthesize

## 4. Performance & Evaluation

### Expected Latencies (p99):
- Query Embedding: ~50ms
- Qdrant Search: ~40-80ms
- Network Overhead: ~20ms
- **Total: ~110-150ms** (well under 200ms SLA)

### Evaluation Strategy:
- Golden dataset of 100 Q&A pairs
- Metrics: Hit Rate@5, Latency p99
- Run automatically on every commit

### Degradation:
- Circuit breaker trips if p99 > 150ms for 1 minute
- Serves stateless responses for 5 minutes

## 5. Cost Analysis

| Memories | Qdrant | Fargate | Embeddings | Monitoring | Total/Month |
|----------|--------|---------|------------|------------|-------------|
| 1M | $100 | $150 | $50 | $50 | **$350** |
| 10M | $250 | $200 | $400 | $100 | **$950** |
| 100M | $800 | $300 | $1000+ | $250 | **$2350+** |

**Optimization:** Self-host embeddings at scale to shift from per-token to fixed costs

## 6. 3-Month Roadmap

### Weeks 1-2: Scaffolding & Ingestion
- Provision Qdrant Cloud
- Set up IaC (Terraform/CDK)
- Build /ingest endpoint
- Define v1 schema

### Weeks 3-4: Retrieval & MVP
- Build /retrieve endpoint
- Integrate Qdrant search
- Set up CI/CD
- Implement evaluation

### Weeks 5-6: Hardening & Launch
- Load testing
- Monitoring dashboards
- On-call runbooks
- Internal/canary release

### Month 2: Graph Fast-Follow
- Choose graph DB (FalkorDB vs Neo4j)
- Backfill from Qdrant metadata
- Dual writes to both DBs
- v2 retrieval with graph traversal

### Month 3: Optimization
- Self-hosted embeddings
- Advanced reranking
- Conversation summarization

## Key Insights

This design prioritizes:
- **Shipping in 6 weeks** over theoretical perfection
- **Operational simplicity** over feature completeness
- **Proven technology** over cutting-edge research
- **Clear upgrade path** over trying to build everything upfront

The "Vector-First, Graph-Ready" approach delivers immediate value while setting up for future enhancements without painful migrations.