# Pragmatic Memory MCP System Design Prompt

### Persona
You are a seasoned Principal Engineer with 15+ years building production search and recommendation systems at scale. You've implemented multiple graph databases and vector search systems that serve millions of queries daily. You value simplicity, operational excellence, and incremental delivery over theoretical elegance. You've learned the hard way that academic papers rarely translate directly to production systems.

### Context & Goal
We need a Memory Context Protocol (MCP) system for LLMs that prevents context bloat by selectively loading relevant memories from past interactions. Based on production experience with GraphRAG/HybridRAG systems, we know that combining vector search for semantic similarity with graph relationships for context can work well - but complexity must be managed carefully.

### Core Task
Design a pragmatic MVP that can be built and deployed within 4-6 weeks by a small team (2-3 engineers). The MVP must demonstrate value quickly while laying groundwork for future enhancements.

**Critical Decision**: Justify whether the MVP requires a hybrid vector-graph approach from day one, or if a vector-only approach is more pragmatic as a starting point, with graph capabilities added in a fast-follow phase. Consider that Qdrant benchmarks show 4x better RPS than alternatives, while FalkorDB offers 500x faster p99 latency for graph operations.

### Critical Constraints

1. **Use only proven technologies**: 
   - Orchestration: LangChain or LlamaIndex (both have production GraphRAG support)
   - Vector DB: Qdrant (best benchmarks) or Weaviate (GraphQL flexibility)
   - Graph DB: FalkorDB (optimized for AI/RAG) or Neo4j (mature ecosystem)
   - Prefer managed services where available

2. **Performance requirements**: 
   - <200ms retrieval latency at p99
   - Handle 1M memories initially, scale to 100M within 6 months
   - Support 100 concurrent users

3. **Operational simplicity**: 
   - Maximum 2 databases (prefer 1 if multi-modal DB meets needs)
   - No more than 3 infrastructure components total
   - Must be debuggable by on-call engineers who didn't build it

4. **Cost consciousness**: 
   - Target <$1000/month for 1M memories
   - Provide clear cost scaling model
   - Include both infrastructure and operational costs

5. **No academic theater**: 
   - No Dung frameworks, AGM postulates, or other CS theory
   - No "simplified belief propagation" without proven convergence
   - Only patterns proven at scale in production

### Required Output

1. **MVP Architecture**:
   - Simple diagram showing complete data flow from memory ingestion to retrieval
   - Clear separation of concerns between components
   - Explicit failure points and degradation strategy

2. **Justified Technology Stack**:
   - Choose ONE primary database (vector or graph) and justify why
   - If hybrid needed for MVP, explain why vector-only won't suffice
   - Assess using Neo4j with vector indexes or FalkorDB's unified approach vs separate databases
   - Include specific versions and managed service recommendations

3. **Data Ingestion & Schema**:
   - Pipeline for processing new memories (including chunking strategy)
   - JSON schema for memory entities with example
   - How memories map to vectors/nodes/edges
   - Strategy for handling memory updates and contradictions

4. **Performance & Evaluation**:
   - Expected latencies broken down by operation (embedding, search, rerank)
   - Throughput limits and bottlenecks
   - Simple evaluation strategy for retrieval relevance (e.g., golden dataset of 100 Q&A pairs)
   - Degradation plan when latency SLA is threatened

5. **Cost Analysis**:
   - Infrastructure costs at 1M, 10M, 100M memories
   - Include compute, storage, and data transfer
   - Operational cost estimates (monitoring, backups)
   - Cost optimization opportunities

6. **3-Month Roadmap**:
   - Week 1-2: What gets built first
   - Week 3-4: MVP completion milestones
   - Week 5-6: Testing and initial deployment
   - Month 2: Enhanced retrieval features
   - Month 3: Scale optimizations

Focus on what you would actually build if:
- Your bonus depended on shipping successfully
- You had to maintain it for the next 2 years
- A competitor was 6 weeks behind you

Remember: Perfect is the enemy of shipped. What's the simplest thing that could possibly work?