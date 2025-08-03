# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Reverie** is a Relational Memory MCP (Model Context Protocol) implementation that uses a knowledge graph structure to intelligently manage memory and prevent context bloat. The system uses Neo4j graph database with vector embeddings to selectively load only relevant memories based on semantic, topological, and temporal relevance scores.

## Development Setup & Commands

### Python Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (once requirements.txt exists)
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio black ruff mypy
```

### Neo4j Database
```bash
# Start Neo4j using Docker
docker run -d \
  --name reverie-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
  neo4j:5-enterprise

# Check Neo4j status
docker ps | grep reverie-neo4j

# Access Neo4j Browser
# http://localhost:7474
```

### Development Commands
```bash
# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_retrieval.py::test_semantic_search -v

# Code formatting
black src/ tests/

# Linting
ruff check src/ tests/

# Type checking
mypy src/

# Run MCP server locally
python -m src.mcp_server

# Test MCP connection
mcp test localhost:5000
```

## Architecture & Code Structure

### Core Components

1. **Graph Database Layer** (`src/graph/`)
   - `neo4j_client.py`: Neo4j connection management and query execution
   - `schema.py`: Node and edge schema definitions
   - `vector_index.py`: HNSW vector index management

2. **Embedding System** (`src/embeddings/`)
   - `embedder.py`: Sentence transformer integration
   - `cache.py`: Embedding cache management
   - `batch_processor.py`: Batch embedding generation

3. **Retrieval Engine** (`src/retrieval/`)
   - `hybrid_scorer.py`: Multi-factor relevance scoring
   - `graph_traversal.py`: BFS/DFS traversal algorithms
   - `context_manager.py`: Memory deduplication and staleness tracking

4. **MCP Server** (`src/mcp/`)
   - `server.py`: MCP protocol implementation
   - `handlers.py`: Request handlers for memory operations
   - `session.py`: Per-conversation state management

### Key Design Patterns

- **Hybrid Scoring**: Combines semantic similarity, graph topology, temporal relevance, and importance scores
- **Lazy Loading**: Node content loaded only when needed to optimize memory usage
- **Session Isolation**: Each conversation maintains independent context state
- **Incremental Updates**: Embeddings and indexes updated incrementally for performance

## Specialized Agents for This Project

### 1. Graph Schema Agent
Use when designing or modifying the knowledge graph structure:
```
"Help me design the graph schema for storing procedural memories with dependencies"
"What properties should I add to support memory versioning?"
```

### 2. Vector Search Optimization Agent
Use for optimizing semantic search and embeddings:
```
"How can I improve the vector similarity search performance?"
"What's the best embedding model for technical documentation memories?"
```

### 3. Relevance Scoring Agent
Use when tuning the hybrid relevance scoring algorithm:
```
"How should I adjust the relevance weights for a code-focused use case?"
"What temporal decay factor works best for long-running conversations?"
```

### 4. Context Management Agent
Use for memory deduplication and staleness strategies:
```
"How can I prevent loading duplicate memories in the same context?"
"What's an efficient way to track memory staleness?"
```

### 5. Performance Profiling Agent
Use for identifying and resolving performance bottlenecks:
```
"Profile the retrieval pipeline and identify slow operations"
"How can I optimize graph traversal for deep relationship chains?"
```

## Critical Implementation Guidelines

### Neo4j Query Optimization
- Always use parameterized queries to prevent Cypher injection
- Create indexes before bulk inserts
- Use `PROFILE` to analyze query performance
- Leverage native graph algorithms when possible

### Vector Embedding Best Practices
- Normalize embeddings before storage
- Use batch processing for multiple embeddings
- Implement embedding versioning for model updates
- Cache frequently accessed embeddings

### Memory Management
- Set appropriate heap size for Neo4j based on graph size
- Implement connection pooling for database access
- Use streaming for large result sets
- Monitor memory usage during graph traversal

### Error Handling
- Implement exponential backoff for Neo4j connection failures
- Validate embedding dimensions before storage
- Handle partial retrieval failures gracefully
- Log detailed errors for debugging

### Testing Strategies
- Use in-memory Neo4j for unit tests
- Create fixture data representing different memory patterns
- Test edge cases: circular dependencies, orphaned nodes
- Benchmark retrieval performance with varying graph sizes

## Common Pitfalls to Avoid

1. **Over-fetching**: Don't load entire subgraphs when only specific nodes are needed
2. **Embedding Drift**: Monitor for semantic drift when updating embedding models
3. **Relationship Explosion**: Limit relationship creation to maintain query performance
4. **Token Budget Violations**: Always respect the configured token limit
5. **Stale Cache**: Implement proper cache invalidation for embeddings

## Debugging Commands

```bash
# Check Neo4j connection
python -c "from src.graph import neo4j_client; neo4j_client.test_connection()"

# Inspect graph statistics
python -m src.tools.graph_stats

# Validate embeddings
python -m src.tools.validate_embeddings

# Export graph for visualization
python -m src.tools.export_graph --format gephi
```

## Performance Benchmarks

Target metrics based on MCP_PLAN.md:
- Retrieval latency: < 200ms for typical queries
- Embedding generation: < 50ms per text chunk
- Graph traversal: < 100ms for 2-hop expansion
- Memory efficiency: 50% reduction vs naive loading