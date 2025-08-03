# Reverie - Relational Memory MCP Implementation Plan

## Overview

**Reverie** is a Model Context Protocol (MCP) system that implements intelligent memory management using a knowledge graph structure. The system prevents context bloat by selectively loading only relevant memories based on semantic, topological, and temporal relevance scores.

## Core Architecture

### Data Structure: Knowledge Graph

The system uses a directed graph where:
- **Nodes** represent memory units (concepts, facts, procedures)
- **Edges** represent relationships between memories
- **Properties** store metadata, embeddings, and content

### Technology Stack

- **Graph Database**: Neo4j with HNSW vector indexes
- **Embedding Model**: Sentence transformers (e.g., all-MiniLM-L6-v2)
- **MCP Framework**: Python-based MCP server
- **Vector Search**: Neo4j native vector similarity search

## Schema Design

### Memory Node Structure

```json
{
    "node_id": "uuid",
    "content": "text content of the memory",
    "embedding": [/* vector array */],
    "metadata": {
        "created_at": "timestamp",
        "last_accessed": "timestamp",
        "access_count": 0,
        "node_type": "concept|fact|procedure|context",
        "importance": 0.0,  // 0-1 scale
        "source": "user|system|derived"
    }
}
```

### Edge Structure

```json
{
    "relationship": "related_to|depends_on|part_of|leads_to|contradicts",
    "strength": 0.0,  // 0-1 scale
    "created_at": "timestamp",
    "metadata": {}
}
```

## Relevance Scoring Algorithm

### Hybrid Scoring Model

The system calculates relevance using a weighted combination of factors:

```python
relevance_score = (
    w1 * semantic_score +     # Content similarity
    w2 * topological_score +  # Graph structure
    w3 * temporal_score +     # Recency
    w4 * importance_score     # Node importance
)
```

### Component Calculations

1. **Semantic Score**: Cosine similarity between query embedding and node embedding
2. **Topological Score**: `1 / (graph_distance + 1)` with connection density boost
3. **Temporal Score**: `exp(-λ * time_since_access)`
4. **Importance Score**: Pre-computed node importance based on connectivity and usage

## Context Management

### Tracking Loaded Memories

- Maintain session-specific set of loaded memory IDs
- Track token count when each memory was loaded
- Implement staleness threshold for memory refresh

### Memory Refresh Strategy

```python
STALENESS_THRESHOLD = 4096  # tokens

# Memory becomes eligible for reload when:
current_tokens - load_token_count > STALENESS_THRESHOLD
```

### Deduplication

- Check loaded_memory_ids before adding to context
- Use Bloom filter for efficient membership testing at scale

## Retrieval Process

### Step-by-Step Retrieval

1. **Anchor Search**: Find top-k semantically similar nodes to query
2. **Graph Traversal**: Expand from anchors using BFS (configurable depth)
3. **Candidate Collection**: Gather all unique nodes from traversal
4. **Scoring & Ranking**: Calculate hybrid relevance scores
5. **Token Budget Selection**: Select highest scoring nodes within token limit
6. **Context Update**: Track loaded memories and token positions

### Optimization Strategies

- Pre-compute and cache node embeddings
- Use graph database native traversal algorithms
- Implement incremental index updates for new memories

## MCP Interface Design

### Core Methods

```python
# Retrieve relevant memories for a query
retrieve_memories(
    query: str,
    max_tokens: int = 2000,
    search_depth: int = 2,
    staleness_threshold: int = 4096
) -> List[Memory]

# Store new memory with relationships
store_memory(
    content: str,
    relationships: List[Dict],
    node_type: str = "concept",
    importance: float = 0.5
) -> str

# Update memory access metadata
touch_memory(
    node_id: str
) -> None

# Get current context state
get_context_state() -> Dict
```

### Session Management

- Initialize per-conversation state
- Track total token count
- Maintain context history
- Support session persistence/restoration

## Pre-Implementation Research Topics

### 1. Personal Memory System Architecture ✓
- **Entity Extraction**: Using hybrid NLP (SpaCy) + LLM approach for extracting concepts from user commands
- **Relationship Inference**: Automatic and LLM-powered relationship detection between memories
- **Incremental Graph Building**: Efficient single-node operations for personal-scale knowledge graphs
- **Smart Recall**: Multi-stage retrieval combining embeddings, entities, and graph traversal

**Status**: Research completed. See [Personal Memory System Research](./research/personal_memory_system_research.md)

### 2. Embedding Models & Semantic Search
- **Sentence Transformer Selection**: Comparing models (all-MiniLM-L6-v2 vs alternatives) for memory-specific use cases
- **Embedding Drift**: Strategies for handling semantic changes when updating embedding models
- **Multi-modal Embeddings**: Future-proofing for image/code snippet support
- **Chunking Strategies**: Optimal text segmentation for memory units

### 3. Relevance Scoring & Ranking
- **Weight Optimization**: Methods for tuning hybrid scoring weights (semantic, topological, temporal, importance)
- **Temporal Decay Functions**: Comparing exponential decay vs other decay models
- **Importance Propagation**: Graph algorithms for computing node importance (PageRank, centrality measures)
- **Connection Density Metrics**: How to calculate and incorporate connection density into topological scores

### 4. Memory Management & Context Control
- **Bloom Filter Implementation**: Size vs false positive trade-offs for deduplication
- **Token Counting Strategies**: Accurate token measurement across different LLM tokenizers
- **Staleness Threshold Tuning**: Empirical studies on optimal staleness values
- **Memory Compression**: Techniques for frequently accessed nodes

### 5. MCP Protocol & Architecture
- **MCP Server Best Practices**: Authentication, error handling, streaming responses
- **Session State Management**: Efficient storage and restoration of conversation state
- **Async Operations**: Implementing non-blocking retrieval for < 200ms response times
- **Connection Resilience**: Handling Neo4j connection failures and recovery

### 6. Performance & Scalability
- **Incremental Index Updates**: Strategies for real-time index maintenance
- **Caching Layers**: Redis vs in-memory caching for embeddings
- **Batch Processing**: Optimal batch sizes for embedding generation
- **Memory Eviction Policies**: LRU vs LFU for inactive node removal

### 7. Advanced Algorithms
- **Conflict Resolution**: Strategies for handling contradictory memories
- **Adaptive Learning**: Online learning algorithms for weight adjustment
- **Memory Clustering**: Graph clustering algorithms for bulk operations
- **Relationship Strength Learning**: Dynamic edge weight adjustment

### Priority Research Areas (Start with these)
1. **Neo4j Vector Index Configuration** - Critical for basic functionality
2. **Embedding Model Selection & Benchmarking** - Core to semantic search quality
3. **Hybrid Scoring Algorithm Design** - Essential for relevance
4. **MCP Server Architecture Patterns** - Foundation for implementation
5. **Token Counting & Context Management** - Prevents context overflow

## Implementation Phases

### Phase 1: Core Infrastructure
- Set up Neo4j database
- Implement basic node/edge schema
- Create embedding pipeline
- Build basic MCP server

### Phase 2: Retrieval System
- Implement vector similarity search
- Add graph traversal algorithms
- Create hybrid scoring system
- Build context tracking

### Phase 3: Advanced Features
- Add memory compression for frequently accessed nodes
- Implement adaptive weight tuning
- Create memory importance propagation
- Add relationship strength learning

### Phase 4: Optimization
- Performance profiling and optimization
- Caching strategies
- Batch processing for embeddings
- Incremental index updates

## Configuration

### Tunable Parameters

```yaml
memory_config:
  embedding_model: "all-MiniLM-L6-v2"
  vector_dimensions: 384
  
relevance_weights:
  semantic: 0.4
  topological: 0.3
  temporal: 0.2
  importance: 0.1
  
retrieval:
  anchor_count: 3
  traversal_depth: 2
  default_token_budget: 2000
  staleness_threshold: 4096
  
temporal:
  decay_lambda: 0.0001
```

## Performance Considerations

### Scalability
- Neo4j handles millions of nodes efficiently
- HNSW index provides O(log n) search complexity
- Graph traversal limited by configurable depth

### Memory Usage
- Store embeddings efficiently using quantization
- Implement memory eviction for inactive nodes
- Use lazy loading for node content

### Response Time
- Target < 200ms for retrieval operations
- Pre-compute expensive calculations
- Use async operations where possible

## Future Enhancements

1. **Adaptive Learning**: Adjust relevance weights based on user feedback
2. **Memory Clustering**: Group related memories for bulk operations
3. **Conflict Resolution**: Handle contradictory memories intelligently
4. **Memory Decay**: Implement forgetting mechanisms for outdated information
5. **Multi-modal Support**: Extend to handle images, code snippets, etc.

## Success Metrics

- Retrieval precision: > 80% relevant memories in top results
- Context efficiency: 50% reduction in loaded tokens vs. naive approach
- Response time: < 200ms for typical queries
- Memory freshness: > 90% of stale memories refreshed when relevant