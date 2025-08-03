# Personal Memory System Research Report for Reverie

## Executive Summary

Reverie is a personal memory augmentation system that builds a user's knowledge graph incrementally through natural language interactions. Unlike traditional search systems, it starts empty and grows with each user interaction, creating a highly personalized context preservation system for LLM conversations.

## Core Use Case

1. **Memory Creation**: User types `/remember Don't use 'any' type in typescript`
2. **Automatic Processing**: System extracts entities (typescript, type safety) and infers relationships
3. **Contextual Recall**: User types `/recall typescript typing` and gets relevant memories plus related context
4. **Future Goal**: Automatic memory injection based on conversation context

## Architecture Overview

### 1. Hybrid NLP Pipeline

```python
# Three-stage processing for optimal speed and intelligence
class MemoryProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")  # Fast base NLP
        self.matcher = Matcher(self.nlp.vocab)    # Technical terms
        self.llm = LocalLLM("Llama-3.2-1B")      # Smart inference
```

**Performance Targets:**
- SpaCy extraction: 50ms
- Technical matching: 20ms  
- LLM enhancement: 150ms (only when needed)
- **Total: <300ms end-to-end**

### 2. Graph Schema Design

```cypher
// Node Types
(:Memory {
    id: "uuid",
    content: "user's actual text",
    embedding: [384-dim vector],
    created_at: timestamp,
    access_count: 0
})

(:Concept {
    name: "typescript",
    type: "technology|principle|preference",
    first_seen: timestamp,
    mention_count: 1
})

// Relationship Types with Metadata
-[:RELATES_TO {confidence: 0.8, source: "extracted"}]->
-[:UPDATES {reason: "preference_change"}]->
-[:CONTRADICTS {detected_at: timestamp}]->
-[:BROADER_THAN]->  // Concept hierarchies
```

### 3. Entity Extraction Strategy

**Stage 1: Fast NLP (SpaCy)**
- Named entity recognition
- Part-of-speech tagging
- Dependency parsing
- ~50ms latency

**Stage 2: Domain Matching**
- Technical term patterns
- Programming keywords
- Custom matchers
- ~20ms latency

**Stage 3: LLM Enhancement (Conditional)**
- Triggered when <3 entities found
- Local model for privacy
- Extracts implicit concepts
- ~150ms when used

### 4. Relationship Inference

**Automatic Relationships:**
```python
def infer_relationships(memory_text, entities):
    relationships = []
    
    # Co-occurrence in same memory
    for e1, e2 in combinations(entities, 2):
        relationships.append({
            'from': e1, 'to': e2,
            'type': 'RELATES_TO',
            'confidence': 0.8
        })
    
    # LLM-powered inference for deeper connections
    if len(entities) > 2:
        prompt = f"Infer relationships between: {entities}"
        llm_rels = local_llm.infer(prompt)
        # Add with lower confidence (0.5-0.7)
    
    return relationships
```

### 5. Memory Recall Algorithm

```python
def recall_memories(query, max_tokens=2000):
    # Multi-stage retrieval
    candidates = []
    
    # 1. Direct semantic search
    vector_matches = search_by_embedding(query)
    candidates.extend(vector_matches)
    
    # 2. Entity-based search
    entities = extract_entities(query)
    entity_matches = search_by_entities(entities)
    candidates.extend(entity_matches)
    
    # 3. Graph traversal (1-3 hops)
    for match in candidates[:5]:
        related = traverse_graph(match, max_hops=2)
        candidates.extend(related)
    
    # 4. Scoring and ranking
    scored = score_candidates(candidates, {
        'semantic_similarity': 0.4,
        'entity_overlap': 0.3,
        'recency': 0.2,
        'access_frequency': 0.1
    })
    
    # 5. Token-aware selection
    return select_within_budget(scored, max_tokens)
```

## Implementation Challenges & Solutions

### Challenge 1: Contradictory Memories

**Scenario**: User updates preferences over time

**Solution**: Version chain with CONTRADICTS/UPDATES relationships
```cypher
// Old memory: "Never use any type"
// New memory: "Use any for rapid prototyping"
CREATE (new)-[:CONTRADICTS]->(old)
CREATE (new)-[:UPDATES {reason: 'context_specific'}]->(old)
```

### Challenge 2: Ambiguous References

**Scenario**: "The component should be optimized"

**Solution**: Store with low confidence, use context during recall
```python
if ambiguity_score > 0.7:
    relationships = add_with_confidence(0.4)
    mark_for_clarification = True
```

### Challenge 3: Memory Overflow

**Scenario**: Too many relevant memories for context window

**Solution**: Token-aware selection
```python
def select_memories(candidates, available_tokens):
    selected = []
    used_tokens = 0
    
    for memory in sorted(candidates, key=lambda m: m.score, reverse=True):
        memory_tokens = count_tokens(memory)
        if used_tokens + memory_tokens <= available_tokens:
            selected.append(memory)
            used_tokens += memory_tokens
    
    return selected
```

## Performance Optimizations

### 1. Caching Strategy
```python
class MemoryCache:
    def __init__(self):
        self.entity_cache = LRUCache(maxsize=1000)
        self.embedding_cache = LRUCache(maxsize=500)
        self.recent_queries = deque(maxlen=50)
```

### 2. Progressive Enhancement
- Immediate: Store raw memory (<50ms)
- Async: Extract entities (<100ms)
- Background: Infer relationships (<200ms)
- Idle: LLM analysis and enrichment

### 3. Batch Processing
- Queue memories during active conversation
- Process in batch during idle periods
- Maintain read-after-write consistency

## Automatic Recall Design

### Context Monitoring
```python
class ContextMonitor:
    def analyze_conversation(self, message):
        # Extract concepts from current message
        concepts = extract_concepts(message)
        
        # Check relevance threshold
        relevant_memories = []
        for concept in concepts:
            memories = find_related_memories(concept)
            for memory in memories:
                if memory.relevance_score > 0.6:
                    relevant_memories.append(memory)
        
        # Limit injection
        return relevant_memories[:3]
```

### Injection Strategy
- Relevance threshold: 0.6
- Maximum 3 memories per response
- User-configurable aggressiveness
- Clear attribution when injected

## Privacy & Security

1. **Local Processing**: No cloud APIs for personal data
2. **Encrypted Storage**: Memories encrypted at rest
3. **User Control**: Full CRUD operations on memories
4. **Data Portability**: Export/import functionality

## Success Metrics

### Performance Metrics
- Memory creation: <300ms (including all processing)
- Recall query: <100ms for typical search
- Auto-injection decision: <50ms
- Graph traversal: <30ms for 2-hop search

### Quality Metrics
- Entity extraction accuracy: >80%
- Relationship inference precision: >70%
- Recall relevance: >85% user satisfaction
- False injection rate: <5%

### Scale Metrics
- Efficient up to 10,000 memories
- Sub-linear performance degradation
- Memory usage: <1GB for typical user

## Implementation Roadmap

### Phase 1: Core System (Weeks 1-3)
- Basic `/remember` and `/recall` commands
- Simple entity extraction with SpaCy
- Neo4j setup with basic schema
- Direct text matching for recall

### Phase 2: Intelligence Layer (Weeks 4-6)
- Local LLM integration
- Relationship inference
- Confidence scoring
- Contradiction handling

### Phase 3: Automatic Recall (Weeks 7-8)
- Context monitoring
- Relevance scoring
- Configurable injection
- User feedback loop

### Phase 4: Optimization (Weeks 9-10)
- Performance profiling
- Caching implementation
- Batch processing
- Progressive enhancement

## Technology Stack

### Required Components
```yaml
core:
  language: Python 3.11+
  framework: MCP SDK
  
nlp:
  spacy: "3.7+"
  model: "en_core_web_sm"
  
llm:
  model: "Llama-3.2-1B-Instruct"
  inference: "llama-cpp-python"
  
database:
  neo4j: "5.x"
  driver: "neo4j-python-driver"
  
embeddings:
  model: "all-MiniLM-L6-v2"
  library: "sentence-transformers"
  
caching:
  memory: "cachetools"
  persistent: "diskcache"
```

## Conclusion

This personal memory system architecture provides:

1. **Intelligent Understanding**: Hybrid NLP+LLM extracts meaning from user input
2. **Relationship Awareness**: Graph structure captures how concepts connect
3. **Evolution Support**: Handles changing preferences and contradictions
4. **Privacy First**: All processing happens locally
5. **Progressive Enhancement**: Starts simple, grows smarter over time

The system fights LLM context degradation by preserving user preferences and knowledge across conversations, ensuring relevant memories are always available when needed.