# Entity Recognition & Concept Boundaries Research Findings

## Executive Summary

Through collaborative research with advanced AI models, we've developed a comprehensive theoretical framework for entity recognition and concept boundaries in the Reverie memory system. The core innovation is treating **propositions** (not entities) as the fundamental unit of memory, supported by a dual-graph architecture that separates personal memories from domain knowledge.

## Core Architectural Decisions

### 1. Dual Knowledge Graph Architecture

**Components:**
- **Personal Memory Graph**: User-specific, private, experiential knowledge
- **Domain Knowledge Graph**: Shared understanding of technology relationships

**Key Design: Semi-Permeable Membrane**
- Domain KG has curated stable core + dynamic extensions
- User assertions about self = high trust
- User assertions about technology = medium trust (requires validation)
- Prevents contamination while allowing growth

### 2. Proposition as Fundamental Unit

Instead of extracting isolated entities, we capture complete propositions:

**Example Transformation:**
- Input: "I prefer TypeScript over JavaScript for large projects"
- Traditional: Entities = [TypeScript, JavaScript, large projects]
- Proposition-based: `prefer(user, TypeScript, over: JavaScript, context: "large projects")`

### 3. Three-Layer Canonicalization

Each proposition exists at three levels:

1. **Surface Form**: "I love TypeScript"
2. **Normalized Form**: `prefer(user, TypeScript, degree: high)`
3. **Semantic Form**: `positive_sentiment(user, TypeScript)`

This enables flexible querying at different abstraction levels while preserving the user's exact phrasing.

## Key Innovations

### 1. KG-Aware Extraction Pipeline

**Four-Stage Process:**
1. **Entity Mention Recognition**: Fast identification of candidate strings
2. **Entity Linking & Disambiguation**: Map to canonical Domain KG entities
3. **Proposition Extraction**: LLM generates structured propositions with KG context
4. **Reconciliation & Storage**: Check contradictions, update evolution chains

**Example Flow:**
```
"Moving from Rocket to Actix" 
→ Entities: ["Rocket", "Actix"] 
→ Linked: [dkg:Rocket_Framework, dkg:Actix_Framework]
→ Propositions: [
    intends_to_use(user, Actix),
    believes(user, is_faster_than(Actix, Rocket))
]
```

### 2. Proposition Assembly Buffer

**Problem**: Complex ideas span multiple utterances
**Solution**: Ephemeral workspace for building propositions

**Example Conversation:**
- T1: "I'm looking at Rust web frameworks"
- T2: "Rocket seems nice for its type safety"  
- T3: "But I need something faster"
- T4: "Maybe Actix would be better"

**Buffer Evolution:**
- T1: `[explores(user, "Rust web frameworks")]`
- T2: `[... , has_sentiment(user, Rocket, positive, reason: type_safety)]`
- T3: `[... , has_requirement(user, performance)]`
- T4: `[prefers(user, Actix, over: Rocket, reason: performance)]` ← Committable

### 3. Memory Type Classification

**Four Types with Different Temporal Dynamics:**
1. **Persistent Facts**: "I have a CS degree" - version only on correction
2. **Evolving Preferences**: Tool choices - full evolution chain
3. **Contextual Opinions**: "This bug is frustrating" - timestamp-bound
4. **Learned Skills**: "I know React" - monotonic growth

### 4. Hypothesis-Driven Clarification

**Implicit Knowledge Handling:**
- Store inferences as hypotheses with confidence scores
- Use activation thresholds to surface for clarification
- Convert to explicit knowledge through user confirmation

**Activation Score Formula:**
```
Score = Confidence × Relevance × (log(Frequency) + 1)
```

## Practical Implementation Guidelines

### 1. Handling Predicate Synonyms

**Many-to-One Mapping via Embeddings:**
- Rich normalized predicates preserve nuance
- Map to small semantic ontology using embeddings
- Example: `love`, `enjoy`, `prefer` → `has_sentiment`

### 2. Thought-Thread Boundary Detection

**Multi-Signal Approach:**
- Entity coherence score between utterances
- Temporal gaps (>30s weak, >2min strong)
- Discourse markers ("BTW", "Another topic...")
- Proposition coherence via Domain KG distance

**Buffer State Machine:**
```
EMPTY → ACTIVE → SYNTHESIZING → COMMITTING → EMPTY
         ↓           ↓
      STALE     ABANDONED
```

### 3. Negative Space Tracking

**Distinguish Between:**
- **Absence of Evidence**: User hasn't mentioned backend → weak signal
- **Evidence of Absence**: "I never use ORMs" → strong signal

Only track high-confidence negative assertions to avoid incorrect inferences.

## Domain Knowledge Graph Bootstrapping

### Three-Phase Strategy

**Phase 1: Foundation (Week 1)**
- Extract programming language data from DBpedia
- Filter ConceptNet for software concepts
- Manually curate top 50 core relationships

**Phase 2: Enrichment (Week 2)**
- LLM extraction from Stack Overflow surveys
- Parse package manager dependencies
- Mine "awesome-*" lists for relationships

**Phase 3: Validation (Week 3)**
- Cross-reference multiple sources
- Apply confidence scoring
- Community review of controversial relationships

### Existing Resources to Leverage

- **ConceptNet**: General concepts including tech
- **DBpedia**: Structured Wikipedia data
- **Open source ontologies**: DOAP, SWO
- **Graph databases**: Neo4j, ArangoDB for implementation

## Design Trade-offs and Decisions

### 1. Granularity vs. Noise
- **Decision**: Context qualifiers ("large projects") aren't independent concepts
- **Rationale**: Prevents meaningless entity proliferation

### 2. Perfect Capture vs. Responsiveness  
- **Decision**: Buffer adds latency but improves quality
- **Rationale**: Complex ideas are worth the wait

### 3. User Mental Models vs. Objective Truth
- **Decision**: Track both as separate propositions
- **Rationale**: The gap enables educational opportunities

## Validation Criteria

### Conceptual Coherence
- ✓ All components have clear purpose
- ✓ Well-defined interactions
- ✓ No circular dependencies
- ✓ Graceful degradation paths

### Theoretical Soundness
- ✓ Based on cognitive science principles
- ✓ Leverages information retrieval best practices
- ✓ Assumptions are testable
- ✓ Evolution path is clear

### User Value Alignment
- ✓ Preserves user's voice and phrasing
- ✓ Captures complex, multi-turn ideas
- ✓ Respects context and nuance
- ✓ Enables both recall and reasoning

## Next Steps

1. **Prototype the Proposition Assembly Buffer** to validate multi-utterance synthesis
2. **Design Domain KG schema** based on bootstrapping strategy
3. **Create extraction prompt templates** for different proposition types
4. **Define reconciliation rules** for contradiction handling
5. **Build confidence scoring models** for hypothesis activation

## Key Insights

The shift from entity-centric to proposition-centric memory fundamentally changes how we think about knowledge representation. By maintaining both personal and domain knowledge graphs with different trust boundaries, we can build a system that:

- Understands what the user means, not just what they say
- Tracks how knowledge evolves over time
- Distinguishes between facts, preferences, and beliefs
- Learns from conversation flow, not just isolated statements

This research provides a solid theoretical foundation for building a memory system that truly augments human cognition in LLM conversations.