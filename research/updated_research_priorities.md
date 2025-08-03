# Updated Research Priorities for Reverie

## Overview

Based on completed research into Memory Semantics and Entity Recognition, we've made fundamental discoveries that shift our research priorities. The core innovations - propositions as memory units and dual-graph architecture - require us to reframe remaining research questions.

## Completed Research (No Longer Priority)

### ✓ Memory Semantics Fundamentals
- Graph structure mirrors human memory - VALIDATED
- Memory type distinctions are crucial - CONFIRMED
- Multi-factor relevance is essential - IMPLEMENTED
- Version chains for preferences - DESIGNED

### ✓ Entity Recognition Core Concepts  
- Shifted from entities to propositions - REVOLUTIONARY FINDING
- Taxonomy vs folksonomy - RESOLVED (hybrid approach)
- Granularity questions - ANSWERED (context qualifiers)
- Three-layer canonicalization - DESIGNED

## High Priority Research Areas

### 1. Proposition Dynamics & Graph Evolution
**New Focus**: How propositions interact and evolve in the dual-graph architecture

**Critical Questions:**
- How do proposition relationships differ from entity relationships?
- When should propositions in Personal Graph influence Domain KG?
- How to handle proposition conflicts across time?
- What's the lifecycle of a hypothesis proposition?

**Research Needed:**
- Proposition relationship taxonomy (supports, contradicts, refines, extends)
- Confidence propagation between related propositions
- Hypothesis promotion criteria and mechanisms
- Cross-graph influence patterns

### 2. Context Window Optimization for Propositions
**New Focus**: Injecting proposition chains vs individual memories

**Critical Questions:**
- How to summarize related propositions for injection?
- What's the cognitive load of proposition chains?
- How to maintain narrative coherence with propositions?
- When to inject hypotheses vs confirmed knowledge?

**Research Needed:**
- Proposition compression algorithms
- Narrative generation from proposition chains
- Token budgeting for complex propositions
- Context relevance scoring for multi-part ideas

### 3. Buffer Management & Conversation Flow
**New Focus**: Managing the Proposition Assembly Buffer lifecycle

**Critical Questions:**
- How to detect thought-thread boundaries reliably?
- When to commit vs abandon buffer contents?
- How to handle interrupted conversations?
- Multi-conversation buffer management?

**Research Needed:**
- Thought-thread boundary detection algorithms
- Buffer state persistence strategies
- Conversation resumption patterns
- Multi-modal conversation handling

### 4. Domain Knowledge Graph Dynamics
**New Focus**: Building and maintaining the shared Domain KG

**Critical Questions:**
- How to bootstrap without bias?
- When do user contributions promote to canonical?
- How to handle conflicting domain knowledge?
- Privacy-preserving aggregation methods?

**Research Needed:**
- Bootstrapping strategy validation
- Promotion criteria for provisional nodes
- Conflict resolution mechanisms
- Anonymization techniques for contributions

### 5. User Interaction Design for Dual Graphs
**New Focus**: Making the dual-graph + proposition model intuitive

**Critical Questions:**
- How to explain why certain memories appeared?
- When to surface hypotheses for confirmation?
- How to show proposition evolution?
- Visualization of memory connections?

**Research Needed:**
- Explanation generation for proposition retrieval
- Hypothesis surfacing strategies
- Evolution visualization methods
- Trust-building through transparency

## Medium Priority Research Areas

### 6. System State Management
**Enhanced Focus**: States now include buffer states and dual-graph coordination

**New Considerations:**
- Buffer state affects system state
- Domain KG maturity influences behavior
- Hypothesis confidence distribution indicates learning

### 7. Performance & Scaling
**New Challenges**: Dual graphs and proposition assembly

**Critical Areas:**
- Proposition synthesis latency
- Cross-graph query optimization
- Buffer memory management
- Domain KG growth limits

### 8. Privacy in Dual-Graph Architecture
**New Complexity**: Shared Domain KG introduces new challenges

**Key Questions:**
- Anonymization for Domain KG contributions
- Privacy of inferred propositions
- Selective sharing of proposition chains
- Right to deletion across graphs

## Deprioritized Research Areas

### Entity Extraction Techniques
- Superseded by proposition extraction
- Basic entity recognition sufficient for pipeline stage 1

### Simple Relationship Inference
- Replaced by complex proposition relationships
- Now handled by KG-aware pipeline

### Memory Decay Curves
- Less critical with proposition versioning
- Can use simple time-based scoring initially

## New Research Areas (Not in Original Plan)

### 1. Proposition Assembly Patterns
- Multi-utterance synthesis strategies
- Partial proposition handling
- Conversation repair mechanisms

### 2. Cross-Graph Consistency
- Maintaining coherence between graphs
- Trust boundary enforcement
- Update propagation patterns

### 3. Hypothesis Lifecycle Management
- Generation, testing, promotion, decay
- Confidence calibration over time
- User feedback integration

## Research Methodology Updates

### Phase 1: System Integration Research (Week 1)
- How components work with dual-graph architecture
- Proposition flow through the system
- Buffer integration patterns

### Phase 2: Prototype Validation (Week 2)
- Test proposition assembly buffer
- Validate dual-graph queries
- Measure synthesis latency

### Phase 3: User Studies (Week 3)
- Test explanation strategies
- Validate hypothesis surfacing
- Measure cognitive load

### Phase 4: Performance Optimization (Week 4)
- Cross-graph query optimization
- Buffer efficiency improvements
- Scaling simulations

## Success Criteria Updates

### Conceptual Coherence
- ✓ Proposition model is internally consistent
- ✓ Dual-graph boundaries are clear
- Buffer lifecycle is well-defined
- Cross-graph interactions are predictable

### Implementation Feasibility
- Proposition synthesis is fast enough (< 200ms)
- Domain KG bootstrapping is practical
- Buffer doesn't create memory bloat
- Privacy boundaries are enforceable

### User Value
- Propositions capture intent better than entities
- Hypotheses feel helpful, not intrusive
- Evolution tracking provides insights
- System builds trusted mental model

## Next Steps

1. **Immediate Priority**: Prototype the Proposition Assembly Buffer
2. **Critical Path**: Design Domain KG bootstrapping strategy
3. **User Validation**: Test hypothesis surfacing approaches
4. **Performance**: Benchmark proposition synthesis latency

## Key Insight

The shift to propositions and dual-graph architecture is more than an implementation detail - it fundamentally changes how we think about memory systems. Research priorities must align with this new understanding, focusing on the unique challenges and opportunities it creates.