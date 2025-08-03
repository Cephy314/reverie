# Reverie Local-First Implementation Plan (LLM-Agentic Development)

## Overview

This plan is specifically designed for LLM-based development (Claude/Gemini) of a local-first memory MCP system. It accounts for LLM strengths and weaknesses, providing exact prompts and guardrails for successful implementation.

## System Architecture

- **Local-first**: All data stored in `.reverie/` directory per project
- **Zero external dependencies**: No cloud services, API keys, or external databases
- **Target scale**: 10K-100K memories per project
- **Tech stack**: LanceDB (embedded), FastAPI, sentence-transformers

## 🔴 CRITICAL PREFLIGHT CHECKLIST (Human-Only Tasks)

**MUST BE COMPLETED BEFORE ANY LLM CODING**

### 1. Create Project Structure
```bash
mkdir -p reverie/{core,api,cli,tests,models}
touch reverie/__init__.py
touch reverie/{core,api,cli,models,tests}/__init__.py
```

### 2. Create pyproject.toml with Pinned Dependencies
```toml
[tool.poetry]
name = "reverie"
version = "0.1.0"
description = "Local-first memory system for LLM projects"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.9"
lancedb = "0.6.0"
fastapi = "0.110.0"
uvicorn = {extras = ["standard"], version = "0.27.0"}
sentence-transformers = "2.6.1"
pydantic = "2.6.4"
pydantic-settings = "2.2.1"
httpx = "0.27.0"
click = "8.1.7"
PyYAML = "6.0.1"

[tool.poetry.group.dev.dependencies]
pytest = "^8.0.0"
pytest-asyncio = "^0.23.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 3. Create Configuration Module
Create `reverie/config.py`:
```python
from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    project_id: str = os.getenv("REVERIE_PROJECT_ID", "default")
    db_path_template: str = str(Path.home() / ".reverie" / "{project_id}" / "memories.lance")
    embedding_model: str = "all-MiniLM-L6-v2"
    default_limit: int = 5
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    
    @property
    def db_path(self) -> Path:
        return Path(self.db_path_template.format(project_id=self.project_id))
    
settings = Settings()
```

### 4. Install Dependencies
```bash
poetry install
```

## 📋 PHASE-BY-PHASE IMPLEMENTATION

---

## Phase 1: Data Models & Contracts (2-3 hours)

### Objective
Define all data structures used throughout the application.

### EXACT PROMPT for LLM:
```
Create reverie/models.py with these EXACT Pydantic models:

1. Memory model:
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Literal
import uuid

class Memory(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    embedding: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_file: Optional[str] = None
    memory_type: Literal['code_snippet', 'error', 'decision', 'todo', 'observation'] = 'observation'
    tags: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }
```

2. MemoryInput model (for API input) - same fields EXCEPT: id, embedding, timestamp

3. SearchResult model:
   - memory: Memory
   - score: float
   - distance: float

Include all necessary imports. Use Field with default_factory for mutable defaults.
```

### Review Checklist:
- [ ] All imports are present and correct
- [ ] UUID generation uses .hex for string ID
- [ ] Datetime uses UTC
- [ ] No overly complex validators
- [ ] JSON serialization configured

### Anti-patterns to Avoid:
- ❌ Complex inheritance hierarchies
- ❌ Custom validators unless absolutely necessary
- ❌ Abstract base classes
- ❌ Overly nested models

---

## Phase 2: Database Layer (4-8 hours) - HIGHEST RISK

### Objective
Implement LanceDB integration with proper error handling and state management.

### EXACT PROMPT for LLM:
```
Create reverie/core/database.py implementing MemoryDB class:

Requirements:
1. Import: from reverie.models import Memory
2. Import: from reverie.config import settings
3. Use pathlib.Path for all path operations

Constructor (__init__):
- Accept project_id: str parameter
- Calculate db path: Path(settings.db_path_template.format(project_id=project_id))
- Create parent directories with path.parent.mkdir(parents=True, exist_ok=True)
- Import lancedb and connect: self.db = lancedb.connect(str(path.parent))
- Check if "memories" table exists in self.db.table_names()
- If not exists: create table with empty data matching Memory schema
- Store table reference: self.table = self.db.open_table("memories")

Method: add(self, memory: Memory) -> str:
- Convert memory to dict: memory_dict = memory.model_dump(exclude_none=True)
- Extract embedding: embedding = memory_dict.pop('embedding')
- Prepare data in LanceDB format: data = [memory_dict]
- If embedding exists: self.table.add(data, embeddings=[embedding])
- Else: self.table.add(data)
- Return memory.id

Method: search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
- Start with: query = self.table.search(query_vector).limit(limit)
- If filters provided:
  - Build WHERE clause from filters (e.g., filters={'memory_type': 'decision'} -> "memory_type = 'decision'")
  - Apply: query = query.where(where_clause)
- Execute: results = query.to_list()
- Return results

Include proper error handling and logging.
```

### Critical Review Points:
- [ ] Path handling uses pathlib consistently
- [ ] Table creation logic is correct
- [ ] Vector dimensions are handled properly
- [ ] Filter string construction is SQL-injection safe
- [ ] Connection is reused, not recreated

### Common LLM Mistakes:
- Creating new connections on every operation
- Mixing string paths and Path objects
- Incorrect table creation syntax
- Not handling the case where table already exists

---

## Phase 3: Embedding Service (2-3 hours)

### Objective
Create a singleton embedding service for efficient model usage.

### EXACT PROMPT for LLM:
```
Create reverie/core/embeddings.py with EmbeddingService class:

```python
from sentence_transformers import SentenceTransformer
from typing import List
import functools
from reverie.config import settings

class EmbeddingService:
    def __init__(self):
        # Load model ONCE in initialization
        self.model = SentenceTransformer(settings.embedding_model)
    
    def embed(self, text: str) -> List[float]:
        # Single text embedding
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Batch embedding for efficiency
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return [emb.tolist() for emb in embeddings]

# Singleton pattern using lru_cache
@functools.lru_cache(maxsize=1)
def get_embedder() -> EmbeddingService:
    return EmbeddingService()
```

Important:
- Model should be loaded ONCE
- Use convert_to_tensor=False for CPU efficiency
- Always convert numpy arrays to Python lists
```

### Review Checklist:
- [ ] Model loaded only once
- [ ] Proper singleton pattern
- [ ] Numpy arrays converted to lists
- [ ] No GPU assumptions

---

## Phase 4: FastAPI Application (3-4 hours)

### Objective
Create REST API with proper dependency injection and error handling.

### EXACT PROMPT for LLM:
```
Create reverie/api/main.py with FastAPI app:

```python
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from reverie.models import Memory, MemoryInput, SearchResult
from reverie.core.database import MemoryDB
from reverie.core.embeddings import get_embedder
from reverie.config import settings

app = FastAPI(title="Reverie Memory API")
executor = ThreadPoolExecutor(max_workers=4)

# Dependency injection
def get_db() -> MemoryDB:
    return MemoryDB(settings.project_id)

# Endpoints
@app.post("/memory", response_model=dict)
async def add_memory(
    memory_input: MemoryInput,
    db: MemoryDB = Depends(get_db)
):
    # Get embedder singleton
    embedder = get_embedder()
    
    # Generate embedding in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    embedding = await loop.run_in_executor(
        executor, 
        embedder.embed, 
        memory_input.content
    )
    
    # Create full memory object
    memory = Memory(
        **memory_input.model_dump(),
        embedding=embedding
    )
    
    # Add to database
    memory_id = await loop.run_in_executor(executor, db.add, memory)
    
    return {"id": memory_id, "message": "Memory added successfully"}

@app.get("/search", response_model=List[SearchResult])
async def search_memories(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=5, ge=1, le=50),
    memory_type: Optional[str] = None,
    tags: Optional[List[str]] = Query(default=None),
    db: MemoryDB = Depends(get_db)
):
    embedder = get_embedder()
    loop = asyncio.get_event_loop()
    
    # Generate query embedding
    query_embedding = await loop.run_in_executor(
        executor, 
        embedder.embed, 
        q
    )
    
    # Build filters
    filters = {}
    if memory_type:
        filters['memory_type'] = memory_type
    if tags:
        filters['tags'] = tags  # LanceDB handles list filtering
    
    # Search database
    results = await loop.run_in_executor(
        executor,
        db.search,
        query_embedding,
        limit,
        filters if filters else None
    )
    
    # Format results
    search_results = []
    for r in results:
        # Calculate distance from _distance field if present
        distance = r.pop('_distance', 0.0)
        score = 1.0 - distance  # Convert distance to similarity score
        
        # Create Memory object from result
        memory = Memory(**r)
        search_results.append(SearchResult(
            memory=memory,
            score=score,
            distance=distance
        ))
    
    return search_results

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "project": settings.project_id,
        "embedding_model": settings.embedding_model
    }

# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
```

Key requirements:
- Use ThreadPoolExecutor for blocking operations
- Proper dependency injection with Depends()
- Convert LanceDB results to proper models
- Handle filters properly
```

### Review Checklist:
- [ ] All blocking operations in thread pool
- [ ] Proper dependency injection
- [ ] Error handling implemented
- [ ] Query parameters validated
- [ ] Results properly formatted

---

## Phase 5: CLI Implementation (2-3 hours)

### Objective
Create simple CLI that communicates with the API.

### EXACT PROMPT for LLM:
```
Create reverie/cli/main.py using Click:

```python
import click
import httpx
import json
import yaml
from pathlib import Path
from typing import Optional, List
from reverie.config import settings

API_URL = f"http://{settings.api_host}:{settings.api_port}"

@click.group()
def cli():
    """Reverie - Local-first memory system for LLM projects"""
    pass

@cli.command()
@click.option('--project', '-p', default='default', help='Project ID')
def init(project: str):
    """Initialize a new Reverie project"""
    config_dir = Path.home() / '.reverie' / project
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = config_dir / 'config.yaml'
    config_data = {
        'project_id': project,
        'embedding_model': 'all-MiniLM-L6-v2',
        'default_limit': 5
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    click.echo(f"Initialized Reverie project '{project}' at {config_dir}")

@cli.command()
@click.argument('content')
@click.option('--type', '-t', 'memory_type', 
              type=click.Choice(['code_snippet', 'error', 'decision', 'todo', 'observation']),
              default='observation')
@click.option('--source', '-s', 'source_file', help='Source file path')
@click.option('--tags', '-g', multiple=True, help='Tags for the memory')
def add(content: str, memory_type: str, source_file: Optional[str], tags: tuple):
    """Add a new memory"""
    data = {
        'content': content,
        'memory_type': memory_type,
        'tags': list(tags)
    }
    if source_file:
        data['source_file'] = source_file
    
    try:
        response = httpx.post(f"{API_URL}/memory", json=data)
        response.raise_for_status()
        result = response.json()
        click.echo(f"✓ Memory added with ID: {result['id']}")
    except httpx.HTTPError as e:
        click.echo(f"✗ Error: {e}", err=True)

@cli.command()
@click.argument('query')
@click.option('--limit', '-l', default=5, help='Number of results')
@click.option('--type', '-t', 'memory_type', help='Filter by memory type')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def search(query: str, limit: int, memory_type: Optional[str], output_json: bool):
    """Search memories"""
    params = {'q': query, 'limit': limit}
    if memory_type:
        params['memory_type'] = memory_type
    
    try:
        response = httpx.get(f"{API_URL}/search", params=params)
        response.raise_for_status()
        results = response.json()
        
        if output_json:
            click.echo(json.dumps(results, indent=2))
        else:
            if not results:
                click.echo("No memories found.")
                return
            
            for i, result in enumerate(results, 1):
                memory = result['memory']
                score = result['score']
                click.echo(f"\n--- Result {i} (score: {score:.3f}) ---")
                click.echo(f"Type: {memory['memory_type']}")
                click.echo(f"Content: {memory['content']}")
                if memory.get('tags'):
                    click.echo(f"Tags: {', '.join(memory['tags'])}")
                if memory.get('source_file'):
                    click.echo(f"Source: {memory['source_file']}")
                    
    except httpx.HTTPError as e:
        click.echo(f"✗ Error: {e}", err=True)

if __name__ == '__main__':
    cli()
```

Requirements:
- Keep it SIMPLE - just format and forward requests
- No direct database access
- Nice formatting for search results
- Handle errors gracefully
```

### Review Checklist:
- [ ] Uses httpx for API calls
- [ ] Proper error handling
- [ ] Clean output formatting
- [ ] No complex state management

---

## Phase 6: Integration Tests (3-6 hours)

### Objective
Verify the complete system works end-to-end.

### EXACT PROMPT for LLM:
```
Create tests/test_integration.py:

```python
import pytest
from fastapi.testclient import TestClient
import tempfile
import shutil
from pathlib import Path

from reverie.api.main import app
from reverie.core.database import MemoryDB
from reverie.config import settings

@pytest.fixture
def test_project_id():
    return "test_project"

@pytest.fixture
def test_db_path(test_project_id):
    # Create temporary directory for test database
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir) / test_project_id
    # Cleanup after test
    shutil.rmtree(temp_dir)

@pytest.fixture
def client(test_db_path, monkeypatch):
    # Monkey patch settings to use test database
    monkeypatch.setattr(settings, 'project_id', 'test_project')
    monkeypatch.setattr(settings, 'db_path_template', str(test_db_path / "memories.lance"))
    
    with TestClient(app) as client:
        yield client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_add_and_search_memory(client):
    # Add a memory
    memory_data = {
        "content": "Python uses duck typing for polymorphism",
        "memory_type": "observation",
        "tags": ["python", "typing", "oop"]
    }
    
    add_response = client.post("/memory", json=memory_data)
    assert add_response.status_code == 200
    memory_id = add_response.json()["id"]
    assert memory_id is not None
    
    # Search for the memory
    search_response = client.get("/search", params={"q": "what typing system does Python use"})
    assert search_response.status_code == 200
    
    results = search_response.json()
    assert len(results) > 0
    
    # Verify the added memory is in results
    found = False
    for result in results:
        if result["memory"]["content"] == memory_data["content"]:
            found = True
            assert result["score"] > 0.5  # Should have good similarity
            break
    
    assert found, "Added memory not found in search results"

def test_search_with_filters(client):
    # Add multiple memories with different types
    memories = [
        {"content": "Fix null pointer exception", "memory_type": "error"},
        {"content": "Implement user authentication", "memory_type": "todo"},
        {"content": "Use dependency injection pattern", "memory_type": "decision"}
    ]
    
    for memory in memories:
        client.post("/memory", json=memory)
    
    # Search with type filter
    response = client.get("/search", params={
        "q": "implementation",
        "memory_type": "todo"
    })
    
    assert response.status_code == 200
    results = response.json()
    
    # Should only return todo items
    for result in results:
        assert result["memory"]["memory_type"] == "todo"

def test_empty_search(client):
    # Search in empty database
    response = client.get("/search", params={"q": "nonexistent query"})
    assert response.status_code == 200
    assert response.json() == []
```

Key requirements:
- Use fixtures for test isolation
- Test the complete flow from API to database
- Verify search actually returns relevant results
- Clean up test data after each run
```

### Review Checklist:
- [ ] Proper test isolation with fixtures
- [ ] Tests actual semantic search quality
- [ ] Cleanup after tests
- [ ] Tests error cases

---

## ⏱️ REALISTIC TIME ESTIMATES

| Phase | LLM Coding Time | Human Review Time | Total Time |
|-------|----------------|-------------------|------------|
| Preflight Setup | 0 hours | 2 hours | 2 hours |
| Phase 1: Models | 1 hour | 1 hour | 2 hours |
| Phase 2: Database | 4 hours | 4 hours | 8 hours |
| Phase 3: Embeddings | 1 hour | 1 hour | 2 hours |
| Phase 4: API | 2 hours | 2 hours | 4 hours |
| Phase 5: CLI | 2 hours | 1 hour | 3 hours |
| Phase 6: Tests | 3 hours | 3 hours | 6 hours |
| **TOTAL** | **13 hours** | **14 hours** | **27 hours** |

**Realistic Timeline: 3-4 days** with active human supervision

## 🚨 LLM-SPECIFIC WARNINGS & MITIGATIONS

### 1. State Management
**Problem**: LLMs recreate connections/models on every function call
**Mitigation**: Force singleton patterns, explicit caching

### 2. Path Handling
**Problem**: LLMs mix strings and Path objects inconsistently
**Mitigation**: Specify "use pathlib.Path for ALL path operations"

### 3. Error Boundaries
**Problem**: LLMs either ignore errors or over-handle them
**Mitigation**: Specify exact exception types to catch

### 4. Concurrency Issues
**Problem**: LLMs don't understand embedded DB limitations
**Mitigation**: Explicitly state "no concurrent writes" in prompts

### 5. Configuration Hardcoding
**Problem**: LLMs hardcode values instead of using config
**Mitigation**: Force "from reverie.config import settings" in every prompt

### 6. Over-Engineering
**Problem**: LLMs add unnecessary complexity
**Mitigation**: Use words like "SIMPLE", "MINIMAL", "BASIC" repeatedly

### 7. Outdated APIs
**Problem**: LLMs use old syntax from training data
**Mitigation**: Provide exact, current API examples in prompts

## 🎯 SUCCESS CRITERIA

The implementation is successful when:

1. **Basic Operations Work**:
   ```bash
   reverie add "User authentication uses JWT tokens"
   reverie search "how does auth work"
   # Returns the JWT memory
   ```

2. **Performance Targets Met**:
   - Search latency < 200ms
   - Memory usage < 2GB for 100K memories
   - No external API calls

3. **Quality Metrics**:
   - All tests pass
   - No hardcoded values
   - Clean error messages

4. **Time Targets**:
   - Complete implementation in < 4 days
   - Each phase matches time estimates

## 📝 PROMPT ENGINEERING TIPS

1. **Be Extremely Specific**: Instead of "handle errors", say "catch LanceError and return 500 with message"

2. **Provide Examples**: Show exact import statements and function signatures

3. **Set Boundaries**: Use "ONLY", "EXACTLY", "MUST NOT" to prevent scope creep

4. **Reference Previous Work**: "Import Memory from reverie.models that we created earlier"

5. **Iterative Refinement**: Plan for 2-3 iterations per component

## 🚀 GETTING STARTED

1. Complete the preflight checklist
2. Use each phase's exact prompt
3. Review against the checklist before proceeding
4. Test each component in isolation
5. Run integration tests frequently

Remember: The key to success with LLM development is **extreme specificity** and **active human review** at each step.