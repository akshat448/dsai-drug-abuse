## Pipeline Architecture

```
1. Chunking    → chunk_json_segments.py    → rag_dataset.jsonl
2. Embedding   → chroma_ingest.py          → ChromaDB (with BGE-M3)
3. Retrieval   → chroma_query.py           → Search results
```

## Setup

### 1. Install Dependencies

```bash
pip install chromadb sentence-transformers transformers torch tqdm
```

### 2. Verify Data

Ensure processed JSON files exist in `output/processed/`:
```bash
ls output/processed/*.json
```

## Usage

### Step 1: Create Chunks from JSON Segments

```bash
python -m rag_pipeline.chunker.chunk_json_segments
```

This will:
- Read all JSON files from `output/processed/`
- Chunk `original_text` into ~180 token segments
- Save to `rag_pipeline/output/rag_dataset.jsonl`

**Output**: `rag_pipeline/output/rag_dataset.jsonl`

---

### Step 2: Ingest Chunks into ChromaDB

```bash
python -m rag_pipeline.vectordb.chroma_ingest
```

Options:
```bash
# Reset collection before ingestion
python -m rag_pipeline.vectordb.chroma_ingest --reset
```

This will:
- Load chunks from `rag_dataset.jsonl`
- Embed using BGE-M3
- Store in ChromaDB at `rag_pipeline/output/chroma_db/`

---

### Step 3: Test Retrieval

```bash
python rag_pipeline/main_test_retrieval.py --q "how to reduce cravings at night"
```

Options:
```bash
# Top 10 results
python rag_pipeline/main_test_retrieval.py --q "family support" --top_k 10

# Filter by segment type
python rag_pipeline/main_test_retrieval.py --q "coping strategies" --segment_type "coping"

# Include unsafe content
python rag_pipeline/main_test_retrieval.py --q "relapse triggers" --include_unsafe

# Minimum similarity score
python rag_pipeline/main_test_retrieval.py --q "withdrawal symptoms" --min_score 0.7
```

## Example Queries

```bash
# Coping strategies
python rag_pipeline/main_test_retrieval.py --q "how do people manage cravings"

# Family support
python rag_pipeline/main_test_retrieval.py --q "role of family in recovery"

# Triggers and relapse
python rag_pipeline/main_test_retrieval.py --q "what causes relapse"

# Technology use
python rag_pipeline/main_test_retrieval.py --q "digital tools for recovery"
```

## Configuration

Edit `rag_pipeline/config/settings.py` to customize:

- `CHUNK_TOKEN_TARGET`: Target tokens per chunk (default: 180)
- `EMBED_MODEL`: Embedding model (default: "BAAI/bge-m3")
- `COLLECTION_NAME`: ChromaDB collection name

## Data Flow

```
JSON Files (output/processed/)
    ↓
chunk_json_segments.py
    ↓
rag_dataset.jsonl
    ↓
chroma_ingest.py (+ BGE-M3 embeddings)
    ↓
ChromaDB (rag_pipeline/output/chroma_db/)
    ↓
chroma_query.py
    ↓
Search Results
```

## Output Structure

### rag_dataset.jsonl
Each line contains:
```json
{
  "chunk_id": "uuid",
  "chunk_text": "...",
  "tokens": 180,
  "parent_segment_id": "segment_1",
  "source_file": "Interview.docx",
  "language": "en",
  "generalized_summary": "...",
  "tags": ["coping", "family_support"],
  "segment_type": "coping",
  "safety": "safe",
  "metadata": {...}
}
```

### Query Results
```python
{
  "chunk_id": "uuid",
  "chunk_text": "...",
  "score": 0.85,
  "source_file": "Interview.docx",
  "segment_type": "coping",
  "tags": ["coping", "triggers"],
  "safety": "safe",
  "tokens": 180
}
```

## Performance

- **Embedding speed**: ~100 chunks/sec (GPU) or ~20 chunks/sec (CPU)
- **Query latency**: <100ms for top-5 retrieval
- **Storage**: ~1MB per 1000 chunks

## Troubleshooting

### "No chunks found"
Run chunking first:
```bash
python -m rag_pipeline.chunker.chunk_json_segments
```

### "Collection is empty"
Run ingestion:
```bash
python -m rag_pipeline.vectordb.chroma_ingest --reset
```

### GPU out of memory
Set smaller batch size in `chroma_ingest.py`:
```python
ingest_chunks(chunks, batch_size=32)  # Reduce from 100
```