# FastAPI RAG Document Processor

This FastAPI application uses **Django's exact processing pipeline** for document processing and RAG-based question answering.

## Features

- **Same processing pipeline as Django**:
  - Uses `UniversalTextExtractor` for text extraction (PDF, DOCX, images)
  - OCR with EasyOCR for image-heavy documents
  - Same RAG integration with Pinecone and hybrid search (semantic + keyword)
  - Same LLM models (BAAI/bge-small-en + Qwen/Qwen2.5-3B-Instruct)

## API Endpoints

### 1. `/process_and_query` - Main Endpoint
Upload a file and immediately query it:
```bash
curl -X POST "http://localhost:8001/process_and_query" \
  -F "file=@document.pdf" \
  -F "prompt=What is this document about?"
```

### 2. `/process` - Process File Only
Upload and process a document:
```bash
curl -X POST "http://localhost:8001/process" \
  -F "file=@document.pdf" \
  -F "category=general"
```

### 3. `/query` - Query RAG System
Query previously processed documents:
```bash
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your Pinecone API key
```

3. **Run the server:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Key Components

### From Django Project:
- `scripts/universal_extractor.py` - Text extraction (PDF, DOCX, OCR)
- `rag_utils.py` - RAG pipeline, embeddings, vector search
- Same models and configuration

### FastAPI Additions:
- `main.py` - FastAPI endpoints wrapping Django's pipeline
- Async file handling
- Pydantic models for request/response validation
- CORS support for frontend integration

## Processing Pipeline

1. **File Upload** → temp file storage
2. **Text Extraction** → UniversalTextExtractor (same as Django)
3. **Text Chunking** → LangChain splitter (300 chars, 50 overlap)
4. **Embedding Generation** → BAAI/bge-small-en
5. **Vector Storage** → Pinecone + BM25 index
6. **Query Processing** → Hybrid search (semantic + keyword)
7. **Answer Generation** → Qwen/Qwen2.5-3B-Instruct

## Example Usage

```python
import requests

# Process and query in one step
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/process_and_query',
        files={'file': f},
        data={'prompt': 'Summarize this document'}
    )

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Processing time: {result['processing_time']}s")
```

## Models & Storage

- **Models**: Cached locally after first download (same as Django)
- **Vector DB**: Pinecone (configure API key in .env)
- **File Storage**: Local temp files, extracted text saved to `extracted_texts/`

## Development

- FastAPI auto-docs: http://localhost:8001/docs
- Health check: http://localhost:8001/health
- Same processing quality as Django backend