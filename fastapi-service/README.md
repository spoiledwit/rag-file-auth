# Document Processing FastAPI Service

This FastAPI service handles document processing, OCR, embedding generation, and RAG (Retrieval-Augmented Generation) operations for the File-Auth-AI system.

## Features

- Document OCR and text extraction (PDF, DOCX, images)
- Embedding generation using BAAI/bge-large-en
- Semantic, keyword, and hybrid retrieval
- RunPod integration for LLM generation
- Serverless deployment ready

## Endpoints

### `/process-document` (POST)
Process a document file with OCR and answer a query about it.

**Parameters:**
- `file`: Document file (multipart/form-data)
- `query`: Question about the document
- `method`: Retrieval method - "semantic", "keyword", or "hybrid" (default: "hybrid")
- `top_k`: Number of chunks to retrieve (default: 30)

### `/query-text` (POST)
Process raw text content with a query (no file upload needed).

**Parameters:**
- `text`: Document text content
- `query`: Question about the text
- `method`: Retrieval method (default: "hybrid")
- `top_k`: Number of chunks to retrieve (default: 30)

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your RunPod API key
```

3. Run the service:
```bash
uvicorn main:app --reload
```

## Docker Deployment

1. Build the image:
```bash
docker build -t document-processor .
```

2. Run the container:
```bash
docker run -p 8000:8000 -e RUNPOD_API_KEY=your_key document-processor
```

## RunPod Deployment

This service is designed to be deployed as a serverless function on RunPod. The Dockerfile includes all necessary dependencies for document processing and ML operations.

## Environment Variables

- `RUNPOD_API_KEY`: Your RunPod API key (required)
- `RUNPOD_API_URL`: Custom RunPod endpoint (optional)
- `LOG_LEVEL`: Logging level (default: INFO)

## Response Format

All endpoints return a standardized JSON response:

```json
{
  "success": true,
  "query": "user question",
  "answer": "AI-generated answer in JSON format",
  "accuracy_score": 85.5,
  "extracted_fields": {...},
  "retrieval_method": "hybrid",
  "processing_time": 2.45,
  "chunks_processed": 15,
  "relevant_chunks": 8,
  "num_docs_retrieved": 8,
  "doc_sources": ["uploaded_doc"],
  "evaluation": {...},
  "document_info": {
    "filename": "document.pdf",
    "text_length": 5000,
    "extraction_method": "pymupdf",
    "pages": 3,
    "images_processed": 0
  }
}
```