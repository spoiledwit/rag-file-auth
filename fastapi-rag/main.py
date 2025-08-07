"""
FastAPI RAG Application - Using Django's exact processing pipeline
"""

import os
import sys
import uuid
import time
import tempfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

# Import Django's exact modules
from scripts.universal_extractor import UniversalTextExtractor
from rag_utils import (
    ingest_new_text_file,
    rag_chat,
    rag_chat_flexible,
    add_documents_to_system
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Document Processing RAG API",
    description="Process documents and answer questions using the same pipeline as Django",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global extractor instance (like Django)
extractor = None

def get_extractor():
    """Get or create UniversalTextExtractor instance"""
    global extractor
    if extractor is None:
        extractor = UniversalTextExtractor(
            output_dir="extracted_texts",
            use_ocr=True,
            enable_gpu=True
        )
        logger.info("UniversalTextExtractor initialized")
    return extractor

# Request/Response models
class ProcessFileResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    extracted_text_path: str
    rag_ingested: bool
    metadata: Dict[str, Any]

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float
    num_docs_retrieved: int

class ProcessAndQueryResponse(BaseModel):
    doc_id: str
    filename: str
    answer: str
    sources: list
    confidence: float
    extracted_text_path: str
    processing_time: float
    metadata: Dict[str, Any]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FastAPI RAG Document Processor",
        "description": "Using Django's exact processing pipeline",
        "endpoints": {
            "/process": "Process a document (same as Django's submit_file)",
            "/query": "Query the RAG system (same as Django's ask_rag_question)",
            "/process_and_query": "Process file and immediately query it"
        }
    }

@app.post("/process", response_model=ProcessFileResponse)
async def process_file(
    file: UploadFile = File(...),
    category: str = Form(default="general")
):
    """
    Process a document file using Django's exact pipeline:
    1. Extract text using UniversalTextExtractor
    2. Save extracted text to file
    3. Ingest into RAG system
    """
    try:
        # Generate unique identifier (same as Django)
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Get original filename
        original_filename = file.filename
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        try:
            # Initialize extractor (same as Django)
            extractor = get_extractor()
            
            # Process the file (same as Django views.py line 422)
            logger.info(f"Processing file: {original_filename}")
            extraction_results = extractor.process_file(temp_file_path)
            
            if 'error' in extraction_results:
                raise ValueError(f"Extraction failed: {extraction_results['error']}")
            
            # Save extracted text to a .txt file (same as Django views.py line 427-428)
            output_filename = f"{unique_id}_{original_filename}_extracted_text.txt"
            output_file_path = extractor.save_results(extraction_results, output_filename=output_filename)
            logger.info(f"Extracted text saved to: {output_file_path}")
            
            # RAG Integration (same as Django views.py line 431-440)
            rag_success = False
            try:
                doc_id = f"doc_{unique_id}"
                rag_success = ingest_new_text_file(output_file_path, doc_id)
                if rag_success:
                    logger.info(f"Successfully ingested document into RAG: {doc_id}")
                else:
                    logger.warning(f"Failed to ingest document into RAG: {doc_id}")
            except Exception as rag_error:
                logger.error(f"RAG ingestion failed: {rag_error}")
            
            # Clean up temporary files (same as Django)
            extractor.cleanup_temp_files()
            logger.info("Cleaned up temporary files and images")
            
            # Return response
            return ProcessFileResponse(
                doc_id=doc_id,
                filename=original_filename,
                status="completed",
                extracted_text_path=output_file_path,
                rag_ingested=rag_success,
                metadata={
                    "pages": extraction_results.get('pages', 0),
                    "images_found": extraction_results.get('images_found', 0),
                    "images_processed": extraction_results.get('images_processed', 0),
                    "method": extraction_results.get('method', 'unknown'),
                    "file_type": extraction_results.get('file_type', 'unknown'),
                    "text_length": len(extraction_results.get('text', '')),
                    "category": category
                }
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"File processing failed: {str(e)}", exc_info=True)
        
        # Clean up on error (same as Django)
        try:
            if 'extractor' in locals():
                extractor.cleanup_temp_files()
                logger.info("Cleaned up temporary files after error")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File processing failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system (same as Django's ask_rag_question)
    """
    try:
        if not request.question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No question provided"
            )
        
        # Get response from RAG system (same as Django views.py line 550)
        response = rag_chat(request.question)
        
        return QueryResponse(
            answer=response.get('answer', 'No answer available'),
            sources=response.get('doc_sources', []),
            confidence=response.get('evaluation', {}).get('overall_score', 0.0),
            num_docs_retrieved=response.get('num_docs_retrieved', 0)
        )
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/process_and_query", response_model=ProcessAndQueryResponse)
async def process_and_query(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    category: str = Form(default="general")
):
    """
    Process a file and immediately query it with the given prompt
    Combines Django's submit_file and ask_rag_question
    """
    start_time = time.time()
    
    try:
        # Generate unique identifier
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        doc_id = f"doc_{unique_id}"
        
        # Get original filename
        original_filename = file.filename
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(original_filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_file_path = tmp_file.name
        
        try:
            # Initialize extractor
            extractor = get_extractor()
            
            # Process the file
            logger.info(f"Processing file: {original_filename}")
            extraction_results = extractor.process_file(temp_file_path)
            
            if 'error' in extraction_results:
                raise ValueError(f"Extraction failed: {extraction_results['error']}")
            
            # Save extracted text
            output_filename = f"{unique_id}_{original_filename}_extracted_text.txt"
            output_file_path = extractor.save_results(extraction_results, output_filename=output_filename)
            logger.info(f"Extracted text saved to: {output_file_path}")
            
            # Ingest into RAG
            rag_success = False
            try:
                rag_success = ingest_new_text_file(output_file_path, doc_id)
                if rag_success:
                    logger.info(f"Successfully ingested document into RAG: {doc_id}")
                else:
                    logger.warning(f"Failed to ingest document into RAG: {doc_id}")
            except Exception as rag_error:
                logger.error(f"RAG ingestion failed: {rag_error}")
                raise ValueError(f"Failed to ingest document for querying: {rag_error}")
            
            # Clean up temporary files
            extractor.cleanup_temp_files()
            
            # Now query the ingested document
            logger.info(f"Querying with prompt: {prompt}")
            response = rag_chat(prompt)
            
            processing_time = time.time() - start_time
            
            return ProcessAndQueryResponse(
                doc_id=doc_id,
                filename=original_filename,
                answer=response.get('answer', 'No answer available'),
                sources=response.get('doc_sources', []),
                confidence=response.get('evaluation', {}).get('overall_score', 0.0),
                extracted_text_path=output_file_path,
                processing_time=processing_time,
                metadata={
                    "pages": extraction_results.get('pages', 0),
                    "images_found": extraction_results.get('images_found', 0),
                    "images_processed": extraction_results.get('images_processed', 0),
                    "method": extraction_results.get('method', 'unknown'),
                    "file_type": extraction_results.get('file_type', 'unknown'),
                    "text_length": len(extraction_results.get('text', '')),
                    "category": category,
                    "chunks_retrieved": response.get('num_docs_retrieved', 0)
                }
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Process and query failed: {str(e)}", exc_info=True)
        
        # Clean up on error
        try:
            if 'extractor' in locals():
                extractor.cleanup_temp_files()
                logger.info("Cleaned up temporary files after error")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Process and query failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "extractor_initialized": extractor is not None
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )