import os
import logging
import tempfile
import time
import uuid
from typing import Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Import our processing modules
from universal_extractor import UniversalTextExtractor
from rag_utils import process_document_with_query, extract_analysis_data, parse_json_response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing Service",
    description="FastAPI serverless function for document OCR, embedding, and RAG processing",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Document Processing Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document-processing"}

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    query: str = Form(...),
    method: str = Form(default="hybrid"),
    top_k: int = Form(default=30)
):
    """
    Process a document with OCR, embedding generation, and RAG query processing.
    
    Args:
        file: The uploaded document file (PDF, DOCX, image, etc.)
        query: The user's question about the document
        method: Retrieval method - "semantic", "keyword", or "hybrid"
        top_k: Number of chunks to retrieve for context
        
    Returns:
        JSON response with answer, sources, and metadata
    """
    try:
        start_time = time.time()
        
        # Validate inputs
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        if method not in ['semantic', 'keyword', 'hybrid']:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'semantic', 'keyword', or 'hybrid'")
            
        if top_k < 1 or top_k > 100:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
        
        # Get file extension
        original_filename = file.filename or "document"
        file_extension = ""
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        
        logger.info(f"Processing file: {original_filename} with query: {query}")
        
        # Create temporary file to save uploaded content
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f'.{file_extension}' if file_extension else '.pdf'
        ) as temp_file:
            # Read and write uploaded file content
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        logger.info(f"Saved uploaded file to temporary location: {temp_file_path}")
        
        try:
            # Initialize text extractor
            extractor = UniversalTextExtractor(
                output_dir="temp_extracts",
                use_ocr=True,
                enable_gpu=True
            )
            
            # Extract text from document
            extraction_results = extractor.process_file(temp_file_path)
            
            if 'error' in extraction_results:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document processing failed: {extraction_results['error']}"
                )
            
            # Get extracted text
            document_text = extraction_results.get('text', '')
            if not document_text or len(document_text.strip()) < 10:
                raise HTTPException(
                    status_code=400,
                    detail="No readable text found in the document. The document might be empty or corrupted."
                )
            
            logger.info(f"Extracted {len(document_text)} characters of text from document")
            
            # Process document with RAG pipeline
            rag_result = process_document_with_query(
                document_text=document_text,
                query=query,
                method=method,
                top_k=top_k
            )
            
            if 'error' in rag_result:
                raise HTTPException(
                    status_code=500,
                    detail=f"RAG processing failed: {rag_result['error']}"
                )
            
            # Extract analysis data from AI response
            ai_response = rag_result.get('answer', 'No answer generated')
            accuracy_score, extracted_fields = extract_analysis_data(ai_response)
            
            # Clean up temporary files
            try:
                os.unlink(temp_file_path)
                extractor.cleanup_temp_files()
                logger.info("Cleaned up temporary files")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            
            # Format response
            response_data = {
                "success": True,
                "query": query,
                "answer": ai_response,
                "accuracy_score": accuracy_score,
                "extracted_fields": extracted_fields,
                "retrieval_method": rag_result.get('retrieval_method', method),
                "processing_time": processing_time,
                "chunks_processed": rag_result.get('chunks_processed', 0),
                "relevant_chunks": rag_result.get('relevant_chunks', 0),
                "num_docs_retrieved": rag_result.get('num_docs_retrieved', 0),
                "doc_sources": rag_result.get('doc_sources', []),
                "evaluation": rag_result.get('evaluation', {}),
                "document_info": {
                    "filename": original_filename,
                    "text_length": len(document_text),
                    "extraction_method": extraction_results.get('method', 'unknown'),
                    "pages": extraction_results.get('pages', 0),
                    "images_processed": extraction_results.get('images_processed', 0)
                }
            }
            
            logger.info(f"Successfully processed document in {processing_time:.2f}s")
            return JSONResponse(content=response_data)
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as processing_error:
            logger.error(f"Document processing failed: {str(processing_error)}", exc_info=True)
            
            # Clean up temporary files on error
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                if 'extractor' in locals():
                    extractor.cleanup_temp_files()
            except:
                pass
            
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(processing_error)}"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )

@app.post("/query-text")
async def query_text(
    text: str = Form(...),
    query: str = Form(...),
    method: str = Form(default="hybrid"),
    top_k: int = Form(default=30)
):
    """
    Process raw text with RAG query (no file upload/OCR needed).
    
    Args:
        text: The document text content
        query: The user's question about the text
        method: Retrieval method - "semantic", "keyword", or "hybrid"
        top_k: Number of chunks to retrieve for context
        
    Returns:
        JSON response with answer, sources, and metadata
    """
    try:
        start_time = time.time()
        
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text content cannot be empty")
            
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        if method not in ['semantic', 'keyword', 'hybrid']:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'semantic', 'keyword', or 'hybrid'")
            
        if top_k < 1 or top_k > 100:
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
        
        logger.info(f"Processing text query: {query}")
        
        # Process text with RAG pipeline
        rag_result = process_document_with_query(
            document_text=text,
            query=query,
            method=method,
            top_k=top_k
        )
        
        if 'error' in rag_result:
            raise HTTPException(
                status_code=500,
                detail=f"RAG processing failed: {rag_result['error']}"
            )
        
        # Extract analysis data from AI response
        ai_response = rag_result.get('answer', 'No answer generated')
        accuracy_score, extracted_fields = extract_analysis_data(ai_response)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format response
        response_data = {
            "success": True,
            "query": query,
            "answer": ai_response,
            "accuracy_score": accuracy_score,
            "extracted_fields": extracted_fields,
            "retrieval_method": rag_result.get('retrieval_method', method),
            "processing_time": processing_time,
            "chunks_processed": rag_result.get('chunks_processed', 0),
            "relevant_chunks": rag_result.get('relevant_chunks', 0),
            "num_docs_retrieved": rag_result.get('num_docs_retrieved', 0),
            "doc_sources": rag_result.get('doc_sources', []),
            "evaluation": rag_result.get('evaluation', {}),
            "document_info": {
                "text_length": len(text),
                "source": "direct_text_input"
            }
        }
        
        logger.info(f"Successfully processed text query in {processing_time:.2f}s")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in query_text: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )

if __name__ == "__main__":
    # For local development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)