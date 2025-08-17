import os
import logging
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Import our processing modules
from universal_extractor import UniversalTextExtractor
from rag_utils_simplified import process_and_store_document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Embedding Service",
    description="FastAPI service for document text extraction and embedding storage in Qdrant",
    version="2.0.0"
)

@app.get("/")
async def root():
    return {"message": "Document Embedding Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document-embedding"}

@app.post("/process-and-store")
async def process_and_store(
    task_id: str = Form(...),
    file: UploadFile = File(...),
    query_text: str = Form(...),
    doc_id: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    doc_type: Optional[str] = Form(None)
):
    """
    Process both document and query text: extract text from file and process query, 
    storing embeddings in separate collections with the same task_id.
    
    Args:
        task_id: Required task identifier for reference by other services
        file: The uploaded document file (PDF, DOCX, image, etc.)
        query_text: The text query to process and store
        doc_id: Optional document identifier (will be auto-generated if not provided)
        source: Optional source identifier
        doc_type: Optional document type metadata
        
    Returns:
        JSON response with storage status for both document and query
    """
    try:
        start_time = time.time()
        
        # Get file extension
        original_filename = file.filename or "document"
        file_extension = ""
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        
        logger.info(f"Processing file: {original_filename}")
        
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
            # Process 1: File Document
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
                    detail=f"Document extraction failed: {extraction_results['error']}"
                )
            
            # Get extracted text
            document_text = extraction_results.get('text', '')
            if not document_text or len(document_text.strip()) < 10:
                raise HTTPException(
                    status_code=400,
                    detail="No readable text found in the document. The document might be empty or corrupted."
                )
            
            logger.info(f"Extracted {len(document_text)} characters of text from document")
            
            # Prepare document metadata
            doc_metadata = {
                "task_id": task_id,
                "content_type": "document",
                "filename": original_filename,
                "extraction_method": extraction_results.get('method', 'unknown'),
                "pages": extraction_results.get('pages', 0),
                "images_processed": extraction_results.get('images_processed', 0),
                "upload_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if source:
                doc_metadata["source"] = source
            if doc_type:
                doc_metadata["doc_type"] = doc_type
            
            # Store document in documents collection
            doc_storage_result = process_and_store_document(
                text=document_text,
                doc_id=doc_id,
                metadata=doc_metadata,
                collection_name="documents"
            )
            
            # Process 2: Query Text
            logger.info(f"Processing query text: {len(query_text)} characters")
            
            # Prepare query metadata
            query_metadata = {
                "task_id": task_id,
                "content_type": "query",
                "query_text": query_text[:200] + "..." if len(query_text) > 200 else query_text,
                "upload_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if source:
                query_metadata["source"] = source
            
            # Store query in queries collection
            query_storage_result = process_and_store_document(
                text=query_text,
                doc_id=f"{task_id}_query",
                metadata=query_metadata,
                collection_name="queries"
            )
            
            # Check if both storage operations succeeded
            if doc_storage_result['status'] == 'error':
                raise HTTPException(
                    status_code=500,
                    detail=f"Document storage failed: {doc_storage_result['message']}"
                )
            
            if query_storage_result['status'] == 'error':
                raise HTTPException(
                    status_code=500,
                    detail=f"Query storage failed: {query_storage_result['message']}"
                )
            
            # Clean up temporary files
            try:
                os.unlink(temp_file_path)
                extractor.cleanup_temp_files()
                logger.info("Cleaned up temporary files")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
            
            # Calculate total processing time
            total_processing_time = time.time() - start_time
            
            # Format response
            response_data = {
                "success": True,
                "message": "Document and query successfully processed and stored",
                "task_id": task_id,
                "processing_time": f"{total_processing_time:.2f}s",
                "document_storage": {
                    "doc_id": doc_storage_result.get('doc_id'),
                    "collection": "documents",
                    "chunks_created": doc_storage_result.get('chunks_created', 0),
                    "collection_stats": doc_storage_result.get('statistics', {}),
                    "document_info": {
                        "filename": original_filename,
                        "text_length": len(document_text),
                        "extraction_method": extraction_results.get('method', 'unknown'),
                        "pages": extraction_results.get('pages', 0),
                        "images_processed": extraction_results.get('images_processed', 0)
                    }
                },
                "query_storage": {
                    "query_id": query_storage_result.get('doc_id'),
                    "collection": "queries", 
                    "chunks_created": query_storage_result.get('chunks_created', 0),
                    "collection_stats": query_storage_result.get('statistics', {}),
                    "query_info": {
                        "text_length": len(query_text),
                        "query_preview": query_text[:100] + "..." if len(query_text) > 100 else query_text
                    }
                }
            }
            
            logger.info(f"Successfully processed and stored document in {total_processing_time:.2f}s")
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
        logger.error(f"Unexpected error in process_and_store: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )

if __name__ == "__main__":
    # For local development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)