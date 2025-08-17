import os
import logging
import tempfile
import time
import requests
from typing import Dict, Any, Optional
from pathlib import Path
import runpod

# Import our processing modules
from universal_extractor import UniversalTextExtractor
from rag_utils_simplified import process_and_store_document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file_from_url(url: str, timeout: int = 30) -> tuple[str, str]:
    """
    Download file from URL (Cloudinary or any other source).
    
    Args:
        url: URL to download file from
        timeout: Download timeout in seconds
        
    Returns:
        Tuple of (file_path, original_filename)
    """
    try:
        # Extract filename from URL or use default
        url_parts = url.split('/')
        filename = url_parts[-1] if url_parts else "document"
        
        # Try to get actual filename from Content-Disposition header
        response = requests.head(url, timeout=5)
        if 'content-disposition' in response.headers:
            import re
            d = response.headers['content-disposition']
            fname = re.findall("filename=(.+)", d)
            if fname:
                filename = fname[0].strip('"')
        
        # Determine file extension
        file_extension = ""
        if '.' in filename:
            file_extension = filename.lower().split('.')[-1].split('?')[0]  # Remove query params
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f'.{file_extension}' if file_extension else '.pdf'
        ) as temp_file:
            logger.info(f"Downloading file from: {url}")
            
            # Download file content
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Write content to temp file
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            
            temp_file_path = temp_file.name
            
        logger.info(f"Downloaded file to: {temp_file_path} (size: {os.path.getsize(temp_file_path)} bytes)")
        return temp_file_path, filename
        
    except requests.RequestException as e:
        logger.error(f"Failed to download file from URL: {e}")
        raise Exception(f"Failed to download file: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading file: {e}")
        raise


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for document and query processing.
    
    Expected input format:
    {
        "input": {
            "file_url": "https://res.cloudinary.com/...",  # Required
            "query_text": "your query text here",  # Required
            "task_id": "unique_task_identifier",  # Required
            "doc_id": "optional_document_id",
            "source": "optional_source",
            "doc_type": "optional_doc_type"
        }
    }
    
    Returns:
    {
        "success": true/false,
        "task_id": "task_identifier",
        "message": "status_message",
        "processing_time": "X.XXs",
        "document_storage": {...},
        "query_storage": {...},
        "error": "error_message_if_failed"
    }
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Extract input parameters
        job_input = job.get("input", {})
        file_url = job_input.get("file_url")
        query_text = job_input.get("query_text")
        task_id = job_input.get("task_id")
        doc_id = job_input.get("doc_id")
        source = job_input.get("source")
        doc_type = job_input.get("doc_type")
        
        # Validate required inputs
        if not file_url:
            return {
                "success": False,
                "error": "Missing required parameter: file_url"
            }
        
        if not query_text:
            return {
                "success": False,
                "error": "Missing required parameter: query_text"
            }
        
        if not task_id:
            return {
                "success": False,
                "error": "Missing required parameter: task_id"
            }
        
        logger.info(f"Processing job {job.get('id')} with task_id {task_id}: {file_url}")
        
        # Process 1: File Document
        # Download file from URL
        try:
            temp_file_path, original_filename = download_file_from_url(file_url)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to download file: {str(e)}"
            }
        
        # Initialize text extractor
        extractor = UniversalTextExtractor(
            output_dir="temp_extracts",
            use_ocr=True,
            enable_gpu=True  # Use GPU if available in RunPod
        )
        
        # Extract text from document
        logger.info(f"Extracting text from: {original_filename}")
        extraction_results = extractor.process_file(temp_file_path)
        
        if 'error' in extraction_results:
            return {
                "success": False,
                "error": f"Document extraction failed: {extraction_results['error']}"
            }
        
        # Get extracted text
        document_text = extraction_results.get('text', '')
        if not document_text or len(document_text.strip()) < 10:
            return {
                "success": False,
                "error": "No readable text found in the document"
            }
        
        logger.info(f"Extracted {len(document_text)} characters of text")
        
        # Prepare document metadata
        doc_metadata = {
            "task_id": task_id,
            "content_type": "document",
            "filename": original_filename,
            "file_url": file_url,
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
        logger.info("Processing document: generating embeddings and storing in Qdrant...")
        doc_storage_result = process_and_store_document(
            text=document_text,
            doc_id=doc_id,
            metadata=doc_metadata,
            collection_name="documents"
        )
        
        if doc_storage_result['status'] == 'error':
            return {
                "success": False,
                "error": f"Document storage failed: {doc_storage_result['message']}"
            }
        
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
        logger.info("Processing query: generating embeddings and storing in Qdrant...")
        query_storage_result = process_and_store_document(
            text=query_text,
            doc_id=f"{task_id}_query",
            metadata=query_metadata,
            collection_name="queries"
        )
        
        if query_storage_result['status'] == 'error':
            return {
                "success": False,
                "error": f"Query storage failed: {query_storage_result['message']}"
            }
        
        # Clean up temporary files
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            extractor.cleanup_temp_files()
            logger.info("Cleaned up temporary files")
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Prepare successful response
        return {
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
        
    except Exception as e:
        logger.error(f"Unexpected error in handler: {str(e)}", exc_info=True)
        
        # Clean up on error
        try:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        except:
            pass
        
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "processing_time": f"{time.time() - start_time:.2f}s",
            "task_id": job_input.get("task_id", "unknown")
        }


# RunPod serverless entrypoint
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})