import logging
import requests
import time
from celery import shared_task
from django.utils import timezone
from django.conf import settings
from .models import ProcessingTask, SubmittedFile

logger = logging.getLogger(__name__)


@shared_task(bind=True)
def process_document_async(self, task_uuid, file_url, query, method, top_k):
    """
    Asynchronous task to process a document with FastAPI service.
    
    Args:
        task_uuid: UUID of the ProcessingTask record
        file_url: URL of the uploaded file (Cloudinary)
        query: User query about the document
        method: Retrieval method (semantic, keyword, hybrid)
        top_k: Number of chunks to retrieve
    """
    task_record = None
    
    try:
        # Get the task record
        task_record = ProcessingTask.objects.get(id=task_uuid)
        
        # Update task status to processing
        task_record.status = 'processing'
        task_record.started_at = timezone.now()
        task_record.progress_percentage = 10
        task_record.progress_message = 'Starting document processing...'
        task_record.save()
        
        logger.info(f"Starting document processing for task {task_uuid}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 10,
                'total': 100,
                'status': 'Starting document processing...'
            }
        )
        
        # Get FastAPI service URL
        fastapi_url = getattr(settings, 'CELERY_FASTAPI_URL', 'http://localhost:8000')
        
        # Download file from Cloudinary URL
        task_record.progress_percentage = 20
        task_record.progress_message = 'Downloading file...'
        task_record.save()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 20,
                'total': 100,
                'status': 'Downloading file...'
            }
        )
        
        # Download the file from Cloudinary
        file_response = requests.get(file_url, timeout=60)
        file_response.raise_for_status()
        
        # Update progress
        task_record.progress_percentage = 40
        task_record.progress_message = 'Sending to processing service...'
        task_record.save()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 40,
                'total': 100,
                'status': 'Sending to processing service...'
            }
        )
        
        # Prepare files and data for FastAPI request
        files = {
            'file': (task_record.file_name, file_response.content, 'application/octet-stream')
        }
        data = {
            'query': query,
            'method': method,
            'top_k': top_k
        }
        
        logger.info(f"Sending request to FastAPI service: {fastapi_url}/process-document")
        
        # Update progress
        task_record.progress_percentage = 50
        task_record.progress_message = 'Processing document with AI...'
        task_record.save()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 50,
                'total': 100,
                'status': 'Processing document with AI...'
            }
        )
        
        # Call FastAPI processing service
        response = requests.post(
            f"{fastapi_url}/process-document",
            files=files,
            data=data,
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code != 200:
            error_msg = f"FastAPI service error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Parse FastAPI response
        fastapi_result = response.json()
        
        if not fastapi_result.get('success'):
            error_msg = fastapi_result.get('error', 'Unknown processing error')
            logger.error(f"FastAPI processing failed: {error_msg}")
            raise Exception(error_msg)
        
        # Update progress
        task_record.progress_percentage = 80
        task_record.progress_message = 'Saving results...'
        task_record.save()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 80,
                'total': 100,
                'status': 'Saving results...'
            }
        )
        
        # Create SubmittedFile record with results
        submitted_file = SubmittedFile.objects.create(
            file=file_url,
            file_name=task_record.file_name,
            category=task_record.category,
            uploaded_by=task_record.user,
            query=query,
            status='completed',
            ai_response=fastapi_result.get('answer', 'No answer generated'),
            accuracy_score=fastapi_result.get('accuracy_score', 0.0),
            extracted_fields=fastapi_result.get('extracted_fields', {}),
            processing_metadata={
                'retrieval_method': fastapi_result.get('retrieval_method', method),
                'processing_time': fastapi_result.get('processing_time', 0),
                'chunks_processed': fastapi_result.get('chunks_processed', 0),
                'relevant_chunks': fastapi_result.get('relevant_chunks', 0),
                'num_docs_retrieved': fastapi_result.get('num_docs_retrieved', 0),
                'doc_sources': fastapi_result.get('doc_sources', []),
                'evaluation': fastapi_result.get('evaluation', {}),
                'top_k': top_k,
                'document_info': fastapi_result.get('document_info', {})
            },
            processed_at=timezone.now()
        )
        
        # Update task record with final results
        task_record.status = 'completed'
        task_record.completed_at = timezone.now()
        task_record.progress_percentage = 100
        task_record.progress_message = 'Processing completed successfully!'
        task_record.result = {
            'submitted_file_id': submitted_file.id,
            'answer': fastapi_result.get('answer', 'No answer generated'),
            'accuracy_score': fastapi_result.get('accuracy_score', 0.0),
            'extracted_fields': fastapi_result.get('extracted_fields', {}),
            'retrieval_method': fastapi_result.get('retrieval_method', method),
            'processing_time': fastapi_result.get('processing_time', 0),
            'evaluation': fastapi_result.get('evaluation', {}),
            'document_info': fastapi_result.get('document_info', {})
        }
        task_record.save()
        
        logger.info(f"Successfully completed document processing for task {task_uuid}")
        
        # Return final result
        return {
            'current': 100,
            'total': 100,
            'status': 'Processing completed successfully!',
            'result': task_record.result
        }
        
    except Exception as e:
        logger.error(f"Document processing failed for task {task_uuid}: {str(e)}", exc_info=True)
        
        # Update task record with error
        if task_record:
            task_record.status = 'failed'
            task_record.completed_at = timezone.now()
            task_record.error_message = str(e)
            task_record.progress_message = f'Processing failed: {str(e)}'
            task_record.save()
        
        # Update Celery task state
        self.update_state(
            state='FAILURE',
            meta={
                'current': task_record.progress_percentage if task_record else 0,
                'total': 100,
                'status': f'Processing failed: {str(e)}',
                'error': str(e)
            }
        )
        
        # Re-raise the exception so Celery marks the task as failed
        raise e