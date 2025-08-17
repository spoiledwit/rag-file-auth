import logging
import requests
import time
import os
from celery import shared_task
from django.utils import timezone
from django.conf import settings
from .models import ProcessingTask, SubmittedFile

# Import simplified RAG utilities (no embedding models needed)
from .rag_utils_simplified import generate_response_runpod, extract_analysis_data, evaluate_response_simple

# Qdrant import
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


def perform_rag_query_simple(task_id: str, query: str) -> dict:
    """
    Perform RAG query using stored embeddings in Qdrant collections.
    Uses the existing RAG utilities but retrieves chunks from Qdrant.
    """
    try:
        start_time = time.time()
        
        # Connect to Qdrant
        qdrant_url = os.getenv("QDRANT_URL", "localhost").replace("https://", "http://")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url.startswith(('http://', 'https://')):
            qdrant_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                prefer_grpc=False
            )
        else:
            qdrant_client = QdrantClient(
                host=qdrant_url,
                port=qdrant_port
            )
        
        # 1. Get query embedding from queries collection
        query_result = qdrant_client.scroll(
            collection_name="queries",
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="task_id",
                        match=models.MatchValue(value=task_id)
                    )
                ]
            ),
            limit=5,
            with_payload=True,
            with_vectors=True
        )
        
        query_chunks = query_result[0]
        if not query_chunks:
            return {
                "answer": "No query embeddings found for this task in the vector database.",
                "accuracy_score": 0.0,
                "evaluation": {"overall_score": 0.0},
                "num_docs_retrieved": 0,
                "processing_time": time.time() - start_time
            }
        
        # Use the first query embedding for search
        query_vector = query_chunks[0].vector
        
        # 2. Search document chunks using query embedding
        similarity_results = qdrant_client.search(
            collection_name="documents",
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="task_id",
                        match=models.MatchValue(value=task_id)
                    )
                ]
            ),
            limit=15,  # Get more chunks for better context
            score_threshold=0.3
        )
        
        if not similarity_results:
            return {
                "answer": "No relevant document chunks found for your query.",
                "accuracy_score": 0.0,
                "evaluation": {"overall_score": 0.0},
                "num_docs_retrieved": 0,
                "processing_time": time.time() - start_time
            }
        
        # 3. Format chunks for RAG processing (match the existing format)
        context_docs = []
        for hit in similarity_results:
            context_docs.append({
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "doc_id": hit.payload.get("doc_id", task_id),
                "chunk_idx": hit.payload.get("chunk_idx", 0),
                "metadata": hit.payload
            })
        
        logger.info(f"Retrieved {len(context_docs)} document chunks for RAG processing")
        
        # 4. Generate response using existing RAG utilities
        answer = generate_response_runpod(query, context_docs[:10])  # Use top 10 chunks
        
        # 5. Extract analysis data using existing function
        accuracy_score, extracted_fields = extract_analysis_data(answer)
        
        # 6. Evaluate using existing function  
        evaluation = evaluate_response_simple(query, answer, context_docs[:10])
        
        processing_time = time.time() - start_time
        logger.info(f"RAG query completed in {processing_time:.2f}s")
        
        return {
            "answer": answer,
            "accuracy_score": accuracy_score,
            "extracted_fields": extracted_fields,
            "evaluation": evaluation,
            "num_docs_retrieved": len(context_docs),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"RAG query failed for task {task_id}: {e}")
        return {
            "answer": f"Error performing RAG query: Unable to connect to vector database or retrieve embeddings. Error: {str(e)}",
            "accuracy_score": 0.0,
            "evaluation": {"overall_score": 0.0, "error": str(e)},
            "num_docs_retrieved": 0,
            "processing_time": 0.0
        }


@shared_task(bind=True)
def process_document_async(self, task_uuid, file_url, query):
    """
    Asynchronous task to process a document with RunPod serverless.
    
    Args:
        task_uuid: UUID of the ProcessingTask record
        file_url: URL of the uploaded file (Cloudinary)
        query: User query about the document
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
        
        # Get RunPod endpoint URL from settings
        runpod_endpoint = getattr(settings, 'RUNPOD_PROCESSOR_ENDPOINT_URL', None)
        if not runpod_endpoint:
            raise Exception("RUNPOD_PROCESSOR_ENDPOINT_URL not configured in settings")
        
        # Update progress
        task_record.progress_percentage = 40
        task_record.progress_message = 'Sending to RunPod serverless...'
        task_record.save()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 40,
                'total': 100,
                'status': 'Sending to RunPod serverless...'
            }
        )
        
        # Generate unique doc_id for this processing task
        doc_id = f"doc_{task_uuid}_{int(time.time())}"
        
        # Prepare data for RunPod request
        runpod_payload = {
            "input": {
                "file_url": file_url,
                "query_text": query,
                "task_id": str(task_uuid),
                "doc_id": doc_id,
                "source": "django_upload",
                "doc_type": "user_document"
            }
        }
        
        logger.info(f"Sending request to RunPod endpoint: {runpod_endpoint}")
        
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
        
        # Submit job to RunPod serverless endpoint
        run_endpoint = f"{runpod_endpoint}/run"
        # Get RunPod API key
        runpod_api_key = os.getenv('RUNPOD_API_KEY')
        if not runpod_api_key:
            raise Exception("RUNPOD_API_KEY environment variable not set")
            
        response = requests.post(
            run_endpoint,
            json=runpod_payload,
            timeout=60,  # Initial submission timeout
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {runpod_api_key}'
            }
        )
        
        if response.status_code != 200:
            error_msg = f"RunPod job submission error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Parse RunPod submission response
        submission_result = response.json()
        job_id = submission_result.get('id')
        
        if not job_id:
            error_msg = f"No job ID returned from RunPod: {submission_result}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        logger.info(f"RunPod job submitted with ID: {job_id}")
        
        # Update progress
        task_record.progress_percentage = 60
        task_record.progress_message = f'RunPod job {job_id} submitted, polling for results...'
        task_record.save()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 60,
                'total': 100,
                'status': f'RunPod job {job_id} submitted, polling for results...'
            }
        )
        
        # Poll for results (up to 10 minutes)
        status_endpoint = f"{runpod_endpoint}/status/{job_id}"
        max_poll_time = 600  # 10 minutes
        poll_interval = 5  # 5 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_poll_time:
            try:
                status_response = requests.get(
                    status_endpoint, 
                    timeout=30,
                    headers={
                        'Authorization': f'Bearer {runpod_api_key}'
                    }
                )
                
                if status_response.status_code != 200:
                    logger.warning(f"Status check failed: {status_response.status_code} - {status_response.text}")
                    time.sleep(poll_interval)
                    continue
                
                status_result = status_response.json()
                job_status = status_result.get('status')
                
                logger.info(f"RunPod job {job_id} status: {job_status}")
                
                if job_status == 'COMPLETED':
                    # Extract the output from completed job
                    output = status_result.get('output', {})
                    if not output.get('success'):
                        error_msg = output.get('error', 'Unknown processing error')
                        logger.error(f"Document processing failed: {error_msg}")
                        raise Exception(error_msg)
                    break
                
                elif job_status == 'FAILED':
                    error_msg = status_result.get('error', 'RunPod job failed')
                    logger.error(f"RunPod job failed: {error_msg}")
                    raise Exception(error_msg)
                
                elif job_status in ['IN_PROGRESS', 'IN_QUEUE']:
                    # Update progress based on time elapsed
                    elapsed = time.time() - start_time
                    progress = min(60 + int((elapsed / max_poll_time) * 20), 79)
                    
                    task_record.progress_percentage = progress
                    task_record.progress_message = f'RunPod job {job_id} {job_status.lower()}, waiting...'
                    task_record.save()
                    
                    self.update_state(
                        state='PROGRESS',
                        meta={
                            'current': progress,
                            'total': 100,
                            'status': f'RunPod job {job_id} {job_status.lower()}, waiting...'
                        }
                    )
                    
                    time.sleep(poll_interval)
                    continue
                
                else:
                    logger.warning(f"Unknown RunPod job status: {job_status}")
                    time.sleep(poll_interval)
                    continue
                    
            except requests.RequestException as e:
                logger.warning(f"Status check request failed: {e}")
                time.sleep(poll_interval)
                continue
        
        else:
            # Timeout reached
            error_msg = f"RunPod job {job_id} timed out after {max_poll_time} seconds"
            logger.error(error_msg)
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
            ai_response='Document and query have been processed and stored in vector database',
            processing_metadata={
                'document_storage': output.get('document_storage', {}),
                'query_storage': output.get('query_storage', {}),
                'task_id': str(task_uuid),
                'doc_id': doc_id,
                'processing_time': output.get('processing_time', 0),
                'collections_used': ['documents', 'queries']
            },
            processed_at=timezone.now()
        )
        
        # Now perform RAG query using the stored embeddings
        task_record.progress_percentage = 85
        task_record.progress_message = 'Performing RAG query on stored embeddings...'
        task_record.save()
        
        self.update_state(
            state='PROGRESS',
            meta={
                'current': 85,
                'total': 100,
                'status': 'Performing RAG query on stored embeddings...'
            }
        )
        
        # Perform RAG query using stored embeddings
        rag_result = perform_rag_query_simple(str(task_uuid), query)
        
        # Check if RAG query succeeded
        rag_failed = (
            rag_result.get('accuracy_score', 0) == 0.0 and 
            'Error performing RAG query' in rag_result.get('answer', '')
        )
        
        # Update task record with final results
        if rag_failed:
            task_record.status = 'completed'  # Document processed but RAG failed
            task_record.completed_at = timezone.now()
            task_record.progress_percentage = 85  # Partial completion
            task_record.progress_message = 'Document processed successfully, but RAG query failed'
            task_record.error_message = rag_result.get('answer', 'RAG query failed')
        else:
            task_record.status = 'completed'
            task_record.completed_at = timezone.now()
            task_record.progress_percentage = 100
            task_record.progress_message = 'Processing completed successfully!'
        task_record.result = {
            'submitted_file_id': submitted_file.id,
            'document_storage': output.get('document_storage', {}),
            'query_storage': output.get('query_storage', {}),
            'task_id': str(task_uuid),
            'doc_id': doc_id,
            'processing_time': output.get('processing_time', 0),
            'collections_used': ['documents', 'queries'],
            'rag_answer': rag_result.get('answer', 'No answer generated'),
            'rag_accuracy_score': rag_result.get('accuracy_score', 0.0),
            'rag_evaluation': rag_result.get('evaluation', {}),
            'rag_retrieved_chunks': rag_result.get('num_docs_retrieved', 0),
            'message': 'Document and query processed, RAG query completed'
        }
        task_record.save()
        
        # Update SubmittedFile with RAG results
        submitted_file.ai_response = rag_result.get('answer', 'No answer generated')
        submitted_file.accuracy_score = rag_result.get('accuracy_score', 0.0)
        submitted_file.processing_metadata.update({
            'rag_evaluation': rag_result.get('evaluation', {}),
            'rag_retrieved_chunks': rag_result.get('num_docs_retrieved', 0),
            'rag_processing_time': rag_result.get('processing_time', 0)
        })
        submitted_file.save()
        
        logger.info(f"Successfully completed document processing and RAG query for task {task_uuid}")
        
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