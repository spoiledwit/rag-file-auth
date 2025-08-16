from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from rest_framework import generics, serializers, status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken
from django.utils import timezone
from cloudinary.uploader import upload as cloudinary_upload
import uuid
import random
import time
import re
from .rag_utils import ingest_new_text_file
from django.db import transaction
import tempfile
from rest_framework.exceptions import ValidationError
from datetime import datetime
import logging
from scripts.universal_extractor import UniversalTextExtractor
from rest_framework.parsers import JSONParser
logger = logging.getLogger(__name__)
from .serializers import (
    SubmittedFileSerializer, 
    
    
)
from .models import (
    SubmittedFile,
    CategorySchema,
    ProcessingTask
)
from .serializers import CategorySchemaSerializer


# AUTHENTICATION VIEWS
@api_view(['GET'])
@permission_classes([AllowAny])
def index(request):
    return Response({
        'message': 'Welcome to the FileAuthAI API',
        'status': 'success',
        'version': '1.0',
        'endpoints': {
            'auth': {
                'login': '/auth/login/',
                'refresh': '/auth/refresh/',
                'register': '/auth/register/',
                'profile': '/auth/profile/',
            },
            'api': {
                'submit_file': '/api/v1/submit-file/',
                'upload_reference': '/api/v1/upload-ml-reference/',
                'query_document': '/api/v1/query-document/',
                'ask_rag_question': '/api/v1/ask-rag-question/',
            }
        }
    })


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    from django.views.decorators.csrf import csrf_exempt
    from django.utils.decorators import method_decorator
    return JsonResponse({"status": "healthy"})


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password', 'password_confirm', 'first_name', 'last_name')

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        user = User.objects.create_user(**validated_data)
        return user


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'date_joined', 'is_active')
        read_only_fields = ('id', 'username', 'date_joined', 'is_active')


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserRegistrationSerializer
    permission_classes = [AllowAny]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        refresh = RefreshToken.for_user(user)
        return Response({
            'message': 'User registered successfully',
            'user': UserProfileSerializer(user).data,
            'tokens': {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }
        }, status=status.HTTP_201_CREATED)


class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserProfileSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user


# ML REFERENCE UPLOAD 
@api_view(['GET'])
@permission_classes([AllowAny])
def category_list(request):
    categories = CategorySchema.objects.all()
    serializer = CategorySchemaSerializer(categories, many=True)
    return Response(serializer.data)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def upload_ml_reference(request):
    """
    Upload ML reference files for training and categorization.
    
    Restricted to admin users only. Handles document files (DOC, DOCX, PDF)
    using Cloudinary raw upload to prevent ZIP processing errors.
    Creates or updates category schemas as needed.
    
    Expected fields:
    - ml_reference_id: Unique identifier (auto-generated if not provided)
    - category: Category name for the reference
    - description: Description of the reference file
    - reasoning_notes: Explanation of why this file is a good reference
    - metadata: Additional JSON metadata (optional)
    - file: The uploaded reference file
    - file_name: Name of the file (auto-corrected if needed)
    
    Returns success message with the reference ID.
    """
    try:
        if request.user.userprofile.role != 'admin':
            return Response({"error": "Unauthorized"}, status=403)

        ml_reference_id = request.data.get('ml_reference_id') or str(uuid.uuid4())
        category_name = request.data.get('category')
        description = request.data.get('description')
        reasoning_notes = request.data.get('reasoning_notes')
        metadata = request.data.get('metadata')  # Should be JSON string or dict
        file_obj = request.FILES.get('file')
        file_name = request.data.get('file_name')

        if not all([category_name, description, reasoning_notes, file_obj, file_name]):
            return Response({"error": "Missing required fields."}, status=400)

        # Validate and clean filename
        if not file_name or file_name.strip() == '':
            return Response({"error": "File name cannot be empty."}, status=400)
        
        # Get proper filename from the uploaded file object
        original_filename = getattr(file_obj, 'name', '')
        
        # Determine file type - try multiple approaches
        file_extension = ''
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        elif '.' in file_name:
            file_extension = file_name.lower().split('.')[-1]
        else:
            # Try to detect from content type
            content_type = getattr(file_obj, 'content_type', '')
            if 'pdf' in content_type.lower():
                file_extension = 'pdf'
            elif 'word' in content_type.lower() or 'document' in content_type.lower():
                file_extension = 'docx'
        
        # Create a proper filename
        if original_filename and '.' in original_filename:
            # Use the original filename if it's properly formatted
            final_filename = original_filename
        elif file_name and '.' in file_name and not file_name.lower() in ['pdf', 'doc', 'docx']:
            # Use provided filename if it's properly formatted
            final_filename = file_name
        else:
            # Generate a proper filename
            timestamp = int(time.time())
            if file_extension:
                final_filename = f"reference_{timestamp}.{file_extension}"
            else:
                final_filename = f"reference_{timestamp}.pdf"  # Default to PDF
                file_extension = 'pdf'
        
        try:
            if file_extension in ['doc', 'docx', 'pdf']:
                # For document files, use 'raw' resource type to prevent ZIP processing
                public_id = f"ml_references/{ml_reference_id}_{final_filename}"
                upload_result = cloudinary_upload(
                    file_obj,
                    resource_type="raw",
                    public_id=public_id,
                    overwrite=True
                )
            else:
                # For images and other files, use auto detection
                upload_result = cloudinary_upload(
                    file_obj,
                    folder="ml_references",
                    public_id=f"{ml_reference_id}_{final_filename}",
                    overwrite=True
                )
        except Exception as upload_error:
            return Response({
                "error": f"File upload failed: {str(upload_error)}"
            }, status=400)
            
        cloudinary_url = upload_result.get('secure_url')
        
        # For document files, ensure URL has proper file extension
        if file_extension in ['doc', 'docx', 'pdf']:
            # Always reconstruct the URL to ensure proper extension
            import os
            cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME', 'dazphdgdr')
            base_url = f"https://res.cloudinary.com/{cloud_name}"
            version = upload_result.get('version', '')
            if version:
                cloudinary_url = f"{base_url}/raw/upload/v{version}/ml_references/{ml_reference_id}_{final_filename}"
            else:
                cloudinary_url = f"{base_url}/raw/upload/ml_references/{ml_reference_id}_{final_filename}"
        
        if not cloudinary_url:
            return Response({
                "error": "Failed to get upload URL from Cloudinary"
            }, status=500)

        category_obj, _ = CategorySchema.objects.get_or_create(
            category_name=category_name,
            defaults={"description": description or ""}
        )

        # ML reference file processing removed - models no longer exist
        logger.info(f"ML reference would be created with ID: {ml_reference_id}")

        return Response({"message": "ML reference uploaded", "id": ml_reference_id}, status=201)

    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def submit_file(request):
    """
    Upload and process files with UniversalTextExtractor and RAG integration.
    
    Handles document files (DOC, DOCX, PDF) using Cloudinary raw upload
    to prevent ZIP processing errors. Automatically detects file types
    and generates proper filenames with extensions. Saves extracted text to a .txt file.
    
    Expected form data:
    - file: The uploaded file (required)
    - category: File category for classification (required)
    - file_name: Name of the file (auto-corrected if needed)
    - metadata: Additional JSON metadata (optional)
    
    Returns:
    - SubmittedFile data with processing results, including extracted text, metadata, and path to .txt file
    """
    try:
        # Extract and validate input
        category_name = request.data.get('category')
        file_obj = request.FILES.get('file')
        file_name = request.data.get('file_name')
        metadata = request.data.get('metadata', {})

        if not all([category_name, file_obj, file_name]):
            return Response({"error": "Missing required fields (category, file, file_name)."}, status=status.HTTP_400_BAD_REQUEST)

        # Validate and clean filename
        if not file_name or file_name.strip() == '':
            return Response({"error": "File name cannot be empty."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get proper filename from the uploaded file object
        original_filename = getattr(file_obj, 'name', '')
        
        # Determine file type - try multiple approaches
        file_extension = ''
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        elif '.' in file_name:
            file_extension = file_name.lower().split('.')[-1]
        else:
            # Try to detect from content type
            content_type = getattr(file_obj, 'content_type', '')
            if 'pdf' in content_type.lower():
                file_extension = 'pdf'
            elif 'word' in content_type.lower() or 'document' in content_type.lower():
                file_extension = 'docx'
        
        # Create a proper filename
        if original_filename and '.' in original_filename:
            final_filename = original_filename
        elif file_name and '.' in file_name and not file_name.lower() in ['pdf', 'doc', 'docx']:
            final_filename = file_name
        else:
            timestamp = int(time.time())
            if file_extension:
                final_filename = f"document_{timestamp}.{file_extension}"
            else:
                final_filename = f"document_{timestamp}.pdf"  # Default to PDF
                file_extension = 'pdf'
        
        # Generate unique identifier for file
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Upload to Cloudinary
        try:
            if file_extension in ['doc', 'docx', 'pdf']:
                public_id = f"submitted_files/{unique_id}_{final_filename}"
                upload_result = cloudinary_upload(
                    file_obj,
                    resource_type="raw",
                    public_id=public_id,
                    overwrite=True
                )
            else:
                upload_result = cloudinary_upload(
                    file_obj,
                    folder="submitted_files",
                    public_id=f"{unique_id}_{final_filename}",
                    overwrite=True
                )
        except Exception as upload_error:
            return Response({
                "error": f"File upload failed: {str(upload_error)}"
            }, status=status.HTTP_400_BAD_REQUEST)
            
        cloudinary_url = upload_result.get('secure_url')
        
        # For document files, ensure URL has proper file extension
        if file_extension in ['doc', 'docx', 'pdf']:
            import os
            cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME', 'dazphdgdr')
            base_url = f"https://res.cloudinary.com/{cloud_name}"
            version = upload_result.get('version', '')
            if version:
                cloudinary_url = f"{base_url}/raw/upload/v{version}/submitted_files/{unique_id}_{final_filename}"
            else:
                cloudinary_url = f"{base_url}/raw/upload/submitted_files/{unique_id}_{final_filename}"
        
        if not cloudinary_url:
            return Response({
                "error": "Failed to get upload URL from Cloudinary"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Validate category
        category_obj = CategorySchema.objects.filter(category_name=category_name).first()
        if not category_obj:
            return Response({"error": f"Invalid category: {category_name}"}, status=status.HTTP_400_BAD_REQUEST)

        # Create SubmittedFile record
        with transaction.atomic():
            submitted_file = SubmittedFile.objects.create(
                file=cloudinary_url,
                file_name=final_filename,
                category=category_obj,
                uploaded_by=request.user,
                status='processing',
                extracted_fields={}
            )

            # Audit logging removed - model no longer exists
            logger.info(f"File uploaded: {final_filename} by {request.user.username}")

            try:
                # Initialize UniversalTextExtractor
                extractor = UniversalTextExtractor(
                    output_dir="extracted_texts",
                    use_ocr=True,
                    enable_gpu=True
                )

                # Process the file URL
                extraction_results = extractor.process_file(cloudinary_url)
                if 'error' in extraction_results:
                    raise ValueError(f"Extraction failed: {extraction_results['error']}")

                # Save extracted text to a .txt file
                output_filename = f"{unique_id}_{final_filename}_extracted_text.txt"
                output_file_path = extractor.save_results(extraction_results, output_filename=output_filename)
                logger.info(f"Extracted text saved to: {output_file_path}")

                # RAG Integration: Ingest the extracted text into the knowledge base
                try:
                    doc_id = f"doc_{submitted_file.id}_{unique_id}"
                    rag_success = ingest_new_text_file(output_file_path, doc_id)
                    if rag_success:
                        logger.info(f"Successfully ingested document into RAG: {doc_id}")
                    else:
                        logger.warning(f"Failed to ingest document into RAG: {doc_id}")
                except Exception as rag_error:
                    logger.error(f"RAG ingestion failed: {rag_error}")

                # Clean up temporary files and images
                extractor.cleanup_temp_files()
                logger.info("Cleaned up temporary files and images")

                # Update SubmittedFile with processing results
                submitted_file.status = 'completed'
                submitted_file.processed_at = timezone.now()
                submitted_file.save()

                # ProcessedFile record creation removed - model no longer exists
                logger.info(f"Processing completed for {final_filename}")

                # Processing audit log removed - model no longer exists
                logger.info(f"Processing completed for {submitted_file.id}")

                # Serialize and return response
                serializer = SubmittedFileSerializer(submitted_file)
                response_data = serializer.data
                response_data['output_file_path'] = output_file_path  # Include the .txt file path in the response
                response_data['rag_ingested'] = rag_success if 'rag_success' in locals() else False  # Include RAG status
                return Response(response_data, status=status.HTTP_201_CREATED)

            except Exception as processing_error:
                logger.error(f"File processing failed for {cloudinary_url}: {str(processing_error)}", exc_info=True)
                
                # Clean up temporary files even on error
                try:
                    if 'extractor' in locals():
                        extractor.cleanup_temp_files()
                        logger.info("Cleaned up temporary files after error")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")
                
                submitted_file.status = 'failed'
                submitted_file.error_message = str(processing_error)
                submitted_file.processed_at = timezone.now()
                submitted_file.save()

                # Error logging removed - models no longer exist
                logger.error(f"Processing failed for {submitted_file.id}: {str(processing_error)}")

                return Response(
                    {"error": f"File processing failed: {str(processing_error)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

    except ValidationError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return Response({"error": str(ve)}, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        logger.error(f"Unexpected error in file submission: {str(e)}", exc_info=True)
        return Response(
            {"error": "An unexpected error occurred during file submission"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([JSONParser])
def ask_rag_question(request):
    """Handle RAG questions from frontend - now calls FastAPI service for text queries"""
    try:
        question = request.data.get('question')
        text_content = request.data.get('text', '')
        method = request.data.get('method', 'hybrid')
        top_k = int(request.data.get('top_k', 30))
        
        if not question:
            return Response({"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        if not text_content:
            return Response({"error": "No text content provided for query"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Call FastAPI service for text processing
        try:
            import requests
            import os
            
            # Get FastAPI service URL from environment or use default
            fastapi_url = os.getenv('FASTAPI_SERVICE_URL', 'http://localhost:8000')
            
            # Prepare data for FastAPI request
            data = {
                'text': text_content,
                'query': question,
                'method': method,
                'top_k': top_k
            }
            
            logger.info(f"Sending text query to FastAPI service: {fastapi_url}/query-text")
            
            # Call FastAPI text processing service
            response = requests.post(
                f"{fastapi_url}/query-text",
                data=data,
                timeout=60  # 1 minute timeout for text queries
            )
            
            if response.status_code != 200:
                logger.error(f"FastAPI service error: {response.status_code} - {response.text}")
                return Response({
                    "error": f"Text processing failed: {response.text}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Parse FastAPI response
            fastapi_result = response.json()
            
            if not fastapi_result.get('success'):
                logger.error(f"FastAPI text processing failed: {fastapi_result}")
                return Response({
                    "error": fastapi_result.get('error', 'Text processing failed')
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Return response in expected format
            return Response({
                "answer": fastapi_result.get('answer', 'No answer generated'),
                "sources": fastapi_result.get('doc_sources', []),
                "confidence": fastapi_result.get('evaluation', {}),
                "accuracy_score": fastapi_result.get('accuracy_score', 0.0),
                "extracted_fields": fastapi_result.get('extracted_fields', {}),
                "retrieval_method": fastapi_result.get('retrieval_method', method),
                "processing_time": fastapi_result.get('processing_time', 0)
            })
            
        except requests.Timeout:
            logger.error("FastAPI service request timed out")
            return Response({
                "error": "Text processing timed out. Please try again."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        except requests.RequestException as e:
            logger.error(f"FastAPI service request failed: {str(e)}")
            return Response({
                "error": f"Text processing service unavailable: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    except Exception as e:
        logger.error(f"Unexpected error in ask_rag_question: {str(e)}", exc_info=True)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def query_document(request):
    """
    Create an asynchronous task for document processing and return task ID immediately.
    
    This endpoint uploads files to Cloudinary, creates a processing task, and returns
    the task ID for frontend polling. The actual processing happens asynchronously.
    
    Expected form data:
    - file: The document file (PDF, DOCX, or image)
    - query: The user's question about the document
    - category: Document category (required)
    - method: Retrieval method - "semantic", "keyword", or "hybrid" (optional, defaults to "hybrid")
    - top_k: Number of chunks to retrieve (optional, defaults to 30)
    
    Returns:
    - task_id: UUID of the processing task for polling
    - message: Success message
    """
    try:
        # Import tasks here to avoid circular imports
        from .tasks import process_document_async
        
        # Extract and validate input
        query = request.data.get('query')
        file_obj = request.FILES.get('file')
        category_name = request.data.get('category', 'General')
        method = request.data.get('method', 'hybrid')
        top_k = int(request.data.get('top_k', 30))
        
        if not all([query, file_obj]):
            return Response({
                "error": "Missing required fields. Both 'query' and 'file' are required."
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Validate category
        category_obj, created = CategorySchema.objects.get_or_create(
            category_name=category_name,
            defaults={'description': f'Auto-created category for {category_name}'}
        )
            
        if method not in ['semantic', 'keyword', 'hybrid']:
            return Response({
                "error": "Invalid method. Use 'semantic', 'keyword', or 'hybrid'."
            }, status=status.HTTP_400_BAD_REQUEST)
        
        logger.info(f"Creating async task for query-document request: '{query}' with method: {method}")
        
        # Get file details
        original_filename = getattr(file_obj, 'name', 'document')
        file_extension = ''
        if '.' in original_filename:
            file_extension = original_filename.lower().split('.')[-1]
        
        # Generate unique identifier for file
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        final_filename = f"{unique_id}_{original_filename}"
        
        # Upload file to Cloudinary first
        try:
            if file_extension in ['doc', 'docx', 'pdf']:
                public_id = f"query_documents/{unique_id}_{original_filename}"
                upload_result = cloudinary_upload(
                    file_obj,
                    resource_type="raw",
                    public_id=public_id,
                    overwrite=True
                )
            else:
                upload_result = cloudinary_upload(
                    file_obj,
                    folder="query_documents",
                    public_id=f"{unique_id}_{original_filename}",
                    overwrite=True
                )
            
            cloudinary_url = upload_result.get('secure_url')
            
            # For document files, ensure URL has proper file extension
            if file_extension in ['doc', 'docx', 'pdf']:
                import os
                cloud_name = os.getenv('CLOUDINARY_CLOUD_NAME', 'dewqsghdi')
                base_url = f"https://res.cloudinary.com/{cloud_name}"
                version = upload_result.get('version', '')
                if version:
                    cloudinary_url = f"{base_url}/raw/upload/v{version}/query_documents/{unique_id}_{original_filename}"
                else:
                    cloudinary_url = f"{base_url}/raw/upload/query_documents/{unique_id}_{original_filename}"
            
            logger.info(f"File uploaded to Cloudinary: {cloudinary_url}")
            
        except Exception as upload_error:
            return Response({
                "error": f"File upload failed: {str(upload_error)}"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create ProcessingTask record
        task_record = ProcessingTask.objects.create(
            task_id='',  # Will be updated after Celery task creation
            user=request.user,
            file_name=original_filename,
            file_url=cloudinary_url,
            category=category_obj,
            query=query,
            method=method,
            top_k=top_k,
            status='pending',
            progress_message='Task created, waiting to start...'
        )
        
        logger.info(f"Created ProcessingTask record with ID: {task_record.id}")
        
        # Start the async task
        celery_task = process_document_async.delay(
            str(task_record.id),
            cloudinary_url,
            query,
            method,
            top_k
        )
        
        # Update task record with Celery task ID
        task_record.task_id = celery_task.id
        task_record.save()
        
        logger.info(f"Started Celery task {celery_task.id} for ProcessingTask {task_record.id}")
        
        # Return task information for frontend polling
        return Response({
            "success": True,
            "task_id": str(task_record.id),
            "celery_task_id": celery_task.id,
            "message": "Document processing task created successfully. Use the task_id to check status.",
            "status": "pending",
            "file_name": original_filename,
            "query": query,
            "category": category_name,
            "method": method
        }, status=status.HTTP_202_ACCEPTED)
            
    except Exception as e:
        logger.error(f"Unexpected error in query_document: {str(e)}", exc_info=True)
        return Response({
            "error": "An unexpected error occurred while creating the processing task"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def task_status(request, task_id):
    """
    Get the status of a processing task.
    
    Args:
        task_id: UUID of the ProcessingTask
        
    Returns:
        Task status, progress, and results if completed
    """
    try:
        # Get the task record
        task_record = ProcessingTask.objects.get(id=task_id, user=request.user)
        
        # Get Celery task status if task is running
        celery_task_info = None
        if task_record.task_id and task_record.status in ['pending', 'processing']:
            try:
                from celery.result import AsyncResult
                celery_task = AsyncResult(task_record.task_id)
                celery_task_info = {
                    'state': celery_task.state,
                    'info': celery_task.info if celery_task.info else {}
                }
            except Exception as e:
                logger.warning(f"Failed to get Celery task status: {e}")
        
        # Prepare response data
        response_data = {
            'task_id': str(task_record.id),
            'celery_task_id': task_record.task_id,
            'status': task_record.status,
            'progress_percentage': task_record.progress_percentage,
            'progress_message': task_record.progress_message,
            'file_name': task_record.file_name,
            'query': task_record.query,
            'category': task_record.category.category_name,
            'method': task_record.method,
            'created_at': task_record.created_at.isoformat(),
            'started_at': task_record.started_at.isoformat() if task_record.started_at else None,
            'completed_at': task_record.completed_at.isoformat() if task_record.completed_at else None,
        }
        
        # Add error message if failed
        if task_record.status == 'failed':
            response_data['error_message'] = task_record.error_message
        
        # Add result if completed
        if task_record.status == 'completed':
            response_data['result'] = task_record.result
        
        # Add Celery task info if available
        if celery_task_info:
            response_data['celery_status'] = celery_task_info
        
        return Response(response_data, status=status.HTTP_200_OK)
        
    except ProcessingTask.DoesNotExist:
        return Response({
            'error': 'Task not found or you do not have permission to view it'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}", exc_info=True)
        return Response({
            'error': 'An unexpected error occurred while getting task status'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)