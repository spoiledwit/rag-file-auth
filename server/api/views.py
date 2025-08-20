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
from cloudinary.uploader import upload as cloudinary_upload
import uuid
import time
import logging
from drf_spectacular.utils import (
    extend_schema, OpenApiParameter, OpenApiExample, OpenApiResponse
)
logger = logging.getLogger(__name__)

from .models import (
    CategorySchema,
    ProcessingTask
)
from .serializers import CategorySchemaSerializer


# AUTHENTICATION VIEWS
@extend_schema(
    methods=['GET'],
    summary="API root information",
    description="Returns basic service metadata and key endpoint shortcuts.",
    responses={200: OpenApiResponse(description="Root info returned")},
    tags=["Meta"]
)
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


@extend_schema(
    methods=['GET'],
    summary="Health check",
    description="Return a simple healthy status JSON for uptime monitoring.",
    responses={200: OpenApiResponse(description="Service healthy", response=None)}
)
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


from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt, name='dispatch')
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

@extend_schema(
    methods=['GET'],
    summary="List categories",
    description="Returns all configured document category schemas.",
    responses={200: CategorySchemaSerializer(many=True)},
    tags=["Categories"]
)
@api_view(['GET'])
@permission_classes([AllowAny])
def category_list(request):
    categories = CategorySchema.objects.all()
    serializer = CategorySchemaSerializer(categories, many=True)
    return Response(serializer.data)


@extend_schema(
    methods=['POST'],
    summary="Create document processing task",
    description="Uploads a file, stores metadata, and dispatches an asynchronous processing task. Returns a task ID for polling.",
    tags=["Processing"],
    request={'multipart/form-data': {
        'type': 'object',
        'properties': {
            'file': {'type': 'string', 'format': 'binary', 'description': 'PDF, DOCX or image file'},
            'query': {'type': 'string', 'description': 'User question about the document'},
            'category': {'type': 'string', 'description': 'Document category (defaults to General)'},
            'method': {'type': 'string', 'enum': ['semantic', 'keyword', 'hybrid'], 'description': 'Retrieval method'},
            'top_k': {'type': 'integer', 'description': 'Number of chunks to retrieve'}
        },
        'required': ['file', 'query']
    }},
    responses={
        202: OpenApiResponse(description="Task accepted"),
        400: OpenApiResponse(description="Validation or upload error"),
        500: OpenApiResponse(description="Unexpected error")
    }
)
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
        
        if not all([query, file_obj]):
            return Response({
                "error": "Missing required fields. Both 'query' and 'file' are required."
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Validate category
        category_obj, created = CategorySchema.objects.get_or_create(
            category_name=category_name,
            defaults={'description': f'Auto-created category for {category_name}'}
        )
        
        # Get file details and upload to Cloudinary
        original_filename = getattr(file_obj, 'name', 'document')
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Upload file to Cloudinary
        try:
            upload_result = cloudinary_upload(
                file_obj,
                resource_type="raw",
                public_id=f"query_documents/{unique_id}_{original_filename}",
                overwrite=True
            )
            cloudinary_url = upload_result.get('secure_url')
            
        except Exception as upload_error:
            return Response({
                "error": f"File upload failed: {str(upload_error)}"
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Create ProcessingTask record
        task_record = ProcessingTask.objects.create(
            user=request.user,
            file_name=original_filename,
            file_url=cloudinary_url,
            category=category_obj,
            query=query,
            status='pending',
            progress_message='Task created, waiting to start...'
        )
        
        logger.info(f"Created ProcessingTask record with ID: {task_record.id}")
        
        # Start the async task
        celery_task = process_document_async.delay(
            str(task_record.id),
            cloudinary_url,
            query
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
        }, status=status.HTTP_202_ACCEPTED)
            
    except Exception as e:
        logger.error(f"Unexpected error in query_document: {str(e)}", exc_info=True)
        return Response({
            "error": "An unexpected error occurred while creating the processing task"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    methods=['GET'],
    summary="Get task status",
    description="Retrieve processing status and (when complete) the extracted results for a previously created document processing task.",
    parameters=[
        OpenApiParameter(name='task_id', description='UUID of the processing task', required=True, type={'type': 'string', 'format': 'uuid'})
    ],
    tags=["Processing"],
    responses={
        200: OpenApiResponse(description="Task status returned"),
        404: OpenApiResponse(description="Task not found"),
        500: OpenApiResponse(description="Unexpected error")
    }
)
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
            
            # Try to get the associated SubmittedFile for additional details
            submitted_file_id = task_record.result.get('submitted_file_id')
            if submitted_file_id:
                try:
                    from .models import SubmittedFile
                    submitted_file = SubmittedFile.objects.get(id=submitted_file_id, uploaded_by=request.user)
                    response_data['submitted_file'] = {
                        'id': submitted_file.id,
                        'ai_response': submitted_file.ai_response,
                        'accuracy_score': submitted_file.accuracy_score,
                        'extracted_fields': submitted_file.extracted_fields,
                        'processing_metadata': submitted_file.processing_metadata,
                        'file_url': str(submitted_file.file),
                        'processed_at': submitted_file.processed_at.isoformat() if submitted_file.processed_at else None
                    }
                except SubmittedFile.DoesNotExist:
                    logger.warning(f"SubmittedFile {submitted_file_id} not found for task {task_id}")
        
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