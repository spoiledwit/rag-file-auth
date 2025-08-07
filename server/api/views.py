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
    MLReferenceFile,
    AuditLog,
    CategorySchema,
    ProcessedFile
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
            }
        }
    })


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

        reference = MLReferenceFile.objects.create(
            ml_reference_id=ml_reference_id,
            file=cloudinary_url,
            file_name=final_filename,
            category=category_obj,
            description=description,
            reasoning_notes=reasoning_notes,
            metadata=metadata or {},
            uploaded_by=request.user
        )

        AuditLog.objects.create(
            action='upload',
            user=request.user,
            ml_reference_file=reference,
            details={"category": category_name},
            ip_address=request.META.get('REMOTE_ADDR'),
            user_agent=request.META.get('HTTP_USER_AGENT')
        )

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
                category=category_name,
                uploaded_by=request.user,
                status='processing',
                extracted_fields={}
            )

            # Create AuditLog for upload
            AuditLog.objects.create(
                action='upload',
                user=request.user,
                submitted_file=submitted_file,
                details={
                    'category': category_name,
                    'file_url': cloudinary_url,
                    'file_name': final_filename,
                    'metadata': metadata
                },
                ip_address=request.META.get('REMOTE_ADDR'),
                user_agent=request.META.get('HTTP_USER_AGENT')
            )

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

                # Create ProcessedFile record
                processed_file = ProcessedFile.objects.create(
                    submitted_file=submitted_file,
                    extracted_text=extraction_results.get('text', ''),
                    extracted_metadata={
                        'pages': extraction_results.get('pages', 0),
                        'images_found': extraction_results.get('images_found', 0),
                        'images_processed': extraction_results.get('images_processed', 0),
                        'method': extraction_results.get('method', 'unknown'),
                        'file_type': extraction_results.get('file_type', 'unknown'),
                        'output_file_path': output_file_path  # Store the path to the .txt file
                    },
                    status='completed'
                )

                # Create AuditLog for processing
                AuditLog.objects.create(
                    action='processing_completed',
                    user=request.user,
                    submitted_file=submitted_file,
                    details={
                        'extractor_metadata': extraction_results.get('extracted_metadata', {}),
                        'output_file_path': output_file_path,
                        'rag_ingested': rag_success if 'rag_success' in locals() else False
                    },
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT')
                )

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

                ProcessedFile.objects.create(
                    submitted_file=submitted_file,
                    status='failed',
                    error_message=str(processing_error)
                )

                AuditLog.objects.create(
                    action='file_processing_failed',
                    user=request.user,
                    submitted_file=submitted_file,
                    details={
                        'error': str(processing_error),
                        'category': category_name,
                        'file_name': final_filename,
                        'file_url': cloudinary_url
                    },
                    ip_address=request.META.get('REMOTE_ADDR'),
                    user_agent=request.META.get('HTTP_USER_AGENT')
                )

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
    """Handle RAG questions from frontend"""
    try:
        question = request.data.get('question')
        if not question:
            return Response({"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Import your RAG function
        from .rag_utils import rag_chat
        
        # Get response from RAG system
        response = rag_chat(question)
        
        return Response({
            "answer": response['answer'],
            "sources": response['doc_sources'],
            "confidence": response['evaluation']
        })
        
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)