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
    CategorySchema
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


@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FormParser])
def query_document(request):
    """
    Process a document with a user query in a single request.
    
    This endpoint combines document upload, text extraction, and RAG processing
    into a single API call. Files and processing results are saved to the database.
    
    Expected form data:
    - file: The document file (PDF, DOCX, or image)
    - query: The user's question about the document
    - category: Document category (required)
    - method: Retrieval method - "semantic", "keyword", or "hybrid" (optional, defaults to "hybrid")
    - top_k: Number of chunks to retrieve (optional, defaults to 30)
    
    Returns:
    - answer: AI-generated answer based on document content
    - retrieval_method: Method used for retrieval
    - processing_time: Time taken to process the request
    - chunks_processed: Number of text chunks created from document
    - relevant_chunks: Number of relevant chunks found
    - doc_sources: Document sources used
    - evaluation: Response quality metrics
    - submitted_file_id: Database record ID
    """
    try:
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
        
        logger.info(f"Processing query-document request: '{query}' with method: {method}")
        
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
        
        # Create SubmittedFile record in database
        submitted_file = SubmittedFile.objects.create(
            file=cloudinary_url,
            file_name=original_filename,
            category=category_obj,
            uploaded_by=request.user,
            query=query,
            status='processing',
            extracted_fields={}
        )
        
        logger.info(f"Created SubmittedFile record with ID: {submitted_file.id}")
        
        # Initialize UniversalTextExtractor for document processing
        try:
            extractor = UniversalTextExtractor(
                output_dir="temp_extracts",
                use_ocr=True,
                enable_gpu=True
            )
            
            # Create a temporary file to save the uploaded file for processing
            import tempfile
            import os
            
            # Create temporary file with proper extension
            with tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=f'.{file_extension}' if file_extension else '.pdf'
            ) as temp_file:
                # Reset file pointer and write uploaded file content to temp file
                file_obj.seek(0)
                for chunk in file_obj.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            logger.info(f"Created temporary file for processing: {temp_file_path}")
            
            # Extract text from the document
            extraction_results = extractor.process_file(temp_file_path)
            
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
                logger.info("Cleaned up temporary file")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
            
            if 'error' in extraction_results:
                return Response({
                    "error": f"Document processing failed: {extraction_results['error']}"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Get extracted text
            document_text = extraction_results.get('text', '')
            if not document_text or len(document_text.strip()) < 10:
                # Update database record with error
                submitted_file.status = 'failed'
                submitted_file.error_message = "No readable text found in the document"
                submitted_file.processed_at = timezone.now()
                submitted_file.save()
                
                return Response({
                    "error": "No readable text found in the document. The document might be empty or corrupted.",
                    "submitted_file_id": submitted_file.id
                }, status=status.HTTP_400_BAD_REQUEST)
            
            logger.info(f"Extracted {len(document_text)} characters of text from document")
            
            # Update database record with extracted text
            submitted_file.extracted_text = document_text
            submitted_file.save()
            
            # Process document with query using our RAG system
            from .rag_utils import process_document_with_query
            
            rag_result = process_document_with_query(
                document_text=document_text,
                query=query,
                method=method,
                top_k=top_k
            )
            
            # Update database record with AI response and metadata
            ai_response = rag_result.get('answer', 'No answer generated')
            submitted_file.ai_response = ai_response
            
            # Extract analysis data from JSON response
            from .rag_utils import extract_analysis_data
            accuracy_score, extracted_fields = extract_analysis_data(ai_response)
            
            # Update analysis fields
            submitted_file.accuracy_score = accuracy_score
            submitted_file.extracted_fields = extracted_fields
            
            submitted_file.processing_metadata = {
                'retrieval_method': rag_result.get('retrieval_method', method),
                'processing_time': rag_result.get('processing_time', 0),
                'chunks_processed': rag_result.get('chunks_processed', 0),
                'relevant_chunks': rag_result.get('relevant_chunks', 0),
                'num_docs_retrieved': rag_result.get('num_docs_retrieved', 0),
                'doc_sources': rag_result.get('doc_sources', []),
                'evaluation': rag_result.get('evaluation', {}),
                'top_k': top_k,
                'document_info': {
                    'filename': original_filename,
                    'text_length': len(document_text),
                    'extraction_method': extraction_results.get('method', 'unknown'),
                    'pages': extraction_results.get('pages', 0),
                    'images_processed': extraction_results.get('images_processed', 0)
                }
            }
            submitted_file.status = 'completed'
            submitted_file.processed_at = timezone.now()
            submitted_file.save()
            
            logger.info(f"Updated SubmittedFile record {submitted_file.id} with AI response and metadata")
            
            # Clean up any temporary files created by extractor
            try:
                extractor.cleanup_temp_files()
                logger.info("Cleaned up extractor temporary files")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup extractor files: {cleanup_error}")
            
            # Format response
            response_data = {
                "success": True,
                "submitted_file_id": submitted_file.id,
                "query": query,
                "answer": rag_result.get('answer', 'No answer generated'),
                "accuracy_score": accuracy_score,
                "extracted_fields": extracted_fields,
                "retrieval_method": rag_result.get('retrieval_method', method),
                "processing_time": rag_result.get('processing_time', 0),
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
                },
                "file_url": cloudinary_url
            }
            
            # Check if there was an error in RAG processing
            if 'error' in rag_result:
                response_data['warning'] = f"RAG processing issue: {rag_result['error']}"
            
            logger.info(f"Successfully processed query-document request in {rag_result.get('processing_time', 0):.2f}s")
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as processing_error:
            logger.error(f"Document processing failed: {str(processing_error)}", exc_info=True)
            
            # Update database record with error
            try:
                submitted_file.status = 'failed'
                submitted_file.error_message = str(processing_error)
                submitted_file.processed_at = timezone.now()
                submitted_file.save()
            except Exception as db_error:
                logger.error(f"Failed to update database record with error: {db_error}")
            
            # Clean up any temporary files
            try:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                if 'extractor' in locals():
                    extractor.cleanup_temp_files()
            except:
                pass
            
            return Response({
                "error": f"Document processing failed: {str(processing_error)}",
                "submitted_file_id": submitted_file.id if 'submitted_file' in locals() else None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        logger.error(f"Unexpected error in query_document: {str(e)}", exc_info=True)
        return Response({
            "error": "An unexpected error occurred while processing your request"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)