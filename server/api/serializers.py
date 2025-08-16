from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, CategorySchema, SubmittedFile

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']
# ProcessedFileSerializer removed - model no longer exists

class SubmittedFileSerializer(serializers.ModelSerializer):
    uploaded_by = UserSerializer(read_only=True)

    class Meta:
        model = SubmittedFile
        fields = [
            'id', 'file', 'file_name', 'category',
            'uploaded_by', 'uploaded_at', 'accuracy_score', 'extracted_fields',
            'status', 'processed_at', 'error_message'
        ]        

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    class Meta:
        model = UserProfile
        fields = ['id', 'user', 'role', 'created_at']

class CategorySchemaSerializer(serializers.ModelSerializer):
    class Meta:
        model = CategorySchema
        fields = ['id', 'category_name', 'prompt', 'description', 'created_at', 'updated_at']

# MLReferenceFileSerializer and AuditLogSerializer removed - models no longer exist 