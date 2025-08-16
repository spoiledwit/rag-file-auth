from django.contrib.auth.models import User
from django.db import models
from cloudinary.models import CloudinaryField
import uuid


class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('user', 'Portal User'), 
        ('admin', 'Client Admin'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"
    
    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"


class CategorySchema(models.Model):
    category_name = models.CharField(max_length=100, unique=True)
    prompt = models.TextField(blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.category_name
    
    class Meta:
        verbose_name = "Category Schema"
        verbose_name_plural = "Category Schemas"
        ordering = ['-created_at']

class SubmittedFile(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending Analysis'),
        ('processing', 'Being Processed'),
        ('completed', 'Analysis Complete'),
        ('failed', 'Processing Failed'),
    ]
    
    MATCH_CHOICES = [
        ('Y', 'Yes'),
        ('N', 'No'),
    ]
    
    file = CloudinaryField('submitted_file')
    file_name = models.CharField(max_length=255)
    category = models.ForeignKey(
        CategorySchema,
        on_delete=models.CASCADE,
        related_name='submitted_files'
    )

    uploaded_by = models.ForeignKey(
        User, 
        on_delete=models.CASCADE,
        limit_choices_to={'userprofile__role': 'user'}
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # AI Analysis Results
    accuracy_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Accuracy score from 0-100"
    )

    extracted_fields = models.JSONField(
        default=dict, 
        blank=True,
    )
    
    # Document processing fields
    extracted_text = models.TextField(
        blank=True,
        help_text="Full extracted text from the document"
    )
    
    query = models.TextField(
        blank=True,
        help_text="User query about the document"
    )
    
    ai_response = models.TextField(
        blank=True,
        help_text="AI-generated response to the query"
    )
    
    processing_metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="Processing metadata like chunks, retrieval method, evaluation scores"
    )
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    processed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"{self.file_name}"
    
    class Meta:
        verbose_name = "Submitted File"
        verbose_name_plural = "Submitted Files"
        ordering = ['-uploaded_at']


from django.db.models.signals import post_save
from django.dispatch import receiver

class ProcessingTask(models.Model):
    TASK_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task_id = models.CharField(max_length=255, unique=True, help_text="Celery task ID")
    task_type = models.CharField(max_length=50, default='document_processing')
    status = models.CharField(max_length=20, choices=TASK_STATUS_CHOICES, default='pending')
    
    # User and file information
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    file_url = models.URLField()
    category = models.ForeignKey(CategorySchema, on_delete=models.CASCADE)
    query = models.TextField()
    
    # Processing configuration
    method = models.CharField(max_length=20, default='hybrid')
    top_k = models.IntegerField(default=30)
    
    # Results
    result = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Progress tracking
    progress_percentage = models.IntegerField(default=0)
    progress_message = models.CharField(max_length=255, blank=True)
    
    def __str__(self):
        return f"Task {self.id} - {self.status}"
    
    class Meta:
        verbose_name = "Processing Task"
        verbose_name_plural = "Processing Tasks"
        ordering = ['-created_at']


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)