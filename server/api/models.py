from django.contrib.auth.models import User
from django.db import models
from cloudinary.models import CloudinaryField


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

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)