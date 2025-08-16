from django.contrib import admin
from django.contrib.auth.models import User, Group
from django.utils.html import format_html
from unfold.admin import ModelAdmin
from .models import UserProfile, CategorySchema, SubmittedFile

# Unregister the default User and Group admin
admin.site.unregister(User)
admin.site.unregister(Group)

# User admin with Unfold
@admin.register(User)
class UserAdmin(ModelAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'is_staff']
    list_filter = ['is_staff', 'is_active']
    search_fields = ['username', 'email']
    list_display_links = ['username']

# Group admin with Unfold
@admin.register(Group)
class GroupAdmin(ModelAdmin):
    list_display = ['name']
    search_fields = ['name']
    list_display_links = ['name']

# UserProfile admin
@admin.register(UserProfile)
class UserProfileAdmin(ModelAdmin):
    list_display = ['user', 'role', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['user__username']
    readonly_fields = ['created_at']
    fieldsets = (
        (None, {
            'fields': ('user', 'role')
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
    list_display_links = ['user']

# CategorySchema admin
@admin.register(CategorySchema)
class CategorySchemaAdmin(ModelAdmin):
    list_display = ['category_name', 'description', 'created_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['category_name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    list_display_links = ['category_name']

# MLReferenceFile admin removed - model no longer exists

# SubmittedFile admin
@admin.register(SubmittedFile)
class SubmittedFileAdmin(ModelAdmin):
    list_display = (
        'file_name', 'category', 'file_link', 'query_short',
        'accuracy_score', 'status', 'uploaded_at'
    )
    search_fields = ('file_name', 'query', 'ai_response')
    list_filter = ('category', 'status', 'uploaded_at')
    readonly_fields = (
        'file_name', 'file', 'category', 'query', 'extracted_text', 'ai_response',
        'accuracy_score', 'extracted_fields', 'processing_metadata', 'uploaded_by',
        'status', 'error_message', 'uploaded_at', 'processed_at'
    )
    fieldsets = (
        ('File Information', {
            'fields': (
                'file_name',
                'file',
                'category',
                'uploaded_by',
                'status',
                'error_message'
            )
        }),
        ('Document Processing', {
            'fields': (
                'query',
                'extracted_text',
                'ai_response',
                'processing_metadata'
            ),
            'classes': ('collapse',)
        }),
        ('Analysis Results', {
            'fields': (
                'accuracy_score',
                'extracted_fields'
            )
        }),
        ('Timestamps', {
            'fields': ('uploaded_at', 'processed_at')
        }),
    )
    list_display_links = ['file_name']
    raw_id_fields = ['uploaded_by']
    
    def query_short(self, obj):
        """Truncate query for list display."""
        if obj.query:
            return obj.query[:50] + '...' if len(obj.query) > 50 else obj.query
        return "No query"
    query_short.short_description = 'Query'

    def file_link(self, obj):
        """Display the file as a clickable link with proper URL."""
        if obj.file:
            file_url = str(obj.file)
            
            # If it's already a full URL, use it
            if file_url.startswith('http'):
                url = file_url
            else:
                # Construct proper Cloudinary URL
                if obj.file_name and '.' in obj.file_name:
                    extension = obj.file_name.split('.')[-1].lower()
                    if extension in ['doc', 'docx', 'pdf']:
                        url = f"https://res.cloudinary.com/dewqsghdi/raw/upload/{file_url}"
                    else:
                        url = f"https://res.cloudinary.com/dewqsghdi/image/upload/{file_url}"
                else:
                    url = f"https://res.cloudinary.com/dewqsghdi/raw/upload/{file_url}"
            
            return format_html('<a href="{}" target="_blank">{}</a>', url, obj.file_name or 'View File')
        return "No file"
    file_link.short_description = 'File'

    def has_add_permission(self, request):  # noqa: ARG002
        """Prevent admins from creating SubmittedFile records."""
        return False

    def has_change_permission(self, request, obj=None):  # noqa: ARG002
        """Prevent admins from editing SubmittedFile records."""
        return False

# AuditLog admin removed - model no longer exists