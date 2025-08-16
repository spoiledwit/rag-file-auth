from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)
from . import views
from .views import (
    index, RegisterView, UserProfileView,
    submit_file, upload_ml_reference, category_list, ask_rag_question, query_document, health_check, task_status
)

app_name = 'api'

urlpatterns = [
    # Main API endpoint
    path('', index),
    
    # Health check endpoint
    path('health/', health_check, name='health_check'),
    
    # JWT Authentication endpoints
    path('auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/verify/', TokenVerifyView.as_view(), name='token_verify'),
    
    # Custom authentication endpoints
    path('auth/register/', RegisterView.as_view(), name='register'),
    path('auth/profile/', UserProfileView.as_view(), name='profile'),
    
    # API v1 endpoints
    path('v1/submit-file/', submit_file, name='submit_file'),
    path('v1/upload-ml-reference/', upload_ml_reference, name='upload_ml_reference'),
    path('v1/categories/', category_list, name='category_list'),
    path('v1/ask-rag-question/', ask_rag_question, name='ask_rag_question'),
    path('v1/query-document/', query_document, name='query_document'),
    path('v1/task/<uuid:task_id>/', task_status, name='task_status'),
]
