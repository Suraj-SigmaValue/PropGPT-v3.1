"""
API URL Configuration
"""
from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('comparison-items/', views.get_items, name='get_items'),
    path('query/', views.process_query, name='process_query'),
    path('feedback/', views.submit_feedback, name='submit_feedback'),
    path('cache/stats/', views.cache_statistics, name='cache_stats'),
    path('cache/clear/', views.clear_cache, name='clear_cache'),
]
