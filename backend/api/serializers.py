"""
API Serializers for PropGPT
"""
from rest_framework import serializers


class QueryRequestSerializer(serializers.Serializer):
    """Serializer for query requests."""
    query = serializers.CharField(required=True, max_length=5000)
    comparison_type = serializers.ChoiceField(
        choices=['Location', 'City', 'Project'],
        required=True
    )
    items = serializers.ListField(
        child=serializers.CharField(max_length=200),
        required=True,
        min_length=1,
        max_length=5
    )
    categories = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=True,
        min_length=1
    )
    mapping_llm_provider = serializers.ChoiceField(
        choices=['openai', 'gemini'],
        default='openai',
        required=False
    )
    response_llm_provider = serializers.ChoiceField(
        choices=['openai', 'gemini'],
        default='openai',
        required=False
    )
    stream = serializers.BooleanField(default=True, required=False)


class QueryResponseSerializer(serializers.Serializer):
    """Serializer for query responses."""
    response = serializers.CharField()
    mapping_keys = serializers.ListField(child=serializers.CharField())
    selected_columns = serializers.ListField(child=serializers.CharField())
    input_tokens = serializers.IntegerField()
    output_tokens = serializers.IntegerField()
    total_tokens = serializers.IntegerField()
    mapping_provider = serializers.CharField()
    response_provider = serializers.CharField()
    mapping_model = serializers.CharField()
    response_model = serializers.CharField()
    cache_hit = serializers.BooleanField()
    retrieved_sources = serializers.ListField(required=False)


class ComparisonItemsRequestSerializer(serializers.Serializer):
    """Serializer for comparison items request."""
    comparison_type = serializers.ChoiceField(
        choices=['Location', 'City', 'Project'],
        required=True
    )
    search = serializers.CharField(required=False, allow_blank=True)
    limit = serializers.IntegerField(required=False, default=100)


class ComparisonItemsResponseSerializer(serializers.Serializer):
    """Serializer for comparison items response."""
    comparison_type = serializers.CharField()
    items = serializers.ListField(child=serializers.CharField())
    count = serializers.IntegerField()


class FeedbackRequestSerializer(serializers.Serializer):
    """Serializer for feedback requests."""
    feedback_type = serializers.ChoiceField(
        choices=['thumbs_up', 'thumbs_down'],
        required=True
    )
    query = serializers.CharField(required=True)
    items = serializers.ListField(child=serializers.CharField(), required=True)
    categories = serializers.ListField(child=serializers.CharField(), required=True)
    mapping_keys = serializers.ListField(child=serializers.CharField(), required=True)
    comparison_type = serializers.CharField(required=True)
    provider = serializers.CharField(required=True)


class FeedbackResponseSerializer(serializers.Serializer):
    """Serializer for feedback responses."""
    status = serializers.CharField()
    message = serializers.CharField()
    corrected_response = serializers.CharField(required=False)
    new_mapping_keys = serializers.ListField(child=serializers.CharField(), required=False)


class CacheStatsResponseSerializer(serializers.Serializer):
    """Serializer for cache statistics."""
    total_entries = serializers.IntegerField()
    cache_dir = serializers.CharField()
    similarity_threshold = serializers.FloatField()
    ttl_seconds = serializers.IntegerField()


class HealthCheckResponseSerializer(serializers.Serializer):
    """Serializer for health check response."""
    status = serializers.CharField()
    version = serializers.CharField()
    data_loaded = serializers.BooleanField()
    excel_file_exists = serializers.BooleanField()
    pickle_file_exists = serializers.BooleanField()
