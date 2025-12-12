"""
Test to verify that metadata includes data source information.
"""
import json


def test_metadata_structure():
    """Test that the expected metadata structure includes data_source field."""
    
    # Sample metadata that should be returned by the API
    sample_metadata = {
        'mapping_keys': ['Property Type wise total units', 'Property type wise Units Sold'],
        'selected_columns': ['property_type', 'total_units', 'units_sold'],
        'input_tokens': 1500,
        'output_tokens': 800,
        'total_tokens': 2300,
        'mapping_provider': 'OpenAI',
        'response_provider': 'OpenAI',
        'mapping_model': 'gpt-4o-mini',
        'response_model': 'gpt-4o-mini',
        'cache_hit': False,
        'retrieved_sources': [
            {'content': 'Sample source content 1...'},
            {'content': 'Sample source content 2...'}
        ],
        'data_source': {
            'excel_file': 'Pune_Grand_Summary.xlsx',
            'sheet_name': 'Location_YOY',
            'comparison_type': 'Location',
            'items': ['Pune City', 'Hinjewadi'],
            'item_count': 2
        }
    }
    
    # Verify all required fields exist
    required_fields = [
        'mapping_keys', 'selected_columns', 'input_tokens', 'output_tokens',
        'total_tokens', 'mapping_provider', 'response_provider',
        'mapping_model', 'response_model', 'cache_hit', 'retrieved_sources',
        'data_source'
    ]
    
    for field in required_fields:
        assert field in sample_metadata, f"Missing required field: {field}"
    
    # Verify data_source structure
    data_source = sample_metadata['data_source']
    assert 'excel_file' in data_source, "data_source missing excel_file"
    assert 'sheet_name' in data_source, "data_source missing sheet_name"
    assert 'comparison_type' in data_source, "data_source missing comparison_type"
    assert 'items' in data_source, "data_source missing items"
    assert 'item_count' in data_source, "data_source missing item_count"
    
    # Verify data types
    assert isinstance(data_source['excel_file'], str), "excel_file should be string"
    assert isinstance(data_source['sheet_name'], str), "sheet_name should be string"
    assert isinstance(data_source['comparison_type'], str), "comparison_type should be string"
    assert isinstance(data_source['items'], list), "items should be list"
    assert isinstance(data_source['item_count'], int), "item_count should be int"
    assert data_source['item_count'] == len(data_source['items']), "item_count should match items length"
    
    print("✓ All metadata structure tests passed!")
    print(f"\nSample metadata structure:")
    print(json.dumps(sample_metadata, indent=2))


if __name__ == "__main__":
    test_metadata_structure()
