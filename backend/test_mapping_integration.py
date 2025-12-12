"""
Diagnostic script to test mapping integration.
Run: python test_mapping_integration.py
"""
import os
import sys
from pathlib import Path

# Setup Django
sys.path.append(str(Path(__file__).parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'propgpt_api.settings')
import django
django.setup()

from api.services.data_service import set_mappings_for_type, get_category_keys
from config import get_category_mapping

def test_mapping_integration():
    print("=" * 60)
    print("MAPPING INTEGRATION DIAGNOSTIC TEST")
    print("=" * 60)
    
    for comparison_type in ['Location', 'City', 'Project']:
        print(f"\n{'='*60}")
        print(f"Testing {comparison_type} Mapping")
        print(f"{'='*60}")
        
        # Initialize mappings
        set_mappings_for_type(comparison_type)
        
        # Get the category mapping
        cat_map = get_category_mapping(comparison_type)
        print(f"\nAvailable categories in CATEGORY_MAPPING:")
        for cat in cat_map.keys():
            print(f"  - '{cat}'")
        
        # Test each category from DEFAULT_CATEGORIES
        test_categories = ['all', 'demand', 'supply', 'price', 'demography']
        
        print(f"\nTesting category key resolution:")
        for category in test_categories:
            keys = get_category_keys(category)
            print(f"  Category '{category}': {len(keys)} keys found")
            if len(keys) == 0:
                print(f"    ⚠️  WARNING: No mapping keys found for '{category}'!")
            else:
                print(f"    ✓ Sample keys: {keys[:3]}")
        
        # Test capitalized versions (what frontend might send)
        print(f"\nTesting CAPITALIZED categories (frontend format):")
        capitalized_tests = {
            'General': 'all',
            'Demand': 'demand',
            'Supply': 'supply',
            'Price': 'price',
            'Demographics': 'demography'
        }
        
        for cap_name, expected_key in capitalized_tests.items():
            keys = get_category_keys(cap_name)
            print(f"  Category '{cap_name}': {len(keys)} keys found")
            if len(keys) == 0:
                print(f"    ❌ ERROR: No keys found! Should map to '{expected_key}'")
            else:
                print(f"    ✓ OK: Found {len(keys)} keys")

if __name__ == "__main__":
    test_mapping_integration()
