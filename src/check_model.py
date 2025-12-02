import pickle
import json
from pathlib import Path

def check_pickle_file(filepath):
    """Check if a pickle file can be loaded"""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return True, type(data).__name__, None
    except Exception as e:
        return False, None, str(e)

def check_json_file(filepath):
    """Check if a JSON file can be loaded"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return True, type(data).__name__, len(data) if isinstance(data, (list, dict)) else None
    except Exception as e:
        return False, None, str(e)

def diagnose_models(model_dir='models'):
    """Diagnose all model files"""
    print("="*70)
    print("MODEL FILE DIAGNOSTICS")
    print("="*70)
    
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        print(f"\n❌ Models directory not found: {model_dir}")
        return
    
    print(f"\n📁 Checking directory: {model_dir.absolute()}")
    print()
    
    # Expected files
    pickle_files = [
        'disease_rf_model.pkl',
        'random_forest_model.pkl',
        'tfidf_vectorizer.pkl',
        'column_transformer.pkl',
        'label_encoder.pkl'
    ]
    
    json_files = [
        'disease_rf_metadata.json',
        'disease_names.json',
        'comprehensive_disease_profiles.json'
    ]
    
    print("PICKLE FILES:")
    print("-" * 70)
    found_pickle = False
    for filename in pickle_files:
        filepath = model_dir / filename
        if filepath.exists():
            found_pickle = True
            success, dtype, error = check_pickle_file(filepath)
            size_mb = filepath.stat().st_size / (1024 * 1024)
            
            if success:
                print(f"✓ {filename}")
                print(f"  Type: {dtype}")
                print(f"  Size: {size_mb:.2f} MB")
            else:
                print(f"❌ {filename}")
                print(f"  Error: {error}")
                print(f"  Size: {size_mb:.2f} MB")
            print()
    
    if not found_pickle:
        print("❌ No pickle files found")
        print()
    
    print("JSON FILES:")
    print("-" * 70)
    found_json = False
    for filename in json_files:
        filepath = model_dir / filename
        if filepath.exists():
            found_json = True
            success, dtype, length = check_json_file(filepath)
            size_kb = filepath.stat().st_size / 1024
            
            if success:
                print(f"✓ {filename}")
                print(f"  Type: {dtype}")
                if length:
                    print(f"  Items: {length}")
                print(f"  Size: {size_kb:.2f} KB")
            else:
                print(f"❌ {filename}")
                print(f"  Error: {error}")
            print()
    
    if not found_json:
        print("❌ No JSON files found")
        print()
    
    print("ALL FILES IN DIRECTORY:")
    print("-" * 70)
    all_files = sorted(model_dir.glob('*'))
    if all_files:
        for filepath in all_files:
            if filepath.is_file():
                size = filepath.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.2f} MB"
                else:
                    size_str = f"{size / 1024:.2f} KB"
                print(f"  {filepath.name} ({size_str})")
    else:
        print("  (empty directory)")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    # Check for required files
    required_files = {
        'rf_model': ['disease_rf_model.pkl', 'random_forest_model.pkl'],
        'tfidf': ['tfidf_vectorizer.pkl'],
        'column_transformer': ['column_transformer.pkl'],
        'label_encoder': ['label_encoder.pkl'],
        'metadata': ['disease_rf_metadata.json', 'disease_names.json'],
        'profiles': ['comprehensive_disease_profiles.json']
    }
    
    missing = []
    for key, files in required_files.items():
        found = any((model_dir / f).exists() for f in files)
        if not found:
            missing.append(f"{key} ({', '.join(files)})")
    
    if missing:
        print("\n❌ Missing required files:")
        for item in missing:
            print(f"  - {item}")
        print("\nPlease ensure all model files are in the models/ directory.")
    else:
        print("\n✓ All required files present")
        print("\nIf you're getting pickle errors, try:")
        print("  1. Re-train the models with your current Python version")
        print("  2. Or load with: pickle.load(f, encoding='latin1')")


def try_load_with_encoding(filepath):
    """Try loading pickle with different encodings"""
    print(f"\nTrying different methods to load: {filepath}")
    print("-" * 70)
    
    encodings = ['ASCII', 'latin1', 'bytes']
    
    for encoding in encodings:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f, encoding=encoding)
            print(f"✓ SUCCESS with encoding='{encoding}'")
            print(f"  Type: {type(data).__name__}")
            return True, encoding, data
        except Exception as e:
            print(f"❌ Failed with encoding='{encoding}': {str(e)[:50]}")
    
    return False, None, None


if __name__ == "__main__":
    diagnose_models()
    
    # If you have a specific problematic file, test it:
    problem_file = Path('models/disease_rf_model.pkl')
    if not problem_file.exists():
        problem_file = Path('models/random_forest_model.pkl')
    
    if problem_file.exists():
        print("\n" + "="*70)
        print("TESTING ALTERNATIVE LOADING METHODS")
        print("="*70)
        try_load_with_encoding(problem_file)