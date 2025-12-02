"""
Try loading pickle files using joblib instead of pickle
"""
import joblib
import pickle
from pathlib import Path

def try_load_with_joblib(filepath):
    """Try loading with joblib"""
    print(f"\nTrying to load: {filepath}")
    print("-" * 70)
    
    # Method 1: joblib.load
    try:
        data = joblib.load(filepath)
        print(f"✅ SUCCESS with joblib.load()")
        print(f"   Type: {type(data).__name__}")
        return True, data
    except Exception as e:
        print(f"❌ joblib.load() failed: {str(e)[:60]}")
    
    # Method 2: Check if file is compressed
    try:
        import gzip
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ SUCCESS - File was gzip compressed!")
        print(f"   Type: {type(data).__name__}")
        return True, data
    except Exception as e:
        print(f"❌ gzip.open() failed: {str(e)[:60]}")
    
    # Method 3: Check first few bytes to diagnose
    try:
        with open(filepath, 'rb') as f:
            first_bytes = f.read(10)
            print(f"First 10 bytes (hex): {first_bytes.hex()}")
            print(f"First 10 bytes (repr): {repr(first_bytes)}")
            
            # Check for common file signatures
            if first_bytes[:2] == b'\x1f\x8b':
                print("   → Detected: GZIP compressed file")
            elif first_bytes[:4] == b'PK\x03\x04':
                print("   → Detected: ZIP file")
            elif first_bytes[0] in [0x80, 0x81, 0x82, 0x83, 0x84, 0x85]:
                print(f"   → Detected: Pickle protocol {first_bytes[0] - 0x80}")
            else:
                print(f"   → Unknown format (first byte: 0x{first_bytes[0]:02x})")
    except Exception as e:
        print(f"❌ Could not read file: {e}")
    
    return False, None


def convert_all_models():
    """Try to load all model files"""
    print("="*70)
    print("TRYING JOBLIB AND ALTERNATIVE LOADING METHODS")
    print("="*70)
    
    model_dir = Path('models')
    files = [
        'disease_rf_model.pkl',
        'tfidf_vectorizer.pkl',
        'column_transformer.pkl',
        'label_encoder.pkl'
    ]
    
    results = {}
    loaded_data = {}
    
    for filename in files:
        filepath = model_dir / filename
        if filepath.exists():
            success, data = try_load_with_joblib(filepath)
            results[filename] = success
            if success:
                loaded_data[filename] = data
        else:
            print(f"\n⚠️ File not found: {filename}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nSuccessfully loaded: {success_count}/{total_count} files")
    
    if success_count > 0:
        print("\n✅ Some files loaded successfully!")
        print("\nRe-saving with standard pickle protocol 4...")
        
        output_dir = Path('models_fixed')
        output_dir.mkdir(exist_ok=True)
        
        for filename, data in loaded_data.items():
            output_path = output_dir / filename
            try:
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f, protocol=4)
                print(f"   ✓ Saved: {output_path}")
            except Exception as e:
                print(f"   ❌ Failed to save {filename}: {e}")
        
        # Copy JSON files
        import shutil
        for json_file in ['disease_rf_metadata.json', 'comprehensive_disease_profiles.json']:
            src = model_dir / json_file
            dst = output_dir / json_file
            if src.exists():
                shutil.copy(src, dst)
                print(f"   ✓ Copied: {json_file}")
        
        print(f"\n✅ Fixed models saved to: {output_dir}/")
        print("Update your code to use: model_dir='models_fixed'")
    else:
        print("\n❌ Could not load any files.")
        print("\nThe pickle files may be:")
        print("  1. Saved with Python 3.12+ (protocol 5)")
        print("  2. Corrupted during transfer")
        print("  3. Created with a different pickle library")
        print("\nRecommendation: Ask the model creator to re-export them.")


if __name__ == "__main__":
    convert_all_models()