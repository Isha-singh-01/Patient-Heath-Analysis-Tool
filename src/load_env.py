"""
Environment variable loader
Loads configuration from .env file
"""

import os
from pathlib import Path
from typing import Optional


def load_env(env_file: str = '.env') -> bool:
    """
    Load environment variables from .env file
    Uses python-dotenv if available, otherwise manual parsing
    
    Returns:
        bool: True if file was loaded successfully
    """
    env_path = Path(env_file)
    
    if not env_path.exists():
        print(f"⚠️  No {env_file} file found")
        print(f"   Create one by running: python load_env.py setup")
        return False
    
    # Try using python-dotenv first (more robust)
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        print(f"✅ Loaded environment from {env_file} (via python-dotenv)")
        return True
    except ImportError:
        # Fallback to manual parsing
        pass
    
    # Manual parsing
    loaded_count = 0
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Skip placeholder values
                    if value and not value.startswith('your_'):
                        os.environ[key] = value
                        loaded_count += 1
        
        print(f"✅ Loaded {loaded_count} environment variables from {env_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading {env_file}: {e}")
        return False


def get_config() -> dict:
    """
    Get configuration from environment variables
    
    Returns:
        dict: Configuration dictionary
    """
    config = {
        'hf_token': os.getenv('HF_TOKEN') or os.getenv('HF_API_KEY'),
        'hf_model': os.getenv('HF_MODEL', 'meta-llama/Llama-3.3-70B-Instruct:groq'),
        'use_llm': os.getenv('USE_LLM', 'true').lower() == 'true',
        'model_dir': os.getenv('MODEL_DIR', 'models_fixed'),
        'profiles_path': os.getenv('PROFILES_PATH', 'models_fixed/comprehensive_disease_profiles.json'),
        'openai_api_key': os.getenv('OPENAI_API_KEY')
    }
    
    return config


def setup_env_file():
    """
    Interactive setup to create .env file
    """
    print("="*70)
    print("ENVIRONMENT FILE SETUP")
    print("="*70)
    
    env_path = Path('.env')
    
    if env_path.exists():
        print(f"\n⚠️  .env file already exists")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    print("\n📝 Creating .env file...")
    print("\nGet your HuggingFace token from: https://huggingface.co/settings/tokens")
    print("Make sure to enable 'Make calls to Inference Providers' permission!")
    print("\nPress Enter to skip optional fields.\n")
    
    hf_token = input("HuggingFace Token (HF_TOKEN): ").strip()
    
    if not hf_token:
        print("\n⚠️  No token provided. LLM will be disabled.")
        hf_token = "your_huggingface_token_here"
        use_llm = "false"
    else:
        use_llm = "true"
        print("✅ Token will be saved")
    
    hf_model = input("Model (default: meta-llama/Llama-3.3-70B-Instruct:groq): ").strip()
    if not hf_model:
        hf_model = "meta-llama/Llama-3.3-70B-Instruct:groq"
    
    # Create .env file
    with open(env_path, 'w') as f:
        f.write("# HuggingFace API Configuration\n")
        f.write(f"HF_TOKEN={hf_token}\n")
        f.write(f"HF_API_KEY={hf_token}\n\n")
        
        f.write("# Model Configuration\n")
        f.write(f"HF_MODEL={hf_model}\n\n")
        
        f.write("# Application Settings\n")
        f.write(f"USE_LLM={use_llm}\n")
        f.write("MODEL_DIR=models_fixed\n")
        f.write("PROFILES_PATH=models_fixed/comprehensive_disease_profiles.json\n")
    
    print(f"\n✅ Created {env_path}")
    print("\nYou can now run your application!")
    print("The .env file will be loaded automatically.\n")
    
    # Add to .gitignore
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
        
        if '.env' not in content:
            with open(gitignore_path, 'a') as f:
                f.write("\n# Environment variables\n")
                f.write(".env\n")
            print("✅ Added .env to .gitignore")
    else:
        with open(gitignore_path, 'w') as f:
            f.write("# Environment variables\n")
            f.write(".env\n")
        print("✅ Created .gitignore with .env")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'setup':
        setup_env_file()
    else:
        # Test loading
        load_env()
        config = get_config()
        
        print("\nCurrent Configuration:")
        print("-"*70)
        print(f"HF Token: {'✅ Set' if config['hf_token'] else '❌ Not set'}")
        print(f"HF Model: {config['hf_model']}")
        print(f"Use LLM: {config['use_llm']}")
        print(f"Model Dir: {config['model_dir']}")
        print(f"Profiles Path: {config['profiles_path']}")