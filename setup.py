#!/usr/bin/env python3
"""
Setup script for CTI-NLP System
This script initializes the development environment and checks all dependencies
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("Python 3.9+ is required")
        return False
    print(f"Python {sys.version.split()[0]} detected")
    return True

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("Docker not found or not running")
    return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'models', 
        'logs',
        'nginx/ssl'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def setup_environment():
    """Setup environment variables"""
    env_example = Path('.env.example')
    env_file = Path('.env')
    
    if not env_file.exists() and env_example.exists():
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            content = src.read()
            # Generate a random secret key
            import secrets
            secret_key = secrets.token_urlsafe(32)
            content = content.replace('your-secret-key-here-change-in-production', secret_key)
            dst.write(content)
        print("Created .env file from template")
    else:
        print("ℹ.env file already exists")

def install_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        print("Python dependencies installed")
        
        # Install spaCy model
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], 
                      check=True, capture_output=True)
        print("spaCy English model downloaded")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def check_models():
    """Check if ML models exist"""
    model_files = [
        'models/threat_classifier.pkl',
        'models/tfidf_vectorizer.pkl',
        'models/severity_model.pkl',
        'models/severity_vectorizer.pkl'
    ]
    
    missing_models = []
    for model_file in model_files:
        if not Path(model_file).exists():
            missing_models.append(model_file)
    
    if missing_models:
        print(f" Missing model files: {missing_models}")
        print("   Run training scripts to generate models:")
        print("   python scripts/train_threat_classifier.py")
        print("   python scripts/train_severity_model.py")
        return False
    
    print("All model files found")
    return True

def test_database_connection():
    """Test database connection"""
    try:
        # This would require the database to be running
        # For now, just check if the database module can be imported
        from database.database import get_database_config
        print("Database module loaded successfully")
        return True
    except ImportError as e:
        print(f"Database module import failed: {e}")
        return False

def create_systemd_service():
    """Create systemd service file for production deployment"""
    service_content = """[Unit]
Description=CTI-NLP System
After=docker.service
Requires=docker.service

[Service]
Type=forking
User=cti-nlp
WorkingDirectory=/home/cti-nlp/cti-nlp-system
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
Restart=on-failure
StartLimitInterval=60s
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
"""
    
    with open('cti-nlp.service', 'w') as f:
        f.write(service_content)
    
    print("Created systemd service file: cti-nlp.service")
    print("   To install: sudo cp cti-nlp.service /etc/systemd/system/")
    print("   To enable: sudo systemctl enable cti-nlp")

def run_quick_test():
    """Run a quick test to verify setup"""
    try:
        # Test import of main modules
        from backend.threat_ner import extract_threat_entities
        from backend.classifier import classify_threat
        from backend.severity_predictor import predict_severity
        
        # Quick functionality test
        test_text = "Test phishing attack"
        entities = extract_threat_entities(test_text)
        threat_type = classify_threat(test_text)
        severity = predict_severity(test_text)
        
        print("Quick functionality test passed")
        print(f"   Entities: {len(entities)} types found")
        print(f"   Threat type: {threat_type}")
        print(f"   Severity: {severity}")
        return True
        
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("CTI-NLP System Setup")
    print("=" * 40)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Check Docker
    docker_available = check_docker()
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Check models
    models_available = check_models()
    
    # Test database connection
    test_database_connection()
    
    # Create systemd service
    create_systemd_service()
    
    # Run quick test
    if models_available:
        run_quick_test()
    
    print("\n" + "=" * 40)
    
    if success and docker_available and models_available:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start services: docker-compose up -d")
        print("2. Visit API docs: http://localhost:8000/docs")
        print("3. Run tests: python tests/run_tests.py")
    else:
        print("Setup completed with warnings.")
        print("\nPlease address the issues above before running the system.")
        
        if not models_available:
            print("\nTo train models:")
            print("   python scripts/train_threat_classifier.py")
            print("   python scripts/train_severity_model.py")
        
        if not docker_available:
            print("\nTo install Docker:")
            print("   Visit: https://docs.docker.com/get-docker/")

if __name__ == "__main__":
    main()
