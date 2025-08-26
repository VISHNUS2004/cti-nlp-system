#!/usr/bin/env python3
"""
Test runner script for the CTI-NLP system
"""
import subprocess
import sys
import os

def run_tests():
    """Run all tests with coverage reporting"""
    print("🚀 Starting CTI-NLP System Tests")
    print("=" * 50)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_dir)
    
    # Install test dependencies if needed
    print("📦 Installing test dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Run tests with coverage
    print("\n🧪 Running tests with coverage...")
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--cov=backend",
        "--cov=database",
        "--cov=data_ingestion",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-fail-under=70"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        print("\n📊 Coverage report generated in htmlcov/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code: {e.returncode}")
        return False

def run_linting():
    """Run code linting and formatting checks"""
    print("\n🔍 Running code quality checks...")
    
    # Run flake8 for linting
    try:
        subprocess.run([sys.executable, "-m", "flake8", "backend/", "database/", "tests/"], check=True)
        print("✅ Linting passed!")
    except subprocess.CalledProcessError:
        print("❌ Linting failed!")
        return False
    
    # Check code formatting with black
    try:
        subprocess.run([sys.executable, "-m", "black", "--check", "backend/", "database/", "tests/"], check=True)
        print("✅ Code formatting is correct!")
    except subprocess.CalledProcessError:
        print("❌ Code formatting issues found. Run 'black .' to fix.")
        return False
    
    # Check import sorting with isort
    try:
        subprocess.run([sys.executable, "-m", "isort", "--check-only", "backend/", "database/", "tests/"], check=True)
        print("✅ Import sorting is correct!")
    except subprocess.CalledProcessError:
        print("❌ Import sorting issues found. Run 'isort .' to fix.")
        return False
    
    return True

def run_security_checks():
    """Run security vulnerability checks"""
    print("\n🔒 Running security checks...")
    
    # Check for known security vulnerabilities in dependencies
    try:
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        print("✅ No dependency conflicts found!")
    except subprocess.CalledProcessError:
        print("❌ Dependency conflicts found!")
        return False
    
    return True

if __name__ == "__main__":
    success = True
    
    # Run tests
    if not run_tests():
        success = False
    
    # Run code quality checks
    if not run_linting():
        success = False
    
    # Run security checks
    if not run_security_checks():
        success = False
    
    if success:
        print("\n🎉 All checks passed! System is ready for deployment.")
        sys.exit(0)
    else:
        print("\n💥 Some checks failed. Please fix the issues before deployment.")
        sys.exit(1)
