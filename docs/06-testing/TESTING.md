# CTI-NLP System Testing Guide

This comprehensive guide covers all aspects of testing the CTI-NLP system, from quick smoke tests to comprehensive integration testing.

## Table of Contents

- [Quick Start Testing](#quick-start-testing)
- [Testing Prerequisites](#testing-prerequisites)
- [Test Categories](#test-categories)
- [Manual Testing](#manual-testing)
- [Automated Testing](#automated-testing)
- [API Testing](#api-testing)
- [Database Testing](#database-testing)
- [Docker Testing](#docker-testing)
- [Performance Testing](#performance-testing)
- [Security Testing](#security-testing)
- [Troubleshooting](#troubleshooting)
- [Continuous Testing](#continuous-testing)

## Quick Start Testing

### 1. Smoke Test (2 minutes)

Verify the basic functionality is working:

```bash
# Activate environment
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Linux/Mac

# Quick API test in test mode
$env:TEST_MODE="true"  # Windows PowerShell
export TEST_MODE="true"  # Linux/Mac/Bash

python -c "
from backend.main import app
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post('/analyze', json={'text': 'APT29 phishing campaign'})
print('API Working!' if response.status_code == 200 else '❌ API Failed!')
print('Response:', response.json() if response.status_code == 200 else response.text)
"
```

**Expected Output:**

```
API Working!
Response: {'id': 'test-mode-id', 'threat_type': 'Phishing', 'severity': 'High', ...}
```

### 2. Component Test (5 minutes)

Test individual ML components:

```bash
# Test NER
python -c "
from backend.threat_ner import extract_threat_entities
result = extract_threat_entities('APT29 targeting Microsoft Exchange servers')
print('NER Result:', result)
print('NER Working!' if isinstance(result, dict) else 'NER Failed!')
"

# Test Classifier
python -c "
from backend.classifier import classify_threat
result = classify_threat('Phishing emails targeting bank customers')
print('Classification:', result)
print('Classifier Working!' if result else 'Classifier Failed!')
"

# Test Severity Predictor
python -c "
from backend.severity_predictor import predict_severity
result = predict_severity('Critical zero-day exploit in production systems')
print('Severity:', result)
print('Severity Predictor Working!' if result else 'Severity Predictor Failed!')
"
```

## Testing Prerequisites

### Environment Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ML models
python -m spacy download en_core_web_sm

# 3. Verify models exist
ls models/  # Should show: threat_classifier.pkl, tfidf_vectorizer.pkl, etc.

# 4. Set test environment
echo "TEST_MODE=true" >> .env
```

### Database Setup (Optional for Full Testing)

```bash
# Start test database
docker run -d --name test-postgres -p 5433:5432 \
  -e POSTGRES_DB=test_cti_nlp \
  -e POSTGRES_USER=test_user \
  -e POSTGRES_PASSWORD=test_pass \
  postgres:15-alpine

# Update test environment
echo "TEST_DATABASE_URL=postgresql://test_user:test_pass@localhost:5433/test_cti_nlp" >> .env
```

## Test Categories

### 1. Unit Tests

Test individual functions and components in isolation.

```bash
# Run unit tests only
pytest tests/ -m "not integration" -v

# Run specific test file
pytest tests/test_models.py -v

# Run specific test function
pytest tests/test_models.py::TestNLPModels::test_threat_classification -v
```

### 2. Integration Tests

Test component interactions and API endpoints.

```bash
# Run integration tests only
pytest tests/ -m integration -v

# Run API integration tests
pytest tests/test_api.py -v
```

### 3. End-to-End Tests

Test complete workflows from input to output.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov=database --cov-report=html
```

## Manual Testing

### 1. API Endpoint Testing

#### Basic Analysis Endpoint

```bash
# Test with curl
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "APT29 group launching sophisticated phishing attacks against government agencies"}'
```

**Expected Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "original_text": "APT29 group launching sophisticated phishing attacks against government agencies",
  "entities": {
    "ORG": ["APT29"],
    "LOC": []
  },
  "threat_type": "Phishing",
  "severity": "High",
  "timestamp": "2025-08-26T21:42:00.000Z"
}
```

#### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

**Expected Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-08-26T21:42:00.000Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "api": "running"
  }
}
```

#### Threat Listing

```bash
# Get all threats
curl "http://localhost:8000/threats?limit=5"

# Filter by threat type
curl "http://localhost:8000/threats?threat_type=Phishing&limit=10"

# Filter by severity
curl "http://localhost:8000/threats?severity=High"
```

#### Analytics

```bash
curl "http://localhost:8000/analytics?days=30"
```

### 2. CSV Upload Testing

Create a test CSV file:

```csv
text
"Ransomware attack encrypting hospital systems with .locked extension"
"DDoS attack targeting financial services infrastructure"
"Phishing campaign using fake Microsoft Office 365 login pages"
"APT group exploiting zero-day vulnerability in VPN software"
```

Upload test:

```bash
curl -X POST "http://localhost:8000/upload_csv" \
  -F "file=@test_threats.csv" \
  -o results.json
```

### 3. Real-time Ingestion Testing

```bash
# Trigger manual ingestion
curl -X POST "http://localhost:8000/ingest_now"

# Check ingestion status
curl "http://localhost:8000/"

# View ingested feed
curl "http://localhost:8000/feed?limit=10"
```

## Automated Testing

### 1. Using pytest

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=backend --cov=database --cov=data_ingestion \
  --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest tests/ -m "unit" -v
pytest tests/ -m "integration" -v
pytest tests/ -m "slow" -v
```

### 2. Using the Test Runner

```bash
# Run comprehensive test suite
python tests/run_tests.py

# This will:
# - Install dependencies
# - Run all tests with coverage
# - Generate HTML coverage report
# - Run code quality checks
# - Check for security vulnerabilities
```

### 3. Using Makefile Commands

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run code quality checks
make lint

# Format code
make format
```

## API Testing

### 1. Interactive Testing with Swagger UI

Start the server and visit: `http://localhost:8000/docs`

1. Click on any endpoint
2. Click "Try it out"
3. Fill in parameters
4. Click "Execute"
5. Review response

### 2. Python Testing Script

Create `test_api_manual.py`:

```python
import requests
import json

BASE_URL = "http://localhost:8000"

def test_analyze_endpoint():
    """Test the analyze endpoint"""
    print("Testing /analyze endpoint...")

    test_cases = [
        "APT29 phishing campaign targeting banks",
        "Ransomware encrypting files with .locked extension",
        "DDoS attack on critical infrastructure",
        "Zero-day exploit in web browsers"
    ]

    for text in test_cases:
        response = requests.post(f"{BASE_URL}/analyze", json={"text": text})

        if response.status_code == 200:
            result = response.json()
            print(f"Text: {text[:50]}...")
            print(f"Threat: {result['threat_type']}, Severity: {result['severity']}")
        else:
            print(f"Failed: {text[:50]}...")
            print(f"Error: {response.text}")
        print()

def test_threats_endpoint():
    """Test the threats listing endpoint"""
    print("Testing /threats endpoint...")

    response = requests.get(f"{BASE_URL}/threats?limit=5")
    if response.status_code == 200:
        result = response.json()
        print(f"Retrieved {result['count']} threats")
        for threat in result['threats'][:3]:
            print(f"   - {threat['threat_type']}: {threat['original_text'][:50]}...")
    else:
        print(f"Failed to retrieve threats: {response.text}")

def test_analytics_endpoint():
    """Test the analytics endpoint"""
    print("Testing /analytics endpoint...")

    response = requests.get(f"{BASE_URL}/analytics")
    if response.status_code == 200:
        result = response.json()
        print(f"Analytics retrieved:")
        print(f"Total threats: {result['total_threats']}")
        print(f"By category: {result['threats_by_category']}")
    else:
        print(f"Failed to retrieve analytics: {response.text}")

if __name__ == "__main__":
    test_analyze_endpoint()
    test_threats_endpoint()
    test_analytics_endpoint()
```

Run with:

```bash
python test_api_manual.py
```

### 3. Load Testing with Multiple Requests

```python
import concurrent.futures
import requests
import time

def load_test():
    """Simple load test"""
    url = "http://localhost:8000/analyze"

    def make_request(i):
        response = requests.post(url, json={"text": f"Test threat {i}"})
        return response.status_code == 200

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(50)]
        results = [future.result() for future in futures]

    end_time = time.time()

    success_rate = sum(results) / len(results) * 100
    print(f"Load Test Results:")
    print(f"  Requests: {len(results)}")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Time: {end_time - start_time:.2f}s")
    print(f"  RPS: {len(results) / (end_time - start_time):.1f}")

if __name__ == "__main__":
    load_test()
```

## Database Testing

### 1. Database Connection Test

```python
# Test database connectivity
python -c "
from database.database import get_database_config
try:
    db_config = get_database_config()
    db = next(db_config.get_db())
    db.execute('SELECT 1')
    print('Database connection successful')
    db.close()
except Exception as e:
    print(f'Database connection failed: {e}')
"
```

### 2. Database Operations Test

```python
# Test CRUD operations
python -c "
from database.database import get_database_config
from database.services import ThreatIntelService
from database.models import DataSource, ThreatCategory, SeverityLevel

try:
    db_config = get_database_config()
    db = next(db_config.get_db())
    service = ThreatIntelService(db)

    # Create
    threat = service.create_threat_intel(
        source=DataSource.MANUAL,
        original_text='Test threat for database testing',
        threat_type=ThreatCategory.PHISHING,
        severity=SeverityLevel.HIGH
    )
    print(f'Created threat: {threat.id}')

    # Read
    retrieved = service.get_threat_intel(str(threat.id))
    print(f'Retrieved threat: {retrieved.original_text}')

    # Update
    updated = service.update_threat_intel(str(threat.id), severity=SeverityLevel.CRITICAL)
    print(f'Updated severity: {updated.severity}')

    # Delete
    deleted = service.delete_threat_intel(str(threat.id))
    print(f'Deleted threat: {deleted}')

    db.close()
    print('All database operations successful')

except Exception as e:
    print(f'Database operations failed: {e}')
"
```

### 3. Redis Cache Test

```python
# Test Redis connectivity
python -c "
from database.database import get_redis_client
try:
    redis_client = get_redis_client()
    redis_client.set('test_key', 'test_value')
    result = redis_client.get('test_key')
    redis_client.delete('test_key')
    print(f'Redis test successful: {result}')
except Exception as e:
    print(f'Redis test failed: {e}')
"
```

## Docker Testing

### 1. Docker Build Test

```bash
# Test Docker build
docker build -t cti-nlp-test .

# Verify image
docker images | grep cti-nlp-test
```

### 2. Docker Compose Test

```bash
# Start services
docker-compose up -d

# Check service status
docker-compose ps

# Test service health
docker-compose exec app curl -f http://localhost:8000/health

# View logs
docker-compose logs app

# Stop services
docker-compose down
```

### 3. Container Integration Test

```bash
# Test complete Docker workflow
docker-compose up -d

# Wait for services to start
sleep 30

# Test API through Docker
curl "http://localhost:8000/health"
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "Docker test threat"}'

# Cleanup
docker-compose down
```

## Performance Testing

### 1. Response Time Testing

```python
import requests
import time
import statistics

def measure_response_times():
    """Measure API response times"""
    url = "http://localhost:8000/analyze"
    times = []

    for i in range(10):
        start = time.time()
        response = requests.post(url, json={"text": f"Performance test {i}"})
        end = time.time()

        if response.status_code == 200:
            times.append((end - start) * 1000)  # Convert to ms

    if times:
        print(f"Response Time Statistics (ms):")
        print(f"  Average: {statistics.mean(times):.2f}")
        print(f"  Median: {statistics.median(times):.2f}")
        print(f"  Min: {min(times):.2f}")
        print(f"  Max: {max(times):.2f}")
        print(f"  95th percentile: {sorted(times)[int(0.95 * len(times))]:.2f}")

if __name__ == "__main__":
    measure_response_times()
```

### 2. Memory Usage Test

```bash
# Monitor memory usage during testing
docker stats --no-stream | grep cti

# Or for detailed monitoring
docker exec cti-app ps aux
```

## Security Testing

### 1. Input Validation Test

```python
# Test malicious inputs
test_cases = [
    "",  # Empty input
    "A" * 10000,  # Very long input
    "<script>alert('xss')</script>",  # XSS attempt
    "'; DROP TABLE threat_intel; --",  # SQL injection attempt
    "\x00\x01\x02",  # Binary data
    "{'malicious': 'json'}",  # Malformed JSON in text
]

for i, test_case in enumerate(test_cases):
    response = requests.post("http://localhost:8000/analyze",
                           json={"text": test_case})
    print(f"Test {i+1}: Status {response.status_code}")
```

### 2. Rate Limiting Test

```python
# Test rate limiting
import time

for i in range(100):
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 429:
        print(f"Rate limited after {i+1} requests")
        break
    time.sleep(0.1)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Errors

**Problem:** `FileNotFoundError: models/threat_classifier.pkl`

**Solution:**

```bash
# Train missing models
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py

# Verify models exist
ls models/
```

#### 2. Database Connection Errors

**Problem:** `Connection refused on port 5432`

**Solution:**

```bash
# Start database
docker-compose up -d postgres

# Check if running
docker ps | grep postgres

# Test connection
telnet localhost 5432
```

#### 3. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'backend'`

**Solution:**

```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### 4. Test Failures

**Problem:** Tests failing with 500 errors

**Solution:**

```bash
# Run in test mode
export TEST_MODE=true

# Check logs
tail -f logs/cti-nlp.log

# Run individual test
pytest tests/test_api.py::TestThreatIntelAPI::test_analyze_text_success -v
```

### Debug Mode Testing

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with detailed output
uvicorn backend.main:app --reload --log-level debug

# View debug logs
tail -f logs/debug.log
```

## Continuous Testing

### 1. Pre-commit Testing

Create `.pre-commit-hooks.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/, -x, -v]
```

### 2. GitHub Actions Testing

Create `.github/workflows/test.yml`:

```yaml
name: Test CTI-NLP System

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_pass
          POSTGRES_USER: test_user
          POSTGRES_DB: test_cti_nlp
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -m spacy download en_core_web_sm

      - name: Run tests
        run: |
          export TEST_MODE=true
          pytest tests/ --cov=backend --cov=database
```

### 3. Automated Testing Schedule

```bash
# Create cron job for regular testing
crontab -e

# Add line for daily testing at 2 AM
0 2 * * * cd /path/to/cti-nlp-system && python tests/run_tests.py
```

## Test Reporting

### 1. Coverage Report

```bash
# Generate HTML coverage report
pytest tests/ --cov=backend --cov=database --cov-report=html

# View report
open htmlcov/index.html  # Mac
start htmlcov/index.html  # Windows
```

### 2. Test Results Export

```bash
# Generate JUnit XML report
pytest tests/ --junitxml=test-results.xml

# Generate JSON report
pytest tests/ --json-report --json-report-file=test-results.json
```

---

## Testing Checklist

Before considering the system fully tested, ensure:

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] API endpoints return expected responses
- [ ] Database operations work correctly
- [ ] Docker containers start and communicate
- [ ] Performance meets requirements (< 200ms API response)
- [ ] Security tests show no vulnerabilities
- [ ] Load testing shows acceptable performance
- [ ] Error handling works for edge cases
- [ ] Documentation examples work as described

## Quick Testing Commands Reference

```bash
# Essential tests
make test                    # Run all tests
make test-unit              # Unit tests only
make test-integration       # Integration tests only
python tests/run_tests.py   # Comprehensive test suite

# Manual verification
curl http://localhost:8000/health                    # Health check
curl -X POST http://localhost:8000/analyze \         # API test
  -H "Content-Type: application/json" \
  -d '{"text": "test threat"}'

# Docker testing
docker-compose up -d         # Start all services
docker-compose ps           # Check status
docker-compose logs app     # View logs
docker-compose down         # Stop services

# Performance testing
python -c "import test_scripts; test_scripts.load_test()"  # Load test
```

---

_Happy Testing! _

For additional support, refer to the [API Documentation](API.md) or [Deployment Guide](DEPLOYMENT.md).
