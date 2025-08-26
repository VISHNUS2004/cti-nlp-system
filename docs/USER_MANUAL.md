# CTI-NLP System User Manual

**Complete Guide to Working with the Cyber Threat Intelligence NLP System**

Welcome to the CTI-NLP system! This manual will guide you through everything you need to know to work with this repository, from initial setup to advanced usage.

## Table of Contents

- [Overview](#overview)
- [Quick Start (5 Minutes)](#quick-start-5-minutes)
- [Initial Setup](#initial-setup)
- [How to Run the System](#how-to-run-the-system)
- [Working with the API](#working-with-the-api)
- [Data Management](#data-management)
- [Development Workflow](#development-workflow)
- [Deployment Options](#deployment-options)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)

## Overview

The CTI-NLP system is a comprehensive cyber threat intelligence platform that:

- **Analyzes threat intelligence text** using machine learning
- **Classifies threats** into categories (Phishing, Malware, APT, etc.)
- **Predicts severity levels** (Low, Medium, High, Critical)
- **Extracts entities** (organizations, locations, threat actors)
- **Stores and manages** threat data in a database
- **Provides REST API** for integration
- **Ingests data** from multiple sources automatically

### Key Technologies

- **Backend**: FastAPI (Python)
- **ML Models**: BERT-NER, TF-IDF + Logistic Regression, Random Forest
- **Database**: PostgreSQL with Redis caching
- **Containerization**: Docker & Docker Compose
- **Testing**: pytest with comprehensive coverage

---

## Quick Start (5 Minutes)

Get the system running in test mode immediately:

### Step 1: Clone and Setup

```powershell
# Clone repository (if not already done)
git clone <your-repo-url>
cd cti-nlp-system

# Activate virtual environment
myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download language model
python -m spacy download en_core_web_sm
```

### Step 2: Quick Test Run

```powershell
# Set test mode (no database required)
$env:TEST_MODE="true"

# Test the system
python -c "
from backend.main import app
from fastapi.testclient import TestClient
client = TestClient(app)
response = client.post('/analyze', json={'text': 'APT29 phishing campaign targeting banks'})
print('Status Code:', response.status_code)
if response.status_code == 200:
    result = response.json()
    print('Threat Type:', result.get('threat_type'))
    print('Severity:', result.get('severity'))
    print('✅ Success: System is working!')
else:
    print('❌ Error:', response.text)
"
```

**Expected Output:**

```
Status Code: 200
Threat Type: Phishing
Severity: High
✅ Success: System is working!
```

### Step 3: Start the API Server

```powershell
# Start the FastAPI server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/docs for interactive API documentation

---

## Initial Setup

### Prerequisites

- **Python 3.9+** (recommended: 3.11)
- **Git** for version control
- **Docker** (optional, for full deployment)
- **PostgreSQL** (optional, for database mode)

### Environment Setup

#### 1. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv myenv

# Activate it
myenv\Scripts\activate

# Verify activation
python --version
pip --version
```

#### 2. Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# Download NLP model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import spacy; print('✅ spaCy installed')"
python -c "import sklearn; print('✅ scikit-learn installed')"
python -c "import fastapi; print('✅ FastAPI installed')"
```

#### 3. Prepare Machine Learning Models

```powershell
# Train the models (first time only)
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py

# Verify models exist
dir models\
# Should show: threat_classifier.pkl, tfidf_vectorizer.pkl, severity_model.pkl, etc.
```

#### 4. Environment Configuration

Create `.env` file in project root:

```env
# Basic Configuration
TEST_MODE=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Database Configuration (optional)
DATABASE_URL=postgresql://user:password@localhost:5432/cti_nlp
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Data Sources
TWITTER_BEARER_TOKEN=your-twitter-token
MITRE_API_URL=https://cti-taxii.mitre.org/stix/collections/
```

---

## How to Run the System

### Option 1: Test Mode (Easiest)

Perfect for development and testing without database:

```powershell
# Activate environment
myenv\Scripts\activate

# Set test mode
$env:TEST_MODE="true"

# Start server
uvicorn backend.main:app --reload
```

**Features in Test Mode:**

- ✅ All ML models work
- ✅ API endpoints available
- ✅ No database required
- ✅ Returns mock data for persistence
- ✅ Perfect for development

### Option 2: Full Mode with Database

For production-like environment:

```powershell
# Start database (Docker)
docker run -d --name cti-postgres \
  -e POSTGRES_DB=cti_nlp \
  -e POSTGRES_USER=cti_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  postgres:15-alpine

# Start Redis (Docker)
docker run -d --name cti-redis \
  -p 6379:6379 \
  redis:7-alpine

# Update .env file
echo "DATABASE_URL=postgresql://cti_user:secure_password@localhost:5432/cti_nlp" >> .env
echo "REDIS_URL=redis://localhost:6379/0" >> .env

# Initialize database
python -c "
from database.database import init_database
init_database()
print('✅ Database initialized')
"

# Start server
uvicorn backend.main:app --reload
```

### Option 3: Docker Compose (Recommended for Production)

```powershell
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

**Includes:**

- FastAPI application
- PostgreSQL database
- Redis cache
- Nginx reverse proxy
- Automatic health checks

---

## Working with the API

### Interactive Documentation

Visit http://localhost:8000/docs for Swagger UI interface where you can:

- See all available endpoints
- Test API calls interactively
- View request/response schemas
- Copy curl commands

### Core Endpoints

#### 1. Analyze Threat Text

```powershell
# Using curl
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "APT29 group launching sophisticated phishing attacks against government agencies"}'

# Using PowerShell
$body = @{
    text = "Ransomware attack encrypting hospital systems with .locked extension"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/analyze" -Method POST -Body $body -ContentType "application/json"
```

**Response:**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "original_text": "APT29 group launching sophisticated phishing attacks...",
  "entities": {
    "ORG": ["APT29"],
    "LOC": ["government agencies"]
  },
  "threat_type": "Phishing",
  "severity": "High",
  "confidence_scores": {
    "threat_classification": 0.85,
    "severity_prediction": 0.78
  },
  "timestamp": "2025-08-26T21:42:00.000Z"
}
```

#### 2. Health Check

```powershell
curl "http://localhost:8000/health"
```

#### 3. List Threats

```powershell
# Get recent threats
curl "http://localhost:8000/threats?limit=10"

# Filter by threat type
curl "http://localhost:8000/threats?threat_type=Phishing"

# Filter by severity
curl "http://localhost:8000/threats?severity=High&limit=5"
```

#### 4. Analytics

```powershell
# Get threat analytics
curl "http://localhost:8000/analytics?days=30"
```

#### 5. Bulk Upload

```powershell
# Upload CSV file
curl -X POST "http://localhost:8000/upload_csv" \
  -F "file=@threats.csv"
```

### Python Client Example

```python
import requests

class CTINLPClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def analyze_threat(self, text):
        """Analyze a single threat text"""
        response = requests.post(
            f"{self.base_url}/analyze",
            json={"text": text}
        )
        return response.json()

    def get_threats(self, limit=10, threat_type=None, severity=None):
        """Get list of threats with filters"""
        params = {"limit": limit}
        if threat_type:
            params["threat_type"] = threat_type
        if severity:
            params["severity"] = severity

        response = requests.get(f"{self.base_url}/threats", params=params)
        return response.json()

    def get_analytics(self, days=30):
        """Get threat analytics"""
        response = requests.get(f"{self.base_url}/analytics", params={"days": days})
        return response.json()

# Usage example
client = CTINLPClient()

# Analyze threat
result = client.analyze_threat("DDoS attack targeting financial infrastructure")
print(f"Threat: {result['threat_type']}, Severity: {result['severity']}")

# Get recent phishing threats
threats = client.get_threats(threat_type="Phishing", limit=5)
print(f"Found {threats['count']} phishing threats")

# Get analytics
analytics = client.get_analytics(days=7)
print(f"Total threats this week: {analytics['total_threats']}")
```

---

## Data Management

### Input Data Formats

#### 1. Single Text Analysis

```json
{
  "text": "Your threat intelligence text here"
}
```

#### 2. CSV Bulk Upload

CSV format with `text` column:

```csv
text
"APT group targeting critical infrastructure with zero-day exploits"
"Phishing campaign using fake banking websites to steal credentials"
"Ransomware encrypting files with .crypto extension demanding bitcoin"
"DDoS attacks against government websites during election period"
```

#### 3. JSON Bulk Upload

```json
[
  { "text": "Threat description 1" },
  { "text": "Threat description 2" },
  { "text": "Threat description 3" }
]
```

### Automated Data Ingestion

The system can automatically ingest data from multiple sources:

#### Configure Data Sources

```powershell
# Set up Twitter API credentials
echo "TWITTER_BEARER_TOKEN=your_token_here" >> .env

# Configure MITRE ATT&CK ingestion
echo "MITRE_API_URL=https://cti-taxii.mitre.org/stix/collections/" >> .env
```

#### Manual Ingestion

```powershell
# Ingest from all sources
python scripts/ingest_all_sources.py

# Ingest from specific source
python data_ingestion/fetch_twitter.py
python data_ingestion/fetch_mitre_attack.py
```

#### Real-time Ingestion

```powershell
# Start real-time ingestion service
python -c "
from data_ingestion.preprocess import start_real_time_ingestion
start_real_time_ingestion()
"
```

### Data Export

```powershell
# Export threats to CSV
curl "http://localhost:8000/export/csv?days=30" -o threats_export.csv

# Export specific threat types
curl "http://localhost:8000/export/csv?threat_type=Phishing" -o phishing_threats.csv
```

---

## Development Workflow

### Project Structure Overview

```
cti-nlp-system/
├── backend/                 # FastAPI application
│   ├── main.py             # Main API server
│   ├── classifier.py       # Threat classification
│   ├── severity_predictor.py # Severity prediction
│   └── threat_ner.py       # Named entity recognition
├── database/               # Database layer
│   ├── models.py           # SQLAlchemy models
│   ├── database.py         # Database configuration
│   └── services.py         # Business logic
├── data_ingestion/         # Data collection
│   ├── fetch_twitter.py    # Twitter API integration
│   ├── fetch_mitre_attack.py # MITRE ATT&CK data
│   └── preprocess.py       # Data preprocessing
├── models/                 # Trained ML models
├── tests/                  # Test suite
├── scripts/                # Training and utility scripts
├── docs/                   # Documentation
└── data/                   # Data files and cache
```

### Development Commands

#### Using Makefile (Recommended)

```powershell
# Setup development environment
make setup

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Start development server
make dev

# Build Docker image
make build

# Deploy with Docker
make deploy
```

#### Manual Commands

```powershell
# Code quality
black backend/ database/ data_ingestion/  # Format code
flake8 backend/ database/                 # Lint code
mypy backend/                            # Type checking

# Testing
pytest tests/ -v                         # Run tests
pytest tests/ --cov=backend --cov-report=html  # With coverage

# Development server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Git Workflow

```powershell
# Create feature branch
git checkout -b feature/new-model

# Make changes and commit
git add .
git commit -m "Add new threat classification model"

# Push and create PR
git push origin feature/new-model
```

### Model Training

#### Train New Models

```powershell
# Train threat classifier
python scripts/train_threat_classifier.py

# Train severity predictor
python scripts/train_severity_model.py

# Train NER model
python scripts/train_ner_model.py

# Train transformer-based model
python scripts/train_ner_transformer.py
```

#### Evaluate Models

```powershell
# Evaluate all models
python -c "
from backend.classifier import evaluate_model
from backend.severity_predictor import evaluate_severity_model
evaluate_model()
evaluate_severity_model()
"
```

---

## Deployment Options

### 1. Local Development

```powershell
# Activate environment
myenv\Scripts\activate

# Set test mode
$env:TEST_MODE="true"

# Start server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

**Use Case**: Development, testing, demonstrations

### 2. Docker Compose (Recommended)

```powershell
# Start all services
docker-compose up -d

# Monitor logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale app=3

# Update services
docker-compose pull
docker-compose up -d
```

**Use Case**: Production deployment, staging environment

### 3. Cloud Deployment

#### AWS ECS

```powershell
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker build -t cti-nlp .
docker tag cti-nlp:latest your-account.dkr.ecr.us-east-1.amazonaws.com/cti-nlp:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/cti-nlp:latest
```

#### Google Cloud Run

```powershell
# Deploy to Cloud Run
gcloud run deploy cti-nlp \
  --image gcr.io/your-project/cti-nlp \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances

```powershell
# Deploy to Azure
az container create \
  --resource-group myResourceGroup \
  --name cti-nlp-container \
  --image your-registry.azurecr.io/cti-nlp:latest \
  --cpu 2 --memory 4 \
  --ports 8000
```

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

**Error**: `FileNotFoundError: models/threat_classifier.pkl`

**Solution**:

```powershell
# Train missing models
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py

# Verify models exist
dir models\
```

#### 2. Environment Issues

**Error**: `ModuleNotFoundError: No module named 'backend'`

**Solution**:

```powershell
# Ensure virtual environment is activated
myenv\Scripts\activate

# Install in development mode
pip install -e .

# Or add to Python path
$env:PYTHONPATH="$PWD;$env:PYTHONPATH"
```

#### 3. Database Connection Issues

**Error**: `Connection refused on port 5432`

**Solution**:

```powershell
# Check if PostgreSQL is running
docker ps | findstr postgres

# Start PostgreSQL
docker run -d --name cti-postgres -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:15-alpine

# Test connection
python -c "
import psycopg2
try:
    conn = psycopg2.connect('postgresql://postgres:password@localhost:5432/postgres')
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
```

#### 4. API Server Issues

**Error**: `Address already in use`

**Solution**:

```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process
taskkill /PID <process_id> /F

# Or use different port
uvicorn backend.main:app --port 8001
```

#### 5. Docker Issues

**Error**: `docker-compose command not found`

**Solution**:

```powershell
# Install Docker Desktop or use docker compose (new syntax)
docker compose up -d

# Or install docker-compose separately
pip install docker-compose
```

### Debug Mode

```powershell
# Enable debug logging
$env:LOG_LEVEL="DEBUG"

# Run with detailed output
uvicorn backend.main:app --log-level debug

# Check logs
Get-Content logs\cti-nlp.log -Tail 50 -Wait
```

### Health Checks

```powershell
# API health
curl "http://localhost:8000/health"

# Database health
python -c "
from database.database import get_database_config
try:
    db = next(get_database_config().get_db())
    db.execute('SELECT 1')
    print('✅ Database healthy')
except Exception as e:
    print(f'❌ Database unhealthy: {e}')
"

# Redis health
python -c "
from database.database import get_redis_client
try:
    redis_client = get_redis_client()
    redis_client.ping()
    print('✅ Redis healthy')
except Exception as e:
    print(f'❌ Redis unhealthy: {e}')
"
```

---

## Advanced Usage

### Custom Model Training

#### 1. Prepare Your Data

```powershell
# Prepare custom dataset
python scripts/prepare_ner_data.py --input data/custom_threats.csv --output data/ner_prepared/

# Split data
python scripts/preprocess.py --split_ratio 0.8 0.1 0.1
```

#### 2. Train Custom Models

```python
# Custom threat classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load your training data
X_train, y_train = load_your_data()

# Train model
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_vec, y_train)

# Save model
pickle.dump(vectorizer, open('models/custom_vectorizer.pkl', 'wb'))
pickle.dump(classifier, open('models/custom_classifier.pkl', 'wb'))
```

#### 3. Update Backend to Use Custom Models

```python
# In backend/classifier.py
def load_custom_model():
    vectorizer = pickle.load(open('models/custom_vectorizer.pkl', 'rb'))
    classifier = pickle.load(open('models/custom_classifier.pkl', 'rb'))
    return vectorizer, classifier
```

### API Extensions

#### Add Custom Endpoints

```python
# In backend/main.py
@app.post("/custom_analysis")
async def custom_analysis(request: CustomAnalysisRequest):
    """Custom analysis endpoint"""
    # Your custom logic here
    return {"result": "custom analysis"}
```

#### Add Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

@app.post("/secure_analyze")
async def secure_analyze(
    request: ThreatAnalysisRequest,
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    # Protected endpoint
    return await analyze_threat_text(request)
```

### Performance Optimization

#### Enable Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_threat_analysis(text: str):
    """Cache results for repeated analyses"""
    return analyze_threat(text)
```

#### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def analyze_multiple_threats(texts: List[str]):
    """Process multiple threats concurrently"""
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=4)

    tasks = [
        loop.run_in_executor(executor, analyze_threat, text)
        for text in texts
    ]

    results = await asyncio.gather(*tasks)
    return results
```

### Monitoring and Logging

#### Custom Logging

```python
import logging
from pythonjsonlogger import jsonlogger

# Configure JSON logging
logger = logging.getLogger("cti_nlp")
handler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Use in your code
logger.info("Threat analyzed", extra={
    "threat_type": "Phishing",
    "severity": "High",
    "confidence": 0.85
})
```

#### Metrics Collection

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
threat_analysis_counter = Counter('threat_analyses_total', 'Total threat analyses')
analysis_duration = Histogram('analysis_duration_seconds', 'Analysis duration')

@analysis_duration.time()
def analyze_with_metrics(text: str):
    threat_analysis_counter.inc()
    return analyze_threat(text)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

## Quick Reference

### Essential Commands

```powershell
# Setup and activate environment
myenv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Test mode (no database)
$env:TEST_MODE="true"
uvicorn backend.main:app --reload

# Full mode (with database)
docker-compose up -d
uvicorn backend.main:app --reload

# Run tests
pytest tests/ -v

# Train models
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py

# Ingest data
python scripts/ingest_all_sources.py
```

### API Endpoints Summary

| Endpoint        | Method | Purpose                       |
| --------------- | ------ | ----------------------------- |
| `/analyze`      | POST   | Analyze single threat text    |
| `/upload_csv`   | POST   | Bulk upload CSV file          |
| `/threats`      | GET    | List stored threats           |
| `/threats/{id}` | GET    | Get specific threat           |
| `/analytics`    | GET    | Get threat analytics          |
| `/health`       | GET    | System health check           |
| `/feed`         | GET    | Latest ingested threats       |
| `/ingest_now`   | POST   | Trigger manual ingestion      |
| `/docs`         | GET    | Interactive API documentation |

### File Structure Reference

```
📁 cti-nlp-system/
├── 📁 backend/          → FastAPI application
├── 📁 database/         → Database models & services
├── 📁 data_ingestion/   → Data collection modules
├── 📁 models/           → Trained ML models
├── 📁 tests/            → Test suite
├── 📁 scripts/          → Training & utility scripts
├── 📁 docs/             → Documentation
├── 📁 data/             → Data files & cache
├── 📄 requirements.txt  → Python dependencies
├── 📄 docker-compose.yml → Multi-service deployment
├── 📄 Dockerfile        → Container configuration
└── 📄 Makefile          → Development automation
```

---

## Need Help?

- **Documentation**: Check `docs/` folder for detailed guides
- **API Reference**: Visit `/docs` endpoint when server is running
- **Testing**: See `docs/TESTING.md` for comprehensive testing guide
- **Deployment**: Check `docs/DEPLOYMENT.md` for deployment options
- **Issues**: Create GitHub issue with detailed description

---

**Happy Threat Hunting! 🔍🛡️**

_This manual covers the complete workflow for working with the CTI-NLP system. For specific technical details, refer to the individual documentation files in the `docs/` folder._
