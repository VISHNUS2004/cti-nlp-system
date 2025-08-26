![header](assets/header.png)

# AI-Powered Cyber Threat Intelligence System using NLP & FastAPI

A production-ready Cyber Threat Intelligence (CTI) system that leverages **Natural Language Processing (NLP)** and **AI-based classification** to extract meaningful cyber threat indicators from unstructured text, categorize threat types, predict severity levels, and provide insights through an interactive web interface with **comprehensive database integration**, **Docker deployment**, and **automated testing**.

## What's New in v2.0

- **Complete Docker Configuration** with multi-service orchestration
- **PostgreSQL Database Integration** with full CRUD operations
- **Redis Caching** for improved performance
- **Comprehensive Test Suite** with 70%+ coverage
- **Production Deployment** documentation and automation
- **Database Models** for threat intelligence, entities, and analytics
- **REST API** with OpenAPI documentation
- **Health Monitoring** and logging infrastructure
- **Security Hardening** with rate limiting and CORS
- **CI/CD Ready** with automated testing and deployment scripts

## Table of Contents

- [ Overview](#overview)
- [ Features](#features)
- [ Quick Start](#quick-start)
- [ Project Structure](#project-structure)
- [ Technology Stack](#technology-stack)
- [ Installation & Setup](#installation--setup)
- [ API Documentation](#api-documentation)
- [ Testing](#testing)
- [ Deployment](#deployment)
- [ Documentation](#documentation)
- [ Contributing](#contributing)
- [ License](#license)
- [ Maintainers](#maintainers)

## Overview

In today's cyber threat landscape, real-time intelligence is crucial. This platform uses **NLP-based entity recognition**, **machine learning-based threat classification**, and **severity prediction models** to generate actionable insights from threat reports and social media data.

The system is designed for analysts and SOC teams to **triage, investigate, and act** — all within one command-center styled dashboard with full database persistence and API-driven architecture.

## Features

### AI & Machine Learning

- **Named Entity Recognition (NER)** for extracting IOCs (IP addresses, malware names, CVEs, etc.)
- **Threat Classification** into categories like Phishing, Malware, APTs, Ransomware
- **Severity Level Prediction** using keyword extraction + ML models
- **Ensemble Models** for improved accuracy and robustness

### Data Management

- **PostgreSQL Database** with optimized schemas and indexes
- **Redis Caching** for high-performance data retrieval
- **Real-time Data Ingestion** from multiple sources (Twitter, Dark Web, MITRE ATT&CK)
- **Data Validation** and preprocessing pipelines

### API & Integration

- **FastAPI Backend** with automatic OpenAPI documentation
- **RESTful API** with comprehensive endpoints
- **Real-time Analysis** with immediate database persistence
- **Batch Processing** for CSV uploads and bulk analysis

### DevOps & Production

- **Docker Containerization** with multi-service orchestration
- **Production Deployment** with Nginx reverse proxy
- **Health Monitoring** and logging infrastructure
- **Automated Testing** with 70%+ code coverage
- **Security Features** including rate limiting and CORS

### Analytics & Reporting

- **Interactive Dashboard** with real-time updates
- **Threat Analytics** with statistical breakdowns
- **Historical Data** analysis and trending
- **Export Capabilities** for reports and raw data

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/sanjanb/cti-nlp-system.git
cd cti-nlp-system

# Quick setup
make quick-start

# Or manually:
python setup.py
docker-compose up -d

# Access the application
open http://localhost:8000/docs
```

### Local Development

```bash
# Setup Python environment
python -m venv myenv
myenv\Scripts\activate  # Windows
source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Train models (if needed)
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py

# Start the application
uvicorn backend.main:app --reload
```

## Project Structure

```text
cti-nlp-system/
├── backend/                  # FastAPI backend with ML models
│   ├── main.py              # Main API application
│   ├── threat_ner.py        # Named Entity Recognition
│   ├── classifier.py        # Threat classification
│   └── severity_predictor.py # Severity prediction
│
├── database/                 # Database layer
│   ├── models.py            # SQLAlchemy models
│   ├── database.py          # Database configuration
│   ├── services.py          # Database services/CRUD
│   └── init.sql             # Database initialization
│
├── data_ingestion/           # Data collection modules
│   ├── fetch_twitter.py     # Twitter API integration
│   ├── fetch_darkweb.py     # Dark web data collection
│   ├── fetch_mitre_attack.py # MITRE ATT&CK integration
│   └── preprocess.py        # Data preprocessing
│
├── tests/                    # Comprehensive test suite
│   ├── conftest.py          # Test configuration
│   ├── test_api.py          # API endpoint tests
│   ├── test_models.py       # ML model tests
│   └── run_tests.py         # Test runner
│
├── scripts/                  # Utility and training scripts
│   ├── train_threat_classifier.py # Train classification model
│   ├── train_severity_model.py    # Train severity model
│   └── ingest_all_sources.py      # Data ingestion orchestrator
│
├── docs/                     # Documentation
│   ├── DEPLOYMENT.md        # Deployment guide
│   ├── API.md              # API documentation
│   └── [model docs]        # Model-specific documentation
│
├── nginx/                   # Nginx configuration
├── models/                  # Trained ML models
├── data/                    # Raw and processed data
├── docker-compose.yml       # Multi-service orchestration
├── Dockerfile              # Application container
├── Makefile                # Development automation
└── requirements.txt        # Python dependencies
```

## Technology Stack

| Category       | Tools & Libraries                               |
| -------------- | ----------------------------------------------- |
| **Backend**    | FastAPI, Uvicorn, SQLAlchemy, Alembic           |
| **Database**   | PostgreSQL 15, Redis 7                          |
| **ML/NLP**     | spaCy, HuggingFace Transformers, scikit-learn   |
| **Models**     | BERT-NER, TF-IDF + Logistic Regression, XGBoost |
| **Testing**    | pytest, httpx, pytest-cov                       |
| **DevOps**     | Docker, Docker Compose, Nginx                   |
| **Monitoring** | Health checks, Prometheus metrics               |
| **Security**   | CORS, Rate limiting, Environment variables      |

## Installation & Setup

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- 4GB+ RAM
- 10GB+ storage

### Automated Setup

```bash
# Run the setup script
python setup.py

# Check system health
make health

# Run tests
make test
```

### Manual Setup

1. **Environment Configuration**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Database Setup**

   ```bash
   # Start database services
   docker-compose up -d postgres redis

   # Initialize database
   make db-init
   ```

3. **Model Training**

   ```bash
   # Train all models
   make train-models

   # Or individually:
   make train-classifier
   make train-severity
   ```

4. **Start Application**

   ```bash
   # Development mode
   make dev

   # Production mode
   make deploy-prod
   ```

## API Documentation

### Core Endpoints

- `POST /analyze` - Analyze threat intelligence text
- `GET /threats` - Retrieve threat records with filtering
- `GET /threats/{id}` - Get detailed threat information
- `GET /analytics` - Get threat statistics and analytics
- `POST /upload_csv` - Batch analysis from CSV files
- `GET /health` - System health check

### Example Usage

```python
import requests

# Analyze text
response = requests.post("http://localhost:8000/analyze",
    json={"text": "APT29 phishing campaign targeting banks"})

result = response.json()
print(f"Threat: {result['threat_type']}")
print(f"Severity: {result['severity']}")
print(f"Entities: {result['entities']}")
```

**Full API documentation:** [docs/API.md](docs/API.md)

**Interactive docs:** `http://localhost:8000/docs` (when running)

## Testing

### Run All Tests

```bash
# Complete test suite
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage report
pytest --cov=backend --cov=database --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Model Tests**: ML model validation
- **Database Tests**: CRUD operations testing

### Coverage Report

The test suite maintains 70%+ code coverage with detailed HTML reports generated in `htmlcov/`.

## Deployment

### Development Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale application
docker-compose up -d --scale app=3
```

### Production Deployment

```bash
# Production deployment
make deploy-prod

# With custom environment
docker-compose -f docker-compose.prod.yml up -d
```

### Cloud Deployment

Supports deployment on:

- **AWS** (EC2, RDS, ElastiCache)
- **Google Cloud** (Cloud Run, Cloud SQL)
- **Azure** (Container Instances, PostgreSQL)

**Detailed deployment guide:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Monitoring & Maintenance

### Health Monitoring

```bash
# Check application health
curl http://localhost:8000/health

# View service status
docker-compose ps

# Monitor logs
make logs-app
```

### Performance Monitoring

- **Database Performance**: Query optimization and indexing
- **Redis Caching**: Cache hit rates and performance
- **API Response Times**: Request/response monitoring
- **Resource Usage**: CPU, memory, and storage monitoring

### Backup & Recovery

```bash
# Create full backup
make backup-full

# Database backup only
make db-backup

# Restore from backup
make restore
```

## Documentation

| Document                                                         | Description                     |
| ---------------------------------------------------------------- | ------------------------------- |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)                       | Complete deployment guide       |
| [`docs/API.md`](docs/API.md)                                     | Comprehensive API documentation |
| [`docs/1. SCRIPTS_OVERVIEW.md`](docs/1.%20SCRIPTS_OVERVIEW.md)   | Training scripts documentation  |
| [`docs/2. THREAT_CLASSIFIER.md`](docs/2.%20THREAT_CLASSIFIER.md) | Threat classification model     |
| [`docs/3. SEVERITY_MODEL.md`](docs/3.%20SEVERITY_MODEL.md)       | Severity prediction model       |
| [`docs/4. NER_MODEL.md`](docs/4.%20NER_MODEL.md)                 | Named Entity Recognition model  |
| [`docs/5. BACKEND_OVERVIEW.md`](docs/5.%20BACKEND_OVERVIEW.md)   | Backend architecture overview   |
| [`CONTRIBUTING.md`](CONTRIBUTING.md)                             | Contribution guidelines         |

## Performance Benchmarks

| Metric                | Performance               |
| --------------------- | ------------------------- |
| **API Response Time** | < 200ms (95th percentile) |
| **Throughput**        | 100+ requests/second      |
| **Model Inference**   | < 50ms per text sample    |
| **Database Queries**  | < 10ms (indexed queries)  |
| **Memory Usage**      | < 2GB (production)        |

## Contributing

We welcome contributions from students, researchers, and cybersecurity enthusiasts!

### Quick Contribution Guide

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   make test
   make lint
   ```
5. **Submit a pull request**

### Development Workflow

```bash
# Setup development environment
make setup
make install

# Make changes
# ...

# Test your changes
make test
make lint
make format

# Submit PR
git push origin feature/amazing-feature
```

**Detailed guidelines:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

## Maintainers

Developed as part of the final year project (CSE - AI & ML) at **ATME College of Engineering**, Mysuru:

- **Kushal S M** - Lead Developer
- **Sanjan B M** - ML Engineer & DevOps
- **Ponnanna K V** - Backend Developer
- **Vishnu S** - Data Engineer
- _Guided by Prof. Khateeja Ambreen_

### Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/sanjanb/cti-nlp-system/issues)
- **LinkedIn**: [Sanjan B M](https://www.linkedin.com/in/sanjanb/)
- **Email**: team@cti-nlp.com

---

## Star the Repository

If you find this project useful, please give it a star! ⭐

[![GitHub stars](https://img.shields.io/github/stars/sanjanb/cti-nlp-system.svg?style=social)](https://github.com/sanjanb/cti-nlp-system/stargazers)

---

_Last updated: August 2025 | v2.0.0_

## Table of Contents

- [ Overview](#overview)
- [ Features](#features)
- [ Project Structure](#project-structure)
- [ Technology Stack](#technology-stack)
- [ Getting Started](#getting-started)
- [ Testing the Application](#testing-the-application)
- [ Documentation](#documentation)
- [ Contributing](#contributing)
- [ License](#license)
- [ Maintainers](#maintainers)

## Overview

In today’s cyber threat landscape, real-time intelligence is crucial. This platform uses **NLP-based entity recognition**, **machine learning-based threat classification**, and **severity prediction models** to generate actionable insights from threat reports and forum data.

The system is designed for analysts and SOC teams to **triage, investigate, and act** — all within one command-center styled dashboard.

## Features

- **Named Entity Recognition (NER)** for extracting IOCs (IP addresses, malware names, CVEs, etc.)
- **Threat Classification** into categories like Phishing, Malware, APTs, Ransomware
- **Severity Level Prediction** using keyword extraction + ML models
- **Interactive Frontend Dashboard** with expandable result cards, live analysis, and downloadable reports
- Modular design with FastAPI, enabling scalability and API-first development
- **Docker-based Deployment** ready for cloud or local setups

## Project Structure

```text
cti-nlp-system/
│
├── backend/                  # FastAPI backend logic (main.py, NER, ML models)
│   ├── main.py
│   ├── threat_ner.py
│   ├── classifier.py
│   ├── severity_predictor.py
│   └── ...
│
├── dashboard/                # Frontend templates (HTML + CSS + Jinja2/JS)
│   ├── templates/
│   ├── static/
│   └── ...
│
├── data/                     # Raw and processed threat intel datasets
│   ├── raw/
│   ├── cleaned/
│   └── ...
│
├── docs/                     #  Documentation and testing guidelines
│   ├── setup_guide.md
│   ├── testing_guide.md
│   └── api_schema.json
│
├── models/                   # Saved ML models, vectorizers (joblib/pkl files)
│
├── scripts/                  # Preprocessing, training scripts for models
│   ├── train_threat_classifier.py
│   ├── preprocess.py
│   └── ...
│
├── utils/                    # Helper utilities (tokenizer, logger, metrics)
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
└── CONTRIBUTING.md
```

## Technology Stack

| Category     | Tools & Libraries                                   |
| ------------ | --------------------------------------------------- |
| Backend      | **FastAPI**, Uvicorn                                |
| NLP Models   | spaCy, HuggingFace Transformers (BERT, ThreatBERT)  |
| ML Libraries | Scikit-learn, XGBoost, PyTorch                      |
| Frontend     | HTML, Bootstrap 5, JavaScript, Jinja2               |
| Dashboard    | Custom Flask/Static Pages (migrating to React/Vite) |
| Deployment   | Docker, Render, Railway                             |
| Storage      | CSV, JSON                                           |

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sanjanb/cti-nlp-system.git
cd cti-nlp-system
```

### 2. Set Up Environment

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Start the FastAPI Server

```bash
uvicorn backend.main:app --reload
```

Access the API at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Testing the Application

Check the [ docs/testing_guide.md](docs/testing_guide.md) file for full testing procedures.

### Example API Usage

```bash
curl -X POST http://localhost:8000/analyze \
    -H "Content-Type: application/json" \
    -d '{"text": "QakBot malware exploited CVE-2023-1234 via phishing in Russia"}'
```

Expected JSON:

```json
{
  "original_text": "...",
  "entities": [...],
  "threat_type": "Phishing",
  "severity": "High"
}
```

You can also upload `.csv` files with a `text` column at `/upload_csv`.

## Documentation

| File                                             | Description                               |
| ------------------------------------------------ | ----------------------------------------- |
| [`docs/setup_guide.md`](docs/setup_guide.md)     | End-to-end setup and deployment steps     |
| [`docs/testing_guide.md`](docs/testing_guide.md) | Manual and automated testing instructions |
| [`docs/api_schema.json`](docs/api_schema.json)   | Swagger/OpenAPI schema                    |
| [`CONTRIBUTING.md`](CONTRIBUTING.md)             | Contribution guidelines                   |

## Contributing

We welcome contributions from students, researchers, and cybersecurity enthusiasts.

> For setup, conventions, and pull request flow, read [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE) for more details.

## Maintainers

Developed as part of the final year project (CSE - AI & ML) at **ATME College of Engineering**, Mysuru:

- **Kushal S M**
- **Sanjan B M**
- **Ponnanna K V**
- **Vishnu S**
- _Guided by Prof. Khateeja Ambreen_

For questions or suggestions, open a GitHub issue or reach out on [LinkedIn](https://www.linkedin.com/in/sanjanb/).
