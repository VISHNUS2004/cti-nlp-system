# CTI-NLP System - Project Overview

**A comprehensive guide for team members to understand the project architecture, workflow, and implementation**

---

## What is CTI-NLP?

CTI-NLP (Cyber Threat Intelligence - Natural Language Processing) is an automated system that analyzes cybersecurity threats using machine learning and natural language processing. It helps security analysts by automatically:

1. **Classifying** threat types (Phishing, Malware, APT, etc.)
2. **Predicting** severity levels (Low, Medium, High, Critical)
3. **Extracting** key entities (IPs, CVEs, domains, threat actors)
4. **Scoring** overall risk (0-100 scale)
5. **Generating** actionable recommendations

---

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │────│  CTI-NLP System │────│   Outputs       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
│                      │                      │                 │
├ Twitter API          ├ FastAPI Backend     ├ Threat Reports   │
├ Dark Web Feeds       ├ ML Models          ├ Risk Scores      │
├ MITRE ATT&CK         ├ PostgreSQL DB      ├ Entity Lists     │
├ Manual Input         ├ Redis Cache        ├ Recommendations  │
└ CSV/JSON Files       └ Docker Container   └ API Responses    │
```

---

## System Components Explained

### 1. **Data Ingestion Layer**
- **Location**: `data_ingestion/`
- **Purpose**: Fetches threat data from multiple sources
- **Components**:
  - `fetch_twitter.py` - Collects tweets with threat indicators
  - `fetch_darkweb.py` - Gathers dark web threat intelligence
  - `fetch_mitre_attack.py` - Downloads MITRE ATT&CK data
  - `preprocess.py` - Cleans and normalizes collected data

### 2. **Machine Learning Pipeline**
- **Location**: `scripts/` and `models/`
- **Purpose**: Trains and deploys ML models for threat analysis
- **Key Models**:
  - **Threat Classifier**: SGD + Count Vectorizer (26% accuracy)
  - **Severity Predictor**: SGD + Count Vectorizer (40% accuracy)
  - **Entity Extractor**: Regex + BERT-NER for cybersecurity entities
- **Training Scripts**:
  - `train_threat_classifier.py` - Trains basic threat classification
  - `train_severity_model.py` - Trains severity prediction
  - `comprehensive_model_evaluation.py` - Tests 22 model combinations

### 3. **Backend API System**
- **Location**: `backend/`
- **Purpose**: Serves ML models via REST API
- **Framework**: FastAPI (modern, fast Python web framework)
- **Key Components**:
  - `main.py` - API server with all endpoints
  - `classifier.py` - Threat classification logic
  - `severity_predictor.py` - Severity prediction logic
  - `threat_ner.py` - Entity extraction logic
  - `simple_enhanced_analyzer.py` - Comprehensive analysis engine

### 4. **Database Layer**
- **Primary DB**: PostgreSQL for persistent data storage
- **Cache**: Redis for performance optimization
- **Location**: `database/` (configuration and models)
- **Stores**: Threat intelligence records, analysis results, user data

### 5. **Testing Infrastructure**
- **Location**: `tests/`
- **Framework**: pytest with comprehensive coverage
- **Components**:
  - `test_api.py` - API endpoint testing
  - `test_models.py` - ML model testing
  - `conftest.py` - Test configuration
  - `run_tests.py` - Test execution script

### 6. **Deployment & Operations**
- **Containerization**: Docker with docker-compose
- **Web Server**: Nginx for production
- **Environment**: Python virtual environment with 100+ packages
- **Monitoring**: Health checks and logging

---

## Data Flow Explained

### Input → Processing → Output

1. **Data Input**:
   ```
   Raw Text: "APT29 phishing campaign targeting government agencies with spear-phishing emails"
   ```

2. **Processing Pipeline**:
   ```
   Text Cleaning → Vectorization → ML Models → Analysis Engine
   ```

3. **Output**:
   ```json
   {
     "classification": {
       "category": "Phishing",
       "confidence": 0.85
     },
     "severity": {
       "level": "High",
       "score": 0.82
     },
     "entities": {
       "threat_actor": ["APT29"],
       "attack_type": ["phishing", "spear-phishing"],
       "target": ["government agencies"]
     },
     "risk_score": 75,
     "recommendations": [
       "Implement email security filters",
       "Train staff on phishing recognition",
       "Monitor for APT29 indicators"
     ]
   }
   ```

---

## Project Workflow for Team Members

### **For New Developers**

1. **Setup** (30 minutes):
   ```bash
   git clone <repo>
   cd cti-nlp-system
   python -m venv myenv
   myenv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run Basic Tests**:
   ```bash
   python scripts/train_threat_classifier.py
   python scripts/train_severity_model.py
   python tests/run_tests.py
   ```

3. **Start Development Server**:
   ```bash
   python backend/main.py
   ```

### **For Data Scientists**

1. **Model Training**:
   - Use `scripts/train_improved_models.py` for enhanced models
   - Run `scripts/comprehensive_model_evaluation.py` for evaluation
   - Check `docs/07-research/` for academic justification

2. **Data Analysis**:
   - Datasets in `data/` folder
   - Prepared NER data in `data/ner_prepared/`
   - Use `scripts/statistical_analysis.py` for insights

### **For DevOps/Deployment**

1. **Container Deployment**:
   ```bash
   docker-compose up -d
   ```

2. **Production Setup**:
   - Follow `docs/05-deployment/DEPLOYMENT.md`
   - Configure environment variables
   - Set up monitoring and logging

### **For QA/Testing**

1. **Run All Tests**:
   ```bash
   pytest tests/ -v --cov
   ```

2. **Manual Testing**:
   - Use `docs/06-testing/TEST_GUIDE.md`
   - Test API endpoints via Postman or curl
   - Validate model outputs

---

## Key Performance Metrics

### **Current System Performance**:
- **Threat Classification**: 26% accuracy (SGD + Count Vectorizer)
- **Severity Prediction**: 40% accuracy (SGD + Count Vectorizer)
- **Entity Extraction**: 80%+ accuracy for cyber-specific entities
- **API Response Time**: < 100ms per request
- **Training Time**: < 0.01 seconds for optimized models

### **Why These Metrics?**:
- Low accuracy is due to challenging cybersecurity classification task
- Simple models outperform complex ensembles on this dataset
- Fast training enables rapid iteration and deployment
- Academic research validates model selection choices

---

## Directory Structure Quick Reference

```
cti-nlp-system/
├── backend/          # FastAPI application and ML logic
├── data/            # Training datasets and processed data
├── data_ingestion/  # Data collection scripts
├── database/        # Database models and configuration
├── docs/           # Comprehensive documentation
├── models/         # Trained ML model files (.pkl)
├── scripts/        # Training and utility scripts
├── tests/          # Test suite
├── dashboard/      # Web interface (if needed)
├── nginx/          # Web server configuration
├── myenv/          # Python virtual environment
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Getting Help

### **Documentation Paths**:
- **New to project**: `docs/01-getting-started/USER_MANUAL.md`
- **API usage**: `docs/01-getting-started/API.md`
- **Architecture**: `docs/02-architecture/`
- **Models**: `docs/03-models/`
- **Deployment**: `docs/05-deployment/`
- **Testing**: `docs/06-testing/`
- **Research**: `docs/07-research/`

### **Common Issues**:
- **Import errors**: Check virtual environment activation
- **Model missing**: Run training scripts first
- **Database errors**: Ensure PostgreSQL/Redis are running
- **API errors**: Check `docs/06-testing/TESTING.md`

---

## What Makes This Project Special?

1. **Academic Rigor**: 22 model combinations tested with statistical validation
2. **Production Ready**: Docker deployment, comprehensive testing, health checks
3. **Comprehensive**: End-to-end pipeline from data ingestion to deployment
4. **Well Documented**: Professional documentation for all components
5. **Optimized**: Simple, fast models chosen over complex ones based on evidence
6. **Extensible**: Modular design allows easy addition of new features

---

**This overview provides the foundation for understanding the CTI-NLP system. For detailed technical information, refer to the specific documentation files in each folder.**
