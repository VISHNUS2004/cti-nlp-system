# CTI-NLP Team Quick Reference

**Essential commands and information for team members**

---

## Quick Setup Commands

### **Initial Setup**
```bash
# Clone and setup
git clone <repo-url>
cd cti-nlp-system

# Activate environment (Windows)
myenv\Scripts\activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Train models (if not exists)
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py
```

### **Daily Development**
```bash
# Activate environment
myenv\Scripts\activate

# Run tests
python tests/run_tests.py

# Start API server
python backend/main.py

# Access API docs
# http://localhost:8000/docs
```

---

## Key File Locations

### **🔧 Development**
- **API Server**: `backend/main.py`
- **Model Training**: `scripts/train_*.py`
- **Tests**: `tests/`
- **Requirements**: `requirements.txt`

### **📊 Data & Models**
- **Trained Models**: `models/` (*.pkl files)
- **Datasets**: `data/` (*.csv files)
- **Processed Data**: `data/ner_prepared/`

### **📚 Documentation**
- **Getting Started**: `docs/01-getting-started/USER_MANUAL.md`
- **API Reference**: `docs/01-getting-started/API.md`
- **Architecture**: `docs/02-architecture/`
- **Testing Guide**: `docs/06-testing/TEST_GUIDE.md`

### **🚀 Deployment**
- **Docker**: `docker-compose.yml`
- **Deployment Guide**: `docs/05-deployment/DEPLOYMENT.md`
- **Environment Config**: `.env.example`

---

## API Endpoints Quick Reference

### **Base URL**: `http://localhost:8000`

### **Main Endpoints**:
- `GET /health` - System health check
- `POST /analyze` - Analyze threat text
- `GET /docs` - Interactive API documentation
- `GET /dashboard` - Web interface

### **Test Command**:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "APT29 phishing campaign targeting government"}'
```

---

## Common Tasks

### **🧪 Testing**
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_api.py -v

# Test with coverage
pytest tests/ --cov
```

### **🤖 Model Training**
```bash
# Basic models
python scripts/train_threat_classifier.py
python scripts/train_severity_model.py

# Enhanced models
python scripts/train_improved_models.py

# Model evaluation
python scripts/comprehensive_model_evaluation.py
```

### **📊 Data Processing**
```bash
# Prepare NER data
python scripts/prepare_ner_data.py

# Ingest from all sources
python scripts/ingest_all_sources.py

# Data preprocessing
python scripts/preprocess.py
```

### **🚀 Deployment**
```bash
# Local Docker deployment
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

---

## Project Statistics

### **📈 Performance**
- **Threat Classification**: 26% accuracy
- **Severity Prediction**: 40% accuracy
- **API Response Time**: <100ms
- **Model Training**: <0.01s

### **📦 Codebase**
- **Python Files**: 50+ scripts
- **Documentation**: 20+ markdown files
- **Models**: 13 trained model files
- **Tests**: Comprehensive test suite
- **Dependencies**: 100+ Python packages

### **🏗️ Architecture**
- **Backend**: FastAPI + Python
- **Database**: PostgreSQL + Redis
- **ML Stack**: scikit-learn + transformers
- **Deployment**: Docker + nginx
- **Testing**: pytest + coverage

---

## Team Roles & Responsibilities

### **👨‍💻 Developers**
- **Focus**: Backend API, model integration
- **Key Files**: `backend/`, `scripts/`
- **Documentation**: `docs/02-architecture/`

### **🔬 Data Scientists**
- **Focus**: Model training, evaluation, research
- **Key Files**: `scripts/`, `models/`, `data/`
- **Documentation**: `docs/03-models/`, `docs/07-research/`

### **🧪 QA/Testers**
- **Focus**: Testing, validation, documentation
- **Key Files**: `tests/`, `docs/06-testing/`
- **Tools**: pytest, manual testing guides

### **🚀 DevOps**
- **Focus**: Deployment, infrastructure, monitoring
- **Key Files**: `docker-compose.yml`, `nginx/`
- **Documentation**: `docs/05-deployment/`

---

## Emergency Contacts & Resources

### **🆘 Common Issues**

**Problem**: Import errors
**Solution**: Check virtual environment activation

**Problem**: Models not found
**Solution**: Run training scripts first

**Problem**: Database connection failed
**Solution**: Start PostgreSQL/Redis containers

**Problem**: API not responding
**Solution**: Check `docs/06-testing/TESTING.md`

### **📖 Key Documentation**
1. **New Team Member**: Start with `docs/PROJECT_OVERVIEW.md`
2. **Setup Issues**: Check `docs/01-getting-started/USER_MANUAL.md`
3. **API Problems**: See `docs/01-getting-started/API.md`
4. **Model Questions**: Browse `docs/03-models/`
5. **Deployment Issues**: Read `docs/05-deployment/DEPLOYMENT.md`

### **🔗 External Resources**
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **scikit-learn Docs**: https://scikit-learn.org/
- **Docker Docs**: https://docs.docker.com/
- **PostgreSQL Docs**: https://www.postgresql.org/docs/

---

## Version Information

- **Project Version**: 1.0.0
- **Python Version**: 3.13.3
- **FastAPI Version**: 0.116.1
- **Docker Compose Version**: 3.8
- **Last Updated**: August 27, 2025

---

**💡 Tip**: Bookmark this file and the main `PROJECT_OVERVIEW.md` for quick reference!**
