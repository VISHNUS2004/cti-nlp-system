# CTI-NLP System AI/ML Architecture

This document explains the artificial intelligence and machine learning components of the CTI-NLP system.

---

## ML Pipeline Overview

The CTI-NLP system employs a multi-model approach to analyze cybersecurity threats:

```
Text Input → Preprocessing → Vectorization → ML Models → Analysis Engine → Output
```

### Pipeline Components:

1. **Text Preprocessing**: Clean and normalize input text
2. **Vectorization**: Convert text to numerical features
3. **Classification Models**: Predict threat types and severity
4. **Entity Extraction**: Identify cybersecurity entities
5. **Analysis Engine**: Combine results and generate insights

---

## Machine Learning Models

### 1. Threat Classification Model

**Purpose**: Classify threats into categories (Phishing, Malware, APT, etc.)

**Architecture**:
- **Algorithm**: Stochastic Gradient Descent (SGD)
- **Vectorization**: Count Vectorizer (max_features=5000)
- **Performance**: 26% accuracy, 0.214 F1-score
- **Training Time**: <0.01 seconds

**Why SGD?**:
- Best performance among 22 tested combinations
- Extremely fast training and prediction
- Handles high-dimensional sparse text data well
- Robust to overfitting on small datasets

### 2. Severity Prediction Model

**Purpose**: Predict threat severity (Low, Medium, High, Critical)

**Architecture**:
- **Algorithm**: Stochastic Gradient Descent (SGD)
- **Vectorization**: Count Vectorizer (max_features=5000)
- **Performance**: 40% accuracy, 0.289 F1-score
- **Training Time**: <0.01 seconds

**Severity Mapping**:
- Score 1-2: Low
- Score 3: Medium
- Score 4-5: High

### 3. Named Entity Recognition (NER)

**Purpose**: Extract cybersecurity-specific entities

**Architecture**:
- **Primary**: Regex-based pattern matching
- **Fallback**: BERT-based NER (dslim/bert-base-NER)
- **Performance**: 80%+ accuracy for cyber entities

**Extracted Entities**:
- IP addresses
- CVE references
- Hash values (MD5, SHA1, SHA256)
- URLs and domains
- Email addresses
- Threat keywords

---

## Model Selection Rationale

### Why Simple Models Over Complex Ensembles?

After comprehensive evaluation of 22 model combinations, simple models consistently outperformed complex ensembles:

**Simple Models (SGD + Count Vectorizer)**:
- 26% accuracy (threat classification)
- 40% accuracy (severity prediction)
- <0.01s training time
- Low computational requirements
- Robust generalization

**Complex Ensembles (Custom Features + Multiple Models)**:
- 13-45% performance drop
- 300x slower training
- Higher memory requirements
- Overfitting on small dataset

### Technical Reasons:

1. **Dataset Size**: 1,100 samples insufficient for complex models
2. **Text Sparsity**: High-dimensional sparse features favor linear models
3. **Class Balance**: Well-balanced data doesn't benefit from ensemble complexity
4. **Overfitting**: Simple models generalize better with limited data

---

## Feature Engineering

### Text Vectorization

**Count Vectorizer Configuration**:
```python
CountVectorizer(
    max_features=5000,
    lowercase=True,
    token_pattern=r'\b\w+\b',
    stop_words=None  # Domain-specific terms retained
)
```

**Why Count Vectorizer over TF-IDF?**:
- Better performance on this specific dataset
- Preserves importance of repeated threat terms
- Faster computation
- Less complex feature normalization

### Cybersecurity-Specific Features

**Regex Patterns for Entity Extraction**:
- **IP Address**: `\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b`
- **CVE**: `CVE-\d{4}-\d{4,}`
- **MD5**: `\b[a-fA-F0-9]{32}\b`
- **SHA256**: `\b[a-fA-F0-9]{64}\b`
- **Domain**: `\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b`

---

## Model Training Pipeline

### Training Process:

1. **Data Loading**: Load cybersecurity dataset (CSV format)
2. **Preprocessing**: Clean text, handle missing values
3. **Train/Test Split**: 80/20 split with stratification
4. **Vectorization**: Fit vectorizer on training data
5. **Model Training**: Train SGD classifier
6. **Evaluation**: Comprehensive metrics calculation
7. **Model Persistence**: Save models as .pkl files

### Training Scripts:

- `scripts/train_threat_classifier.py` - Basic threat classification
- `scripts/train_severity_model.py` - Severity prediction
- `scripts/train_improved_models.py` - Enhanced variants
- `scripts/comprehensive_model_evaluation.py` - Full evaluation

---

## Performance Optimization

### Speed Optimizations:

1. **Linear Algorithms**: O(n) complexity for prediction
2. **Sparse Matrices**: Memory-efficient text representation
3. **Optimized Libraries**: sklearn's highly optimized implementations
4. **Model Caching**: Pre-loaded models in memory

### Accuracy Optimizations:

1. **Cross-Validation**: 5-fold CV for robust evaluation
2. **Hyperparameter Selection**: Grid search for optimal parameters
3. **Statistical Validation**: Significance testing for model comparison
4. **Domain Adaptation**: Cybersecurity-specific preprocessing

---

## Deployment Architecture

### Production Pipeline:

```
FastAPI Endpoint → Model Loading → Preprocessing → Prediction → Post-processing → Response
```

### Model Loading Strategy:

```python
# Models loaded once at startup
classifier = joblib.load('models/threat_classifier.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
severity_model = joblib.load('models/severity_model.pkl')
```

### Error Handling:

- Fallback to basic classification if advanced models fail
- Graceful degradation for missing dependencies
- Comprehensive logging for debugging

---

## Future Enhancements

### Potential Improvements:

1. **Larger Datasets**: Collect more cybersecurity threat data
2. **Domain-Specific Models**: Fine-tune BERT on cybersecurity text
3. **Real-Time Learning**: Implement online learning capabilities
4. **Multi-Language Support**: Extend to non-English threats
5. **Advanced NER**: Custom cybersecurity entity recognition models

### Research Directions:

1. **Transfer Learning**: Leverage pre-trained security models
2. **Graph Neural Networks**: Model relationships between entities
3. **Attention Mechanisms**: Focus on important threat indicators
4. **Ensemble Methods**: With larger datasets, revisit ensemble approaches

---

## Model Monitoring

### Performance Metrics:

- **Accuracy**: Classification correctness
- **F1-Score**: Balanced precision/recall
- **Training Time**: Model efficiency
- **Prediction Speed**: Real-time performance
- **Memory Usage**: Resource consumption

### Monitoring Strategy:

1. **Automated Testing**: Regular model validation
2. **Performance Tracking**: Metric trending over time
3. **Data Drift Detection**: Monitor input data changes
4. **Model Retraining**: Periodic updates with new data

---

**This AI/ML architecture provides a robust, scalable foundation for cybersecurity threat analysis while maintaining simplicity and performance.**
