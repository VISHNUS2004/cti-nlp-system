"""
Enhanced Backend Integration with Advanced Models
Integrates all improved models for better threat intelligence analysis
"""
import os
import sys
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from transformers import pipeline

# Add scripts directory to path for importing custom classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from train_advanced_classifier import AdvancedThreatClassifier
    from enhanced_cybersecurity_ner import CybersecurityNER
    from train_advanced_severity_predictor import AdvancedSeverityPredictor
except ImportError as e:
    print(f"Warning: Could not import advanced models: {e}")
    print("Falling back to basic models...")

class EnhancedThreatAnalyzer:
    """
    Enhanced threat analyzer that combines multiple advanced models
    for comprehensive threat intelligence analysis
    """
    
    def __init__(self, model_dir="models", use_advanced=True):
        self.model_dir = model_dir
        self.use_advanced = use_advanced
        
        # Model instances
        self.threat_classifier = None
        self.severity_predictor = None
        self.ner_analyzer = None
        
        # Fallback models (original simple models)
        self.simple_classifier = None
        self.simple_severity = None
        self.simple_ner = None
        
        self.load_models()
    
    def load_models(self):
        """Load all available models (advanced first, fallback to simple)"""
        print("Loading threat analysis models...")
        
        if self.use_advanced:
            self._load_advanced_models()
        
        # Always load simple models as fallback
        self._load_simple_models()
        
        print("Threat analysis models loaded successfully")
    
    def _load_advanced_models(self):
        """Load advanced models if available"""
        try:
            # Advanced Threat Classifier
            self.threat_classifier = AdvancedThreatClassifier()
            if not self.threat_classifier.load_models(self.model_dir):
                print("Advanced classifier not available, will use simple classifier")
                self.threat_classifier = None
            
            # Advanced Severity Predictor
            self.severity_predictor = AdvancedSeverityPredictor()
            if not self.severity_predictor.load_models(self.model_dir):
                print("Advanced severity predictor not available, will use simple predictor")
                self.severity_predictor = None
            
            # Enhanced NER
            self.ner_analyzer = CybersecurityNER()
            self.ner_analyzer.setup_models()
            
        except Exception as e:
            print(f"Failed to load advanced models: {e}")
            self.threat_classifier = None
            self.severity_predictor = None
            self.ner_analyzer = None
    
    def _load_simple_models(self):
        """Load simple fallback models"""
        try:
            # Simple TF-IDF classifier
            vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.pkl")
            classifier_path = os.path.join(self.model_dir, "threat_classifier.pkl")
            
            if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
                self.simple_tfidf = joblib.load(vectorizer_path)
                self.simple_classifier = joblib.load(classifier_path)
            
            # Simple severity model
            severity_model_path = os.path.join(self.model_dir, "severity_model.pkl")
            severity_vectorizer_path = os.path.join(self.model_dir, "severity_vectorizer.pkl")
            
            if os.path.exists(severity_model_path) and os.path.exists(severity_vectorizer_path):
                self.simple_severity_vectorizer = joblib.load(severity_vectorizer_path)
                self.simple_severity = joblib.load(severity_model_path)
            
            # Simple NER
            self.simple_ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
            
        except Exception as e:
            print(f"Warning: Failed to load some simple models: {e}")
    
    def classify_threat(self, text: str) -> Dict:
        """Classify threat using best available model"""
        try:
            # Try advanced classifier first
            if self.threat_classifier:
                result = self.threat_classifier.predict_with_ensemble(text)
                return {
                    'prediction': result['final_prediction'],
                    'confidence': result['confidence'],
                    'model_type': 'advanced_ensemble',
                    'individual_predictions': result['individual_predictions'],
                    'success': True
                }
            
            # Fallback to simple classifier
            elif self.simple_classifier and self.simple_tfidf:
                X = self.simple_tfidf.transform([text])
                prediction = self.simple_classifier.predict(X)[0]
                confidence = max(self.simple_classifier.predict_proba(X)[0])
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'model_type': 'simple_tfidf',
                    'success': True
                }
            
            else:
                return {
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'model_type': 'none',
                    'error': 'No classifier available',
                    'success': False
                }
                
        except Exception as e:
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'model_type': 'error',
                'error': str(e),
                'success': False
            }
    
    def predict_severity(self, text: str) -> Dict:
        """Predict severity using best available model"""
        try:
            # Try advanced severity predictor first
            if self.severity_predictor:
                analysis = self.severity_predictor.get_detailed_severity_analysis(text)
                
                return {
                    'severity': analysis['predicted_severity'],
                    'confidence_scores': analysis['confidence_scores'],
                    'model_type': 'advanced_ensemble',
                    'risk_indicators': analysis['risk_indicators'],
                    'recommendations': analysis['recommendations'],
                    'feature_analysis': analysis['feature_analysis'],
                    'success': True
                }
            
            # Fallback to simple severity model
            elif self.simple_severity and self.simple_severity_vectorizer:
                X = self.simple_severity_vectorizer.transform([text])
                prediction = self.simple_severity.predict(X)[0]
                confidence = max(self.simple_severity.predict_proba(X)[0])
                
                return {
                    'severity': prediction,
                    'confidence': confidence,
                    'model_type': 'simple_random_forest',
                    'success': True
                }
            
            else:
                return {
                    'severity': 'unknown',
                    'confidence': 0.0,
                    'model_type': 'none',
                    'error': 'No severity predictor available',
                    'success': False
                }
                
        except Exception as e:
            return {
                'severity': 'error',
                'confidence': 0.0,
                'model_type': 'error',
                'error': str(e),
                'success': False
            }
    
    def extract_entities(self, text: str) -> Dict:
        """Extract entities using best available model"""
        try:
            # Try enhanced NER first
            if self.ner_analyzer:
                analysis = self.ner_analyzer.get_comprehensive_analysis(text)
                
                return {
                    'entities': analysis['entities'],
                    'risk_score': analysis['risk_score'],
                    'threat_type': analysis['threat_type'],
                    'recommendations': analysis['recommendations'],
                    'model_type': 'enhanced_cybersecurity_ner',
                    'success': True
                }
            
            # Fallback to simple NER
            elif self.simple_ner:
                entities = self.simple_ner(text)
                grouped = {}
                
                for ent in entities:
                    label = ent["entity_group"]
                    word = ent["word"]
                    
                    if label not in grouped:
                        grouped[label] = []
                    if word not in grouped[label]:
                        grouped[label].append(word)
                
                return {
                    'entities': {'general_entities': grouped},
                    'model_type': 'simple_bert_ner',
                    'success': True
                }
            
            else:
                return {
                    'entities': {},
                    'model_type': 'none',
                    'error': 'No NER model available',
                    'success': False
                }
                
        except Exception as e:
            return {
                'entities': {},
                'model_type': 'error',
                'error': str(e),
                'success': False
            }
    
    def comprehensive_analysis(self, text: str) -> Dict:
        """Perform comprehensive threat analysis using all models"""
        analysis_start_time = np.datetime64('now')
        
        # Get individual analyses
        classification = self.classify_threat(text)
        severity = self.predict_severity(text)
        entities = self.extract_entities(text)
        
        # Combine results
        comprehensive_result = {
            'text': text,
            'timestamp': str(analysis_start_time),
            'classification': classification,
            'severity': severity,
            'entities': entities,
            'overall_risk_score': 0,
            'summary': {},
            'unified_recommendations': []
        }
        
        # Calculate overall risk score
        risk_components = []
        
        # Classification confidence
        if classification['success']:
            if classification['prediction'] in ['malware', 'attack', 'vulnerability']:
                risk_components.append(classification.get('confidence', 0) * 30)
        
        # Severity score
        if severity['success']:
            severity_weights = {
                'High': 40, 'Medium-High': 30, 'Medium': 20, 
                'Medium-Low': 10, 'Low': 5
            }
            severity_score = severity_weights.get(severity['severity'], 0)
            risk_components.append(severity_score)
        
        # Entity-based risk
        if entities['success']:
            entity_risk = entities.get('risk_score', 0)
            if entity_risk > 0:
                risk_components.append(min(entity_risk, 30))  # Cap at 30
        
        # Calculate overall risk (0-100)
        if risk_components:
            comprehensive_result['overall_risk_score'] = min(100, sum(risk_components))
        
        # Create unified summary
        summary = {
            'threat_detected': classification.get('prediction', 'unknown') != 'unknown',
            'severity_level': severity.get('severity', 'unknown'),
            'entities_found': bool(entities.get('entities', {})),
            'models_used': {
                'classification': classification.get('model_type', 'none'),
                'severity': severity.get('model_type', 'none'),
                'ner': entities.get('model_type', 'none')
            }
        }
        
        comprehensive_result['summary'] = summary
        
        # Unified recommendations
        recommendations = set()
        
        if classification.get('success') and 'recommendations' in classification:
            recommendations.update(classification['recommendations'])
        
        if severity.get('success') and 'recommendations' in severity:
            recommendations.update(severity['recommendations'])
        
        if entities.get('success') and 'recommendations' in entities:
            recommendations.update(entities['recommendations'])
        
        # Add risk-based recommendations
        overall_risk = comprehensive_result['overall_risk_score']
        if overall_risk >= 70:
            recommendations.add("CRITICAL: Immediate security response required")
            recommendations.add("Escalate to security incident response team")
        elif overall_risk >= 40:
            recommendations.add("HIGH: Priority security assessment needed")
            recommendations.add("Monitor for related indicators")
        elif overall_risk >= 20:
            recommendations.add("MEDIUM: Include in security review")
        else:
            recommendations.add("LOW: Log for future reference")
        
        comprehensive_result['unified_recommendations'] = list(recommendations)
        
        return comprehensive_result
    
    def get_model_status(self) -> Dict:
        """Get status of all loaded models"""
        return {
            'advanced_classifier': self.threat_classifier is not None,
            'advanced_severity': self.severity_predictor is not None,
            'enhanced_ner': self.ner_analyzer is not None,
            'simple_classifier': self.simple_classifier is not None,
            'simple_severity': self.simple_severity is not None,
            'simple_ner': self.simple_ner is not None,
            'model_directory': self.model_dir,
            'using_advanced': self.use_advanced
        }

# Global instance for API usage
threat_analyzer = None

def initialize_analyzer(model_dir="models", use_advanced=True):
    """Initialize the global threat analyzer"""
    global threat_analyzer
    threat_analyzer = EnhancedThreatAnalyzer(model_dir, use_advanced)
    return threat_analyzer

def get_analyzer():
    """Get the global threat analyzer instance"""
    global threat_analyzer
    if threat_analyzer is None:
        threat_analyzer = initialize_analyzer()
    return threat_analyzer

# Backwards compatible functions for existing API
def classify_threat(text: str) -> str:
    """Backwards compatible classification function"""
    analyzer = get_analyzer()
    result = analyzer.classify_threat(text)
    return result.get('prediction', 'unknown')

def predict_severity(text: str) -> str:
    """Backwards compatible severity prediction function"""
    analyzer = get_analyzer()
    result = analyzer.predict_severity(text)
    return result.get('severity', 'unknown')

def extract_threat_entities(text: str) -> Dict:
    """Backwards compatible entity extraction function"""
    analyzer = get_analyzer()
    result = analyzer.extract_entities(text)
    entities = result.get('entities', {})
    
    # Return in the format expected by existing code
    if 'general_entities' in entities:
        return entities['general_entities']
    else:
        return entities

# New enhanced functions
def analyze_threat_comprehensive(text: str) -> Dict:
    """Comprehensive threat analysis using all models"""
    analyzer = get_analyzer()
    return analyzer.comprehensive_analysis(text)

def get_model_info() -> Dict:
    """Get information about loaded models"""
    analyzer = get_analyzer()
    return analyzer.get_model_status()

if __name__ == "__main__":
    # Test the enhanced analyzer
    print("Testing Enhanced Threat Analyzer...")
    
    test_text = """
    URGENT: Ransomware attack detected on critical infrastructure. 
    Malicious IP 192.168.1.100 exploiting CVE-2021-34527 vulnerability.
    File hash: d41d8cd98f00b204e9800998ecf8427e connected to command server malicious-domain.com.
    Immediate containment required to prevent lateral movement!
    """
    
    analyzer = initialize_analyzer()
    result = analyzer.comprehensive_analysis(test_text)
    
    print("="*60)
    print("COMPREHENSIVE THREAT ANALYSIS")
    print("="*60)
    print(f"Overall Risk Score: {result['overall_risk_score']}/100")
    print(f"Classification: {result['classification']['prediction']}")
    print(f"Severity: {result['severity']['severity']}")
    print(f"Entities Found: {bool(result['entities']['entities'])}")
    print(f"\nRecommendations:")
    for rec in result['unified_recommendations']:
        print(f"  - {rec}")
    
    print(f"\nModel Status:")
    status = analyzer.get_model_status()
    for model, loaded in status.items():
        print(f"  {model}: {'✅' if loaded else '❌'}")
