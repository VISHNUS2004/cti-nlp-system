#!/usr/bin/env python3
"""
Phase 1 Model Enhancement Script
=================================

This script implements advanced model improvements focused on:
1. Ensemble Classification Methods
2. Advanced Feature Engineering
3. Cybersecurity-Specific Models
4. Comprehensive Model Evaluation

Author: CTI-NLP Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Feature Engineering
import re
from collections import Counter

warnings.filterwarnings('ignore')

class CyberThreatFeatureExtractor:
    """Extract cybersecurity-specific features from threat descriptions"""
    
    def __init__(self):
        # Cybersecurity keywords by category
        self.malware_keywords = [
            'trojan', 'virus', 'worm', 'backdoor', 'rootkit', 'spyware', 
            'adware', 'ransomware', 'botnet', 'keylogger', 'rat', 'apt'
        ]
        
        self.phishing_keywords = [
            'phishing', 'spear', 'spoofing', 'social engineering', 'credential', 
            'fake', 'impersonation', 'fraudulent', 'deceptive'
        ]
        
        self.network_keywords = [
            'ddos', 'dos', 'mitm', 'man-in-the-middle', 'injection', 'xss',
            'sql injection', 'buffer overflow', 'vulnerability', 'exploit'
        ]
        
        self.severity_indicators = {
            'critical': ['critical', 'severe', 'massive', 'devastating', 'widespread'],
            'high': ['high', 'significant', 'major', 'serious', 'dangerous'],
            'medium': ['medium', 'moderate', 'notable', 'considerable'],
            'low': ['low', 'minor', 'minimal', 'limited', 'small']
        }
        
        # IOC patterns
        self.ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        self.domain_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z]{2,})\b'
        self.hash_pattern = r'\b[a-fA-F0-9]{32,64}\b'
        self.cve_pattern = r'CVE-\d{4}-\d{4,7}'
        
    def extract_features(self, text):
        """Extract comprehensive features from threat description"""
        text_lower = text.lower()
        features = {}
        
        # 1. Keyword-based features
        features['malware_count'] = sum(1 for keyword in self.malware_keywords if keyword in text_lower)
        features['phishing_count'] = sum(1 for keyword in self.phishing_keywords if keyword in text_lower)
        features['network_count'] = sum(1 for keyword in self.network_keywords if keyword in text_lower)
        
        # 2. Severity indicators
        for severity, keywords in self.severity_indicators.items():
            features[f'{severity}_indicators'] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # 3. IOC counts
        features['ip_count'] = len(re.findall(self.ip_pattern, text))
        features['domain_count'] = len(re.findall(self.domain_pattern, text))
        features['hash_count'] = len(re.findall(self.hash_pattern, text))
        features['cve_count'] = len(re.findall(self.cve_pattern, text))
        
        # 4. Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # 5. Capital letters ratio (might indicate urgency)
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        # 6. Threat actor indicators
        threat_actors = ['apt', 'group', 'gang', 'actor', 'operation', 'campaign']
        features['threat_actor_mentions'] = sum(1 for actor in threat_actors if actor in text_lower)
        
        return features
    
    def extract_features_batch(self, texts):
        """Extract features for multiple texts"""
        feature_list = []
        for text in texts:
            features = self.extract_features(text)
            feature_list.append(features)
        return pd.DataFrame(feature_list)

class AdvancedThreatClassifier:
    """Advanced threat classification with ensemble methods"""
    
    def __init__(self):
        self.feature_extractor = CyberThreatFeatureExtractor()
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        self.ensemble_model = None
        self.label_encoder = LabelEncoder()
        
    def prepare_features(self, texts, fit_vectorizers=False):
        """Prepare comprehensive features combining TF-IDF, Count, and custom features"""
        
        # 1. TF-IDF features
        if fit_vectorizers:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # Include bigrams
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        # 2. Count vectorizer features
        if fit_vectorizers:
            self.count_vectorizer = CountVectorizer(
                max_features=2000,
                ngram_range=(1, 1),
                stop_words='english',
                binary=True
            )
            count_features = self.count_vectorizer.fit_transform(texts)
        else:
            count_features = self.count_vectorizer.transform(texts)
        
        # 3. Custom cybersecurity features
        custom_features = self.feature_extractor.extract_features_batch(texts)
        
        # Combine all features
        import scipy.sparse
        combined_features = scipy.sparse.hstack([
            tfidf_features,
            count_features,
            scipy.sparse.csr_matrix(custom_features.values)
        ])
        
        return combined_features, custom_features.columns.tolist()
    
    def create_ensemble_model(self):
        """Create an ensemble of different classifiers"""
        
        # Base classifiers
        lr_classifier = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        )
        
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
        
        svm_classifier = SVC(
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        classifiers = [
            ('logistic', lr_classifier),
            ('random_forest', rf_classifier),
            ('svm', svm_classifier)
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_classifier = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            classifiers.append(('xgboost', xgb_classifier))
        
        # Create voting ensemble
        self.ensemble_model = VotingClassifier(
            estimators=classifiers,
            voting='soft'  # Use predicted probabilities
        )
        
        return self.ensemble_model
    
    def train(self, texts, labels):
        """Train the ensemble model"""
        print("🔄 Preparing features...")
        X, feature_names = self.prepare_features(texts, fit_vectorizers=True)
        
        print("🔄 Encoding labels...")
        y = self.label_encoder.fit_transform(labels)
        
        print("🔄 Creating ensemble model...")
        self.create_ensemble_model()
        
        print("🔄 Training ensemble model...")
        self.ensemble_model.fit(X, y)
        
        return self.ensemble_model
    
    def predict(self, texts):
        """Predict using the trained ensemble model"""
        X, _ = self.prepare_features(texts, fit_vectorizers=False)
        predictions = self.ensemble_model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, texts):
        """Get prediction probabilities"""
        X, _ = self.prepare_features(texts, fit_vectorizers=False)
        return self.ensemble_model.predict_proba(X)
    
    def evaluate(self, texts, true_labels):
        """Comprehensive evaluation of the model"""
        predictions = self.predict(texts)
        
        print("\n📊 Classification Report:")
        print("=" * 50)
        print(classification_report(true_labels, predictions))
        
        print(f"\n🎯 Overall Accuracy: {accuracy_score(true_labels, predictions):.4f}")
        
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'predictions': predictions,
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        }
    
    def save_models(self, model_dir="models/enhanced"):
        """Save all trained components"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.ensemble_model, f"{model_dir}/ensemble_threat_classifier.pkl")
        joblib.dump(self.tfidf_vectorizer, f"{model_dir}/enhanced_tfidf_vectorizer.pkl")
        joblib.dump(self.count_vectorizer, f"{model_dir}/enhanced_count_vectorizer.pkl")
        joblib.dump(self.label_encoder, f"{model_dir}/enhanced_label_encoder.pkl")
        joblib.dump(self.feature_extractor, f"{model_dir}/feature_extractor.pkl")
        
        print(f"✅ Enhanced models saved to {model_dir}/")

class AdvancedSeverityPredictor:
    """Enhanced severity prediction with feature engineering"""
    
    def __init__(self):
        self.feature_extractor = CyberThreatFeatureExtractor()
        self.vectorizer = None
        self.model = None
        
    def prepare_features(self, texts, fit_vectorizer=False):
        """Prepare features for severity prediction"""
        
        # TF-IDF features
        if fit_vectorizer:
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            tfidf_features = self.vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.vectorizer.transform(texts)
        
        # Custom features
        custom_features = self.feature_extractor.extract_features_batch(texts)
        
        # Combine features
        import scipy.sparse
        combined_features = scipy.sparse.hstack([
            tfidf_features,
            scipy.sparse.csr_matrix(custom_features.values)
        ])
        
        return combined_features
    
    def train(self, texts, severity_labels):
        """Train severity prediction model"""
        print("🔄 Preparing features for severity prediction...")
        X = self.prepare_features(texts, fit_vectorizer=True)
        
        print("🔄 Training severity model...")
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X, severity_labels)
        return self.model
    
    def predict(self, texts):
        """Predict severity"""
        X = self.prepare_features(texts, fit_vectorizer=False)
        return self.model.predict(X)
    
    def evaluate(self, texts, true_labels):
        """Evaluate severity model"""
        predictions = self.predict(texts)
        
        print("\n📊 Severity Prediction Report:")
        print("=" * 50)
        print(classification_report(true_labels, predictions))
        
        return {
            'accuracy': accuracy_score(true_labels, predictions),
            'predictions': predictions
        }
    
    def save_models(self, model_dir="models/enhanced"):
        """Save severity models"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f"{model_dir}/enhanced_severity_model.pkl")
        joblib.dump(self.vectorizer, f"{model_dir}/enhanced_severity_vectorizer.pkl")
        
        print(f"✅ Enhanced severity models saved to {model_dir}/")

def main():
    """Main training pipeline for enhanced models"""
    print("🚀 CTI-NLP Phase 1 Model Enhancement")
    print("=" * 50)
    
    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv("data/Cybersecurity_Dataset.csv")
    df = df.rename(columns=lambda x: x.strip())
    
    # Clean data
    text_col = "Cleaned Threat Description"
    threat_col = "Threat Category"
    severity_col = "Severity Score"
    
    df = df.dropna(subset=[text_col, threat_col])
    
    print(f"📊 Dataset info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Threat categories: {df[threat_col].nunique()}")
    print(f"   Categories: {list(df[threat_col].unique())}")
    
    # Prepare data
    texts = df[text_col].tolist()
    threat_labels = df[threat_col].tolist()
    
    # Map severity scores to labels
    def map_severity(score):
        if pd.isna(score):
            return "Medium"
        score = int(score)
        if score <= 2:
            return "Low"
        elif score == 3:
            return "Medium"
        else:
            return "High"
    
    severity_labels = df[severity_col].apply(map_severity).tolist()
    
    # Split data
    X_train, X_test, y_threat_train, y_threat_test, y_sev_train, y_sev_test = train_test_split(
        texts, threat_labels, severity_labels, 
        test_size=0.2, 
        random_state=42,
        stratify=threat_labels
    )
    
    print(f"📊 Train/Test split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train Enhanced Threat Classifier
    print("\n🎯 Training Enhanced Threat Classifier...")
    threat_classifier = AdvancedThreatClassifier()
    threat_classifier.train(X_train, y_threat_train)
    
    # Evaluate Threat Classifier
    print("\n📊 Evaluating Threat Classifier...")
    threat_results = threat_classifier.evaluate(X_test, y_threat_test)
    
    # Save Threat Classifier
    threat_classifier.save_models()
    
    # Train Enhanced Severity Predictor
    print("\n🎯 Training Enhanced Severity Predictor...")
    severity_predictor = AdvancedSeverityPredictor()
    severity_predictor.train(X_train, y_sev_train)
    
    # Evaluate Severity Predictor
    print("\n📊 Evaluating Severity Predictor...")
    severity_results = severity_predictor.evaluate(X_test, y_sev_test)
    
    # Save Severity Predictor
    severity_predictor.save_models()
    
    # Save evaluation results
    results = {
        'threat_classification': threat_results,
        'severity_prediction': severity_results,
        'model_info': {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'threat_categories': list(set(threat_labels)),
            'severity_levels': list(set(severity_labels))
        }
    }
    
    joblib.dump(results, "models/enhanced/evaluation_results.pkl")
    
    print("\n🎉 Phase 1 Model Enhancement Complete!")
    print("✅ Enhanced models saved to models/enhanced/")
    print("✅ Evaluation results saved")
    
    return results

if __name__ == "__main__":
    results = main()
