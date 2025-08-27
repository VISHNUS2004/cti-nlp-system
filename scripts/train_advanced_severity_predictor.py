"""
Advanced Severity Prediction Model
Combines multiple features and models for better severity assessment
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import re
from datetime import datetime

class ThreatFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom feature extractor for threat intelligence texts"""
    
    def __init__(self):
        self.severity_keywords = {
            'critical': ['critical', 'severe', 'emergency', 'immediate', 'urgent', 'high priority'],
            'high': ['high', 'important', 'significant', 'major', 'serious'],
            'medium': ['medium', 'moderate', 'average', 'standard'],
            'low': ['low', 'minor', 'minimal', 'negligible']
        }
        
        self.threat_indicators = [
            'exploit', 'vulnerability', 'malware', 'attack', 'breach',
            'compromise', 'unauthorized', 'suspicious', 'malicious',
            'phishing', 'ransomware', 'trojan', 'backdoor', 'injection'
        ]
        
        self.impact_words = [
            'damage', 'loss', 'theft', 'disruption', 'outage',
            'compromise', 'breach', 'leak', 'exposure', 'corruption'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract features from threat descriptions"""
        features = []
        
        for text in X:
            if pd.isna(text):
                text = ""
            
            text_lower = text.lower()
            feature_vector = []
            
            # 1. Text length features
            feature_vector.extend([
                len(text),                          # Character count
                len(text.split()),                  # Word count
                len([s for s in text.split('.') if s.strip()])  # Sentence count
            ])
            
            # 2. Severity keyword counts
            for severity, keywords in self.severity_keywords.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                feature_vector.append(count)
            
            # 3. Threat indicator counts
            threat_count = sum(1 for indicator in self.threat_indicators if indicator in text_lower)
            feature_vector.append(threat_count)
            
            # 4. Impact word counts
            impact_count = sum(1 for word in self.impact_words if word in text_lower)
            feature_vector.append(impact_count)
            
            # 5. Technical indicators
            # CVE mentions
            cve_count = len(re.findall(r'CVE-\d{4}-\d{4,7}', text, re.IGNORECASE))
            feature_vector.append(cve_count)
            
            # IP addresses
            ip_count = len(re.findall(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', text))
            feature_vector.append(ip_count)
            
            # Hash values (MD5, SHA1, SHA256)
            hash_patterns = [r'\b[a-fA-F0-9]{32}\b', r'\b[a-fA-F0-9]{40}\b', r'\b[a-fA-F0-9]{64}\b']
            hash_count = sum(len(re.findall(pattern, text)) for pattern in hash_patterns)
            feature_vector.append(hash_count)
            
            # URLs
            url_count = len(re.findall(r'https?://[^\s<>"{}|\\^`\[\]]*', text))
            feature_vector.append(url_count)
            
            # 6. Linguistic features
            # Exclamation marks (urgency indicators)
            exclamation_count = text.count('!')
            feature_vector.append(exclamation_count)
            
            # Capital letters ratio (shouting/urgency)
            if len(text) > 0:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            else:
                caps_ratio = 0
            feature_vector.append(caps_ratio)
            
            # 7. Temporal indicators
            time_indicators = ['immediate', 'urgent', 'asap', 'now', 'emergency']
            time_count = sum(1 for indicator in time_indicators if indicator in text_lower)
            feature_vector.append(time_count)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def get_feature_names(self):
        """Return feature names for interpretability"""
        names = [
            'char_count', 'word_count', 'sentence_count',
            'critical_keywords', 'high_keywords', 'medium_keywords', 'low_keywords',
            'threat_indicators', 'impact_words', 'cve_count', 'ip_count',
            'hash_count', 'url_count', 'exclamation_count', 'caps_ratio',
            'time_indicators'
        ]
        return names

class AdvancedSeverityPredictor:
    def __init__(self):
        self.models = {}
        self.feature_extractor = ThreatFeatureExtractor()
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.ensemble_weights = None
        
    def prepare_data(self, csv_path="data/Cybersecurity_Dataset.csv"):
        """Load and prepare the dataset with enhanced features"""
        df = pd.read_csv(csv_path)
        df = df.rename(columns=lambda x: x.strip())
        
        text_col = "Cleaned Threat Description"
        severity_col = "Severity Score"
        
        # Clean data
        df = df.dropna(subset=[text_col, severity_col])
        
        # Map numeric severity to labels with more granular mapping
        def map_severity(score):
            score = int(score)
            if score == 1:
                return "Low"
            elif score == 2:
                return "Medium-Low"
            elif score == 3:
                return "Medium"
            elif score == 4:
                return "Medium-High"
            else:  # score == 5
                return "High"
        
        df["Severity_Label"] = df[severity_col].apply(map_severity)
        
        return train_test_split(
            df[text_col], df["Severity_Label"], 
            test_size=0.2, random_state=42, stratify=df["Severity_Label"]
        )
    
    def create_combined_features(self, X_text):
        """Create combined feature matrix from text"""
        # Extract custom features
        custom_features = self.feature_extractor.transform(X_text)
        
        # TF-IDF features
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(X_text)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(X_text)
        
        # Combine features
        tfidf_dense = tfidf_features.toarray()
        combined_features = np.hstack([custom_features, tfidf_dense])
        
        return combined_features
    
    def train_models(self, X_train, y_train):
        """Train multiple models for ensemble"""
        print("Training multiple severity prediction models...")
        
        # Create combined features
        X_train_combined = self.create_combined_features(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_combined)
        
        # Define models
        model_configs = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=2000,
                class_weight='balanced',
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            )
        }
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in model_configs.items():
            print(f"Training {name}...")
            
            # Use scaled features for models that benefit from it
            if name in ['logistic_regression', 'neural_network']:
                model.fit(X_train_scaled, y_train)
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                model.fit(X_train_combined, y_train)
                cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5)
            
            self.models[name] = model
            model_scores[name] = np.mean(cv_scores)
            print(f"{name} CV Score: {model_scores[name]:.4f}")
        
        # Calculate ensemble weights based on performance
        total_score = sum(model_scores.values())
        self.ensemble_weights = {
            name: score / total_score 
            for name, score in model_scores.items()
        }
        
        print(f"Ensemble weights: {self.ensemble_weights}")
        return X_train_combined
    
    def predict_severity(self, texts, return_probabilities=False):
        """Predict severity using ensemble of models"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Create features
        X_combined = self.create_combined_features(texts)
        X_scaled = self.scaler.transform(X_combined)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            if name in ['logistic_regression', 'neural_network']:
                pred = model.predict(X_scaled)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)
                else:
                    prob = None
            else:
                pred = model.predict(X_combined)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_combined)
                else:
                    prob = None
            
            predictions[name] = pred
            if prob is not None:
                probabilities[name] = prob
        
        # Ensemble prediction using weighted voting
        if probabilities:
            # Weight probabilities
            weighted_probs = None
            classes = None
            
            for name, prob in probabilities.items():
                weight = self.ensemble_weights.get(name, 1.0)
                if weighted_probs is None:
                    weighted_probs = prob * weight
                    classes = self.models[name].classes_
                else:
                    weighted_probs += prob * weight
            
            # Final predictions
            final_predictions = classes[np.argmax(weighted_probs, axis=1)]
            
            if return_probabilities:
                return final_predictions, weighted_probs, classes, predictions
            else:
                return final_predictions
        else:
            # Simple majority voting if no probabilities available
            final_predictions = []
            for i in range(len(texts)):
                votes = [predictions[name][i] for name in predictions]
                # Most common prediction
                final_pred = max(set(votes), key=votes.count)
                final_predictions.append(final_pred)
            
            return final_predictions
    
    def get_detailed_severity_analysis(self, text):
        """Get detailed severity analysis with explanations"""
        predictions, probabilities, classes, individual_preds = self.predict_severity(
            [text], return_probabilities=True
        )
        
        # Extract custom features for analysis
        custom_features = self.feature_extractor.transform([text])[0]
        feature_names = self.feature_extractor.get_feature_names()
        
        analysis = {
            'text': text,
            'predicted_severity': predictions[0],
            'confidence_scores': dict(zip(classes, probabilities[0])),
            'individual_model_predictions': {
                name: pred[0] for name, pred in individual_preds.items()
            },
            'feature_analysis': dict(zip(feature_names, custom_features)),
            'risk_indicators': [],
            'recommendations': []
        }
        
        # Analyze key features
        if custom_features[feature_names.index('cve_count')] > 0:
            analysis['risk_indicators'].append("Contains CVE references")
        if custom_features[feature_names.index('threat_indicators')] > 2:
            analysis['risk_indicators'].append("Multiple threat indicators present")
        if custom_features[feature_names.index('time_indicators')] > 0:
            analysis['risk_indicators'].append("Urgency indicators present")
        
        # Generate recommendations based on severity
        severity = predictions[0]
        if severity in ['High', 'Medium-High']:
            analysis['recommendations'].extend([
                "Immediate security team notification required",
                "Implement emergency response procedures",
                "Consider system isolation if applicable"
            ])
        elif severity in ['Medium', 'Medium-Low']:
            analysis['recommendations'].extend([
                "Schedule security assessment",
                "Monitor for related indicators",
                "Update security controls"
            ])
        else:
            analysis['recommendations'].extend([
                "Log for future reference",
                "Include in routine security review"
            ])
        
        return analysis
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\nEvaluating severity prediction models...")
        
        X_test_combined = self.create_combined_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_combined)
        
        for name, model in self.models.items():
            print(f"\n{name.title()} Results:")
            
            if name in ['logistic_regression', 'neural_network']:
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test_combined)
            
            print(classification_report(y_test, y_pred))
        
        # Ensemble results
        print(f"\nEnsemble Results:")
        ensemble_pred = self.predict_severity(X_test)
        print(classification_report(y_test, ensemble_pred))
    
    def save_models(self, output_dir="models"):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/severity_{name}.pkl")
        
        # Save preprocessing components
        joblib.dump(self.feature_extractor, f"{output_dir}/severity_feature_extractor.pkl")
        joblib.dump(self.tfidf_vectorizer, f"{output_dir}/severity_tfidf_vectorizer.pkl")
        joblib.dump(self.scaler, f"{output_dir}/severity_scaler.pkl")
        joblib.dump(self.ensemble_weights, f"{output_dir}/severity_ensemble_weights.pkl")
        
        print(f"✅ Advanced severity models saved to {output_dir}/")
    
    def load_models(self, model_dir="models"):
        """Load pre-trained models"""
        try:
            # Load preprocessing components
            self.feature_extractor = joblib.load(f"{model_dir}/severity_feature_extractor.pkl")
            self.tfidf_vectorizer = joblib.load(f"{model_dir}/severity_tfidf_vectorizer.pkl")
            self.scaler = joblib.load(f"{model_dir}/severity_scaler.pkl")
            self.ensemble_weights = joblib.load(f"{model_dir}/severity_ensemble_weights.pkl")
            
            # Load individual models
            model_names = ['random_forest', 'gradient_boosting', 'logistic_regression', 'neural_network']
            for name in model_names:
                model_path = f"{model_dir}/severity_{name}.pkl"
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            print("✅ Advanced severity models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load severity models: {e}")
            return False

def main():
    """Main training pipeline"""
    predictor = AdvancedSeverityPredictor()
    
    # Prepare data
    X_train, X_test, y_train, y_test = predictor.prepare_data()
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Classes: {sorted(set(y_train))}")
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Evaluate
    predictor.evaluate_models(X_test, y_test)
    
    # Save models
    predictor.save_models()
    
    # Test detailed analysis
    print("\n" + "="*60)
    print("DETAILED SEVERITY ANALYSIS")
    print("="*60)
    
    test_text = """
    CRITICAL: Advanced persistent threat detected exploiting CVE-2021-34527 
    vulnerability. Malicious IP 192.168.1.100 established backdoor connection. 
    Immediate action required to prevent data exfiltration!
    """
    
    analysis = predictor.get_detailed_severity_analysis(test_text)
    
    print(f"Text: {analysis['text'][:100]}...")
    print(f"Predicted Severity: {analysis['predicted_severity']}")
    print(f"Confidence Scores: {analysis['confidence_scores']}")
    print(f"Risk Indicators: {analysis['risk_indicators']}")
    print(f"Recommendations: {analysis['recommendations']}")

if __name__ == "__main__":
    main()
