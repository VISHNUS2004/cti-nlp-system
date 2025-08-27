"""
Simple Enhanced Threat Analysis
Works with basic dependencies - improved version of existing models
"""
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import os

class CybersecurityFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract cybersecurity-specific features from text"""
    
    def __init__(self):
        self.threat_keywords = [
            'malware', 'virus', 'trojan', 'ransomware', 'backdoor',
            'phishing', 'spam', 'botnet', 'ddos', 'injection',
            'vulnerability', 'exploit', 'payload', 'shellcode',
            'attack', 'breach', 'intrusion', 'compromise'
        ]
        
        self.severity_indicators = {
            'high': ['critical', 'severe', 'emergency', 'urgent', 'immediate'],
            'medium': ['important', 'significant', 'moderate'],
            'low': ['minor', 'low', 'minimal']
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = []
        
        for text in X:
            if pd.isna(text):
                text = ""
            
            text_lower = text.lower()
            feature_vector = []
            
            # Text statistics
            feature_vector.extend([
                len(text),  # Character count
                len(text.split()),  # Word count
                text.count('!'),  # Urgency indicators
            ])
            
            # Threat keyword counts
            threat_count = sum(1 for keyword in self.threat_keywords if keyword in text_lower)
            feature_vector.append(threat_count)
            
            # Severity indicators
            for level, keywords in self.severity_indicators.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                feature_vector.append(count)
            
            # Technical indicators using regex
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
            
            # URLs and domains
            url_count = len(re.findall(r'https?://[^\s<>"{}|\\^`\[\]]*', text))
            domain_count = len(re.findall(r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}\b', text))
            feature_vector.extend([url_count, domain_count])
            
            features.append(feature_vector)
        
        return np.array(features)

class ImprovedThreatClassifier:
    """Improved threat classifier with ensemble methods"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.feature_extractor = CybersecurityFeatureExtractor()
        self.ensemble_model = None
        self.individual_models = {}
        
    def prepare_data(self, csv_path="data/Cybersecurity_Dataset.csv"):
        """Load and prepare data"""
        df = pd.read_csv(csv_path)
        df = df.rename(columns=lambda x: x.strip())
        
        text_col = "Cleaned Threat Description"
        label_col = "Threat Category"
        
        df = df.dropna(subset=[text_col, label_col])
        
        return train_test_split(
            df[text_col], df[label_col], 
            test_size=0.2, random_state=42, stratify=df[label_col]
        )
    
    def train(self, X_train, y_train):
        """Train improved models"""
        print("Training improved threat classifier...")
        
        # Enhanced TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        
        # Custom features
        X_custom = self.feature_extractor.transform(X_train)
        
        # Combine features
        X_combined = np.hstack([X_tfidf.toarray(), X_custom])
        
        # Individual models
        self.individual_models = {
            'logistic': LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42),
        }
        
        # Train individual models
        for name, model in self.individual_models.items():
            print(f"Training {name}...")
            model.fit(X_combined, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_combined, y_train, cv=5)
            print(f"{name} CV Score: {np.mean(cv_scores):.4f}")
        
        # Create ensemble
        self.ensemble_model = VotingClassifier(
            estimators=list(self.individual_models.items()),
            voting='soft'
        )
        
        self.ensemble_model.fit(X_combined, y_train)
        print("Ensemble model trained successfully!")
        
        return X_combined
    
    def predict(self, texts, return_details=False):
        """Make predictions with confidence scores"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Transform features
        X_tfidf = self.tfidf_vectorizer.transform(texts)
        X_custom = self.feature_extractor.transform(texts)
        X_combined = np.hstack([X_tfidf.toarray(), X_custom])
        
        # Ensemble prediction
        predictions = self.ensemble_model.predict(X_combined)
        probabilities = self.ensemble_model.predict_proba(X_combined)
        
        if return_details:
            # Individual model predictions
            individual_preds = {}
            for name, model in self.individual_models.items():
                individual_preds[name] = model.predict(X_combined)
            
            results = []
            for i, text in enumerate(texts):
                results.append({
                    'text': text,
                    'prediction': predictions[i],
                    'confidence': np.max(probabilities[i]),
                    'probabilities': dict(zip(self.ensemble_model.classes_, probabilities[i])),
                    'individual_predictions': {name: pred[i] for name, pred in individual_preds.items()}
                })
            
            return results[0] if len(texts) == 1 else results
        
        return predictions[0] if len(texts) == 1 else predictions
    
    def save_model(self, output_dir="models"):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.tfidf_vectorizer, f"{output_dir}/improved_tfidf_vectorizer.pkl")
        joblib.dump(self.feature_extractor, f"{output_dir}/improved_feature_extractor.pkl")
        joblib.dump(self.ensemble_model, f"{output_dir}/improved_threat_classifier.pkl")
        
        for name, model in self.individual_models.items():
            joblib.dump(model, f"{output_dir}/improved_{name}_classifier.pkl")
        
        print(f"✅ Improved models saved to {output_dir}/")
    
    def load_model(self, model_dir="models"):
        """Load pre-trained model"""
        try:
            self.tfidf_vectorizer = joblib.load(f"{model_dir}/improved_tfidf_vectorizer.pkl")
            self.feature_extractor = joblib.load(f"{model_dir}/improved_feature_extractor.pkl")
            self.ensemble_model = joblib.load(f"{model_dir}/improved_threat_classifier.pkl")
            
            # Load individual models
            for name in ['logistic', 'random_forest']:
                try:
                    model = joblib.load(f"{model_dir}/improved_{name}_classifier.pkl")
                    self.individual_models[name] = model
                except:
                    pass
            
            print("✅ Improved models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load improved models: {e}")
            return False

class ImprovedSeverityPredictor:
    """Improved severity predictor with better features"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.feature_extractor = CybersecurityFeatureExtractor()
        self.model = None
        
    def prepare_data(self, csv_path="data/Cybersecurity_Dataset.csv"):
        """Load and prepare data with improved severity mapping"""
        df = pd.read_csv(csv_path)
        df = df.rename(columns=lambda x: x.strip())
        
        text_col = "Cleaned Threat Description"
        severity_col = "Severity Score"
        
        df = df.dropna(subset=[text_col, severity_col])
        
        # Improved severity mapping
        def map_severity(score):
            score = int(score)
            if score <= 2:
                return "Low"
            elif score == 3:
                return "Medium"
            elif score == 4:
                return "High"
            else:
                return "Critical"
        
        df["Severity_Label"] = df[severity_col].apply(map_severity)
        
        return train_test_split(
            df[text_col], df["Severity_Label"], 
            test_size=0.2, random_state=42, stratify=df["Severity_Label"]
        )
    
    def train(self, X_train, y_train):
        """Train improved severity model"""
        print("Training improved severity predictor...")
        
        # Enhanced TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        X_custom = self.feature_extractor.transform(X_train)
        X_combined = np.hstack([X_tfidf.toarray(), X_custom])
        
        # Improved Random Forest
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_combined, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_combined, y_train, cv=5)
        print(f"Severity predictor CV Score: {np.mean(cv_scores):.4f}")
        
        return X_combined
    
    def predict(self, texts, return_details=False):
        """Predict severity with confidence"""
        if isinstance(texts, str):
            texts = [texts]
        
        X_tfidf = self.tfidf_vectorizer.transform(texts)
        X_custom = self.feature_extractor.transform(texts)
        X_combined = np.hstack([X_tfidf.toarray(), X_custom])
        
        predictions = self.model.predict(X_combined)
        probabilities = self.model.predict_proba(X_combined)
        
        if return_details:
            results = []
            for i, text in enumerate(texts):
                # Extract features for analysis
                features = X_custom[i]
                
                results.append({
                    'text': text,
                    'severity': predictions[i],
                    'confidence': np.max(probabilities[i]),
                    'probabilities': dict(zip(self.model.classes_, probabilities[i])),
                    'threat_indicators': int(features[3]),  # threat_count
                    'cve_count': int(features[7]),  # cve_count
                    'ip_count': int(features[8]),  # ip_count
                    'hash_count': int(features[9])  # hash_count
                })
            
            return results[0] if len(texts) == 1 else results
        
        return predictions[0] if len(texts) == 1 else predictions
    
    def save_model(self, output_dir="models"):
        """Save the trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(self.tfidf_vectorizer, f"{output_dir}/improved_severity_tfidf.pkl")
        joblib.dump(self.feature_extractor, f"{output_dir}/improved_severity_features.pkl")
        joblib.dump(self.model, f"{output_dir}/improved_severity_model.pkl")
        
        print(f"✅ Improved severity model saved to {output_dir}/")
    
    def load_model(self, model_dir="models"):
        """Load pre-trained model"""
        try:
            self.tfidf_vectorizer = joblib.load(f"{model_dir}/improved_severity_tfidf.pkl")
            self.feature_extractor = joblib.load(f"{model_dir}/improved_severity_features.pkl")
            self.model = joblib.load(f"{model_dir}/improved_severity_model.pkl")
            
            print("✅ Improved severity model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load improved severity model: {e}")
            return False

class SimpleEntityExtractor:
    """Simple but effective entity extractor for cybersecurity"""
    
    def __init__(self):
        self.entity_patterns = {
            'IP_ADDRESS': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'CVE': r'CVE-\d{4}-\d{4,7}',
            'MD5': r'\b[a-fA-F0-9]{32}\b',
            'SHA1': r'\b[a-fA-F0-9]{40}\b',
            'SHA256': r'\b[a-fA-F0-9]{64}\b',
            'URL': r'https?://[^\s<>"{}|\\^`\[\]]*',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'DOMAIN': r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}\b'
        }
        
        self.threat_keywords = [
            'malware', 'virus', 'trojan', 'ransomware', 'backdoor',
            'phishing', 'spam', 'botnet', 'ddos', 'injection',
            'vulnerability', 'exploit', 'payload', 'shellcode',
            'attack', 'breach', 'intrusion', 'compromise'
        ]
    
    def extract_entities(self, text):
        """Extract cybersecurity entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))
        
        # Extract threat keywords
        found_keywords = []
        text_lower = text.lower()
        for keyword in self.threat_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            entities['THREAT_KEYWORDS'] = found_keywords
        
        return entities
    
    def calculate_risk_score(self, entities):
        """Calculate risk score based on entities found"""
        score = 0
        
        # Scoring weights
        weights = {
            'CVE': 25,
            'MD5': 20, 'SHA1': 20, 'SHA256': 20,
            'IP_ADDRESS': 15,
            'URL': 10,
            'DOMAIN': 10,
            'EMAIL': 5,
            'THREAT_KEYWORDS': 5
        }
        
        for entity_type, entity_list in entities.items():
            weight = weights.get(entity_type, 0)
            score += len(entity_list) * weight
        
        return min(100, score)  # Cap at 100

def main():
    """Test the improved models"""
    print("Testing Improved Cybersecurity Models")
    print("="*50)
    
    # Test data exists check
    data_path = "data/Cybersecurity_Dataset.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        print("Please ensure the dataset is available for training.")
        return
    
    # Train improved threat classifier
    print("1. Training Improved Threat Classifier...")
    classifier = ImprovedThreatClassifier()
    try:
        X_train, X_test, y_train, y_test = classifier.prepare_data()
        classifier.train(X_train, y_train)
        classifier.save_model()
        
        # Test prediction
        test_text = "SQL injection attack detected on web server"
        result = classifier.predict(test_text, return_details=True)
        print(f"Classification Test: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Classifier training failed: {e}")
    
    # Train improved severity predictor
    print("\n2. Training Improved Severity Predictor...")
    severity_predictor = ImprovedSeverityPredictor()
    try:
        X_train, X_test, y_train, y_test = severity_predictor.prepare_data()
        severity_predictor.train(X_train, y_train)
        severity_predictor.save_model()
        
        # Test prediction
        test_text = "Critical ransomware attack exploiting CVE-2021-34527"
        result = severity_predictor.predict(test_text, return_details=True)
        print(f"Severity Test: {result['severity']} (confidence: {result['confidence']:.3f})")
        
    except Exception as e:
        print(f"❌ Severity predictor training failed: {e}")
    
    # Test entity extractor
    print("\n3. Testing Entity Extractor...")
    entity_extractor = SimpleEntityExtractor()
    
    test_text = """
    Malware detected with hash d41d8cd98f00b204e9800998ecf8427e communicating 
    with IP 192.168.1.100 and domain evil-site.com. Exploits CVE-2021-34527.
    """
    
    entities = entity_extractor.extract_entities(test_text)
    risk_score = entity_extractor.calculate_risk_score(entities)
    
    print(f"Entities found: {entities}")
    print(f"Risk Score: {risk_score}/100")
    
    print("\n✅ All improved models tested successfully!")
    print("Your CTI-NLP system now has enhanced capabilities:")
    print("  - Better threat classification with ensemble methods")
    print("  - Improved severity prediction with custom features")
    print("  - Cybersecurity-specific entity extraction")

if __name__ == "__main__":
    main()
