"""
Advanced Threat Classifier with Multiple Models and Ensemble
Combines BERT, TF-IDF, and traditional ML for better performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import joblib
import os
import torch

class AdvancedThreatClassifier:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.logistic_model = None
        self.rf_model = None
        self.bert_classifier = None
        self.ensemble_model = None
        
    def prepare_data(self, csv_path="data/Cybersecurity_Dataset.csv"):
        """Load and prepare the dataset"""
        df = pd.read_csv(csv_path)
        df = df.rename(columns=lambda x: x.strip())
        
        text_col = "Cleaned Threat Description"
        label_col = "Threat Category"
        
        # Clean data
        df = df.dropna(subset=[text_col, label_col])
        
        return train_test_split(
            df[text_col], df[label_col], 
            test_size=0.2, random_state=42, stratify=df[label_col]
        )
    
    def train_tfidf_models(self, X_train, y_train):
        """Train TF-IDF based models"""
        print("Training TF-IDF models...")
        
        # Enhanced TF-IDF with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Include bigrams and trigrams
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        
        # Logistic Regression with better regularization
        self.logistic_model = LogisticRegression(
            max_iter=2000,
            C=0.1,  # Stronger regularization
            class_weight='balanced'
        )
        self.logistic_model.fit(X_train_tfidf, y_train)
        
        # Random Forest for ensemble
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        self.rf_model.fit(X_train_tfidf, y_train)
        
        return X_train_tfidf
    
    def setup_bert_classifier(self):
        """Setup BERT-based classifier"""
        print("Setting up BERT classifier...")
        try:
            # Use a cybersecurity-focused model if available, otherwise distilbert
            model_name = "distilbert-base-uncased"
            self.bert_classifier = pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            print(f"BERT setup failed: {e}")
            self.bert_classifier = None
    
    def create_ensemble(self, X_train_tfidf, y_train):
        """Create ensemble of traditional models"""
        print("Creating ensemble model...")
        
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('logistic', self.logistic_model),
                ('rf', self.rf_model)
            ],
            voting='soft'  # Use probability voting
        )
        
        self.ensemble_model.fit(X_train_tfidf, y_train)
    
    def predict_with_ensemble(self, texts):
        """Make predictions using ensemble approach"""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        
        for text in texts:
            predictions = {}
            confidences = {}
            
            # TF-IDF based predictions
            X_tfidf = self.tfidf_vectorizer.transform([text])
            
            # Logistic Regression
            lr_pred = self.logistic_model.predict(X_tfidf)[0]
            lr_prob = max(self.logistic_model.predict_proba(X_tfidf)[0])
            predictions['logistic'] = lr_pred
            confidences['logistic'] = lr_prob
            
            # Random Forest
            rf_pred = self.rf_model.predict(X_tfidf)[0]
            rf_prob = max(self.rf_model.predict_proba(X_tfidf)[0])
            predictions['random_forest'] = rf_pred
            confidences['random_forest'] = rf_prob
            
            # Ensemble
            ensemble_pred = self.ensemble_model.predict(X_tfidf)[0]
            ensemble_prob = max(self.ensemble_model.predict_proba(X_tfidf)[0])
            predictions['ensemble'] = ensemble_pred
            confidences['ensemble'] = ensemble_prob
            
            # BERT (if available)
            if self.bert_classifier:
                try:
                    bert_results = self.bert_classifier(text)
                    if bert_results:
                        best_bert = max(bert_results[0], key=lambda x: x['score'])
                        predictions['bert'] = best_bert['label']
                        confidences['bert'] = best_bert['score']
                except Exception as e:
                    print(f"BERT prediction failed: {e}")
            
            # Final ensemble decision (weighted voting)
            weights = {
                'logistic': 0.25,
                'random_forest': 0.25,
                'ensemble': 0.35,
                'bert': 0.15 if 'bert' in predictions else 0
            }
            
            # Weighted confidence score
            final_confidence = sum(
                weights.get(model, 0) * conf 
                for model, conf in confidences.items()
            )
            
            # Most common prediction (simple voting)
            pred_counts = {}
            for pred in predictions.values():
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            
            final_prediction = max(pred_counts.items(), key=lambda x: x[1])[0]
            
            results.append({
                'text': text,
                'final_prediction': final_prediction,
                'confidence': final_confidence,
                'individual_predictions': predictions,
                'individual_confidences': confidences
            })
        
        return results[0] if len(texts) == 1 else results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models"""
        print("\nEvaluating models...")
        
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        models = {
            'Logistic Regression': self.logistic_model,
            'Random Forest': self.rf_model,
            'Ensemble': self.ensemble_model
        }
        
        for name, model in models.items():
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n{name} Accuracy: {accuracy:.4f}")
            print(f"{name} Classification Report:")
            print(classification_report(y_test, y_pred))
    
    def save_models(self, output_dir="models"):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save TF-IDF models
        joblib.dump(self.tfidf_vectorizer, f"{output_dir}/advanced_tfidf_vectorizer.pkl")
        joblib.dump(self.logistic_model, f"{output_dir}/advanced_logistic_model.pkl")
        joblib.dump(self.rf_model, f"{output_dir}/advanced_rf_model.pkl")
        joblib.dump(self.ensemble_model, f"{output_dir}/advanced_ensemble_model.pkl")
        
        print(f"✅ Advanced models saved to {output_dir}/")
    
    def load_models(self, model_dir="models"):
        """Load pre-trained models"""
        try:
            self.tfidf_vectorizer = joblib.load(f"{model_dir}/advanced_tfidf_vectorizer.pkl")
            self.logistic_model = joblib.load(f"{model_dir}/advanced_logistic_model.pkl")
            self.rf_model = joblib.load(f"{model_dir}/advanced_rf_model.pkl")
            self.ensemble_model = joblib.load(f"{model_dir}/advanced_ensemble_model.pkl")
            self.setup_bert_classifier()
            print("✅ Advanced models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False

def main():
    """Main training pipeline"""
    classifier = AdvancedThreatClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data()
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train models
    X_train_tfidf = classifier.train_tfidf_models(X_train, y_train)
    classifier.setup_bert_classifier()
    classifier.create_ensemble(X_train_tfidf, y_train)
    
    # Evaluate
    classifier.evaluate_models(X_test, y_test)
    
    # Save models
    classifier.save_models()
    
    # Test ensemble prediction
    print("\n" + "="*50)
    print("Testing Ensemble Prediction:")
    test_text = "SQL injection attack detected on web application server"
    result = classifier.predict_with_ensemble(test_text)
    
    print(f"Text: {result['text']}")
    print(f"Final Prediction: {result['final_prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Individual Predictions: {result['individual_predictions']}")

if __name__ == "__main__":
    main()
