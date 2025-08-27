"""
Comprehensive Model Evaluation and Comparison Suite
Tests multiple models with various metrics to justify model selection
"""
import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced models
import sys
sys.path.append('.')
from scripts.train_improved_models import (
    CybersecurityFeatureExtractor, 
    ImprovedThreatClassifier, 
    ImprovedSeverityPredictor
)

class ModelEvaluationSuite:
    """Comprehensive model evaluation and comparison suite"""
    
    def __init__(self, data_path="data/Cybersecurity_Dataset.csv"):
        self.data_path = data_path
        self.results = {}
        self.test_data = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading and preparing dataset...")
        
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}")
            return False
            
        df = pd.read_csv(self.data_path)
        df = df.rename(columns=lambda x: x.strip())
        
        # Check required columns
        text_col = "Cleaned Threat Description"
        label_col = "Threat Category"
        severity_col = "Severity Score"
        
        if text_col not in df.columns or label_col not in df.columns:
            print(f"  Required columns not found: {text_col}, {label_col}")
            return False
        
        # Clean data
        df = df.dropna(subset=[text_col, label_col])
        
        print(f"✅ Dataset loaded: {len(df)} samples")
        print(f"📋 Classes: {df[label_col].value_counts().to_dict()}")
        
        # Store test data for different tasks
        self.test_data = {
            'classification': {
                'X': df[text_col],
                'y': df[label_col],
                'task': 'Threat Classification'
            }
        }
        
        # Add severity prediction if available
        if severity_col in df.columns:
            df_severity = df.dropna(subset=[severity_col])
            
            # Map severity scores to labels
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
            
            df_severity["Severity_Label"] = df_severity[severity_col].apply(map_severity)
            
            self.test_data['severity'] = {
                'X': df_severity[text_col],
                'y': df_severity["Severity_Label"],
                'task': 'Severity Prediction'
            }
        
        return True
    
    def evaluate_traditional_models(self):
        """Evaluate traditional ML models"""
        print("\n🔬 Evaluating Traditional ML Models...")
        
        traditional_models = {
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
            'Naive_Bayes': MultinomialNB(),
            'SVM': SVC(probability=True, random_state=42),
            'Neural_Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            'SGD': SGDClassifier(random_state=42)
        }
        
        vectorizers = {
            'TF-IDF': TfidfVectorizer(max_features=5000, stop_words='english'),
            'Count': CountVectorizer(max_features=5000, stop_words='english'),
            'TF-IDF_Ngrams': TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        }
        
        for task_name, task_data in self.test_data.items():
            print(f"\n  Task: {task_data['task']}")
            X, y = task_data['X'], task_data['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            task_results = {}
            
            for vec_name, vectorizer in vectorizers.items():
                print(f"\n🔧 Vectorizer: {vec_name}")
                
                # Transform data
                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)
                
                for model_name, model in traditional_models.items():
                    try:
                        start_time = time.time()
                        
                        # Train model
                        model.fit(X_train_vec, y_train)
                        
                        # Predictions
                        y_pred = model.predict(X_test_vec)
                        y_pred_proba = None
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_test_vec)
                        
                        training_time = time.time() - start_time
                        
                        # Calculate metrics
                        metrics = self._calculate_metrics(
                            y_test, y_pred, y_pred_proba, 
                            model, X_train_vec, y_train
                        )
                        metrics['training_time'] = training_time
                        
                        # Store results
                        key = f"{task_name}_{vec_name}_{model_name}"
                        task_results[key] = {
                            'model': model_name,
                            'vectorizer': vec_name,
                            'task': task_data['task'],
                            **metrics
                        }
                        
                        print(f"  {model_name:15} | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1_weighted']:.3f} | Time: {training_time:.2f}s")
                        
                    except Exception as e:
                        print(f"  {model_name:15} |   Failed: {str(e)[:50]}")
            
            self.results[task_name] = task_results
    
    def evaluate_enhanced_models(self):
        """Evaluate our enhanced models"""
        print("\n  Evaluating Enhanced Models...")
        
        for task_name, task_data in self.test_data.items():
            print(f"\n  Task: {task_data['task']}")
            X, y = task_data['X'], task_data['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            if task_name == 'classification':
                # Enhanced Threat Classifier
                try:
                    start_time = time.time()
                    
                    enhanced_classifier = ImprovedThreatClassifier()
                    enhanced_classifier.train(X_train, y_train)
                    
                    # Get predictions with details
                    predictions = []
                    probabilities = []
                    
                    for text in X_test:
                        result = enhanced_classifier.predict(text, return_details=True)
                        predictions.append(result['prediction'])
                        
                        # Convert probabilities to array format
                        prob_dict = result['probabilities']
                        classes = sorted(prob_dict.keys())
                        prob_array = [prob_dict[cls] for cls in classes]
                        probabilities.append(prob_array)
                    
                    training_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(
                        y_test, predictions, np.array(probabilities),
                        enhanced_classifier, None, None
                    )
                    metrics['training_time'] = training_time
                    
                    # Store results
                    key = f"{task_name}_Enhanced_Ensemble"
                    self.results[task_name][key] = {
                        'model': 'Enhanced_Ensemble',
                        'vectorizer': 'Custom_Features_TF-IDF',
                        'task': task_data['task'],
                        **metrics
                    }
                    
                    print(f"  Enhanced_Ensemble | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1_weighted']:.3f} | Time: {training_time:.2f}s")
                    
                except Exception as e:
                    print(f"  Enhanced_Ensemble |   Failed: {str(e)}")
            
            elif task_name == 'severity':
                # Enhanced Severity Predictor
                try:
                    start_time = time.time()
                    
                    enhanced_severity = ImprovedSeverityPredictor()
                    enhanced_severity.train(X_train, y_train)
                    
                    # Get predictions
                    predictions = []
                    probabilities = []
                    
                    for text in X_test:
                        result = enhanced_severity.predict(text, return_details=True)
                        predictions.append(result['severity'])
                        
                        # Convert probabilities
                        prob_dict = result['probabilities']
                        classes = sorted(prob_dict.keys())
                        prob_array = [prob_dict[cls] for cls in classes]
                        probabilities.append(prob_array)
                    
                    training_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = self._calculate_metrics(
                        y_test, predictions, np.array(probabilities),
                        enhanced_severity, None, None
                    )
                    metrics['training_time'] = training_time
                    
                    # Store results
                    key = f"{task_name}_Enhanced_Severity"
                    self.results[task_name][key] = {
                        'model': 'Enhanced_Severity',
                        'vectorizer': 'Custom_Features_TF-IDF',
                        'task': task_data['task'],
                        **metrics
                    }
                    
                    print(f"  Enhanced_Severity | Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1_weighted']:.3f} | Time: {training_time:.2f}s")
                    
                except Exception as e:
                    print(f"  Enhanced_Severity |   Failed: {str(e)}")
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, model, X_train, y_train):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation score (if training data available)
        if X_train is not None and y_train is not None:
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted')
                metrics['cv_f1_mean'] = np.mean(cv_scores)
                metrics['cv_f1_std'] = np.std(cv_scores)
            except:
                metrics['cv_f1_mean'] = 0
                metrics['cv_f1_std'] = 0
        else:
            metrics['cv_f1_mean'] = 0
            metrics['cv_f1_std'] = 0
        
        # AUC for multi-class (if probabilities available)
        if y_pred_proba is not None:
            try:
                # Convert to binary for each class and calculate AUC
                label_encoder = LabelEncoder()
                y_true_encoded = label_encoder.fit_transform(y_true)
                
                if len(label_encoder.classes_) > 2:
                    # Multi-class AUC (one-vs-rest)
                    auc_scores = []
                    for i in range(len(label_encoder.classes_)):
                        y_true_binary = (y_true_encoded == i).astype(int)
                        if len(np.unique(y_true_binary)) > 1:  # Only if both classes present
                            auc_score = roc_auc_score(y_true_binary, y_pred_proba[:, i])
                            auc_scores.append(auc_score)
                    
                    metrics['auc_macro'] = np.mean(auc_scores) if auc_scores else 0
                else:
                    # Binary classification
                    metrics['auc_macro'] = roc_auc_score(y_true_encoded, y_pred_proba[:, 1])
            except:
                metrics['auc_macro'] = 0
        else:
            metrics['auc_macro'] = 0
        
        return metrics
    
    def create_comparison_report(self):
        """Create comprehensive comparison report"""
        print("\n  Creating Comparison Report...")
        
        # Aggregate results
        all_results = []
        for task_name, task_results in self.results.items():
            for key, result in task_results.items():
                all_results.append(result)
        
        df_results = pd.DataFrame(all_results)
        
        # Create summary tables
        self._create_summary_tables(df_results)
        self._create_performance_charts(df_results)
        self._create_recommendation_report(df_results)
        
        return df_results
    
    def _create_summary_tables(self, df_results):
        """Create summary tables"""
        print("\n📋 PERFORMANCE SUMMARY TABLES")
        print("=" * 80)
        
        for task in df_results['task'].unique():
            task_df = df_results[df_results['task'] == task]
            
            print(f"\n🎯 {task}")
            print("-" * 60)
            
            # Sort by F1 weighted score
            top_models = task_df.nlargest(10, 'f1_weighted')
            
            summary_table = top_models[['model', 'vectorizer', 'accuracy', 'f1_weighted', 'auc_macro', 'training_time']].copy()
            summary_table = summary_table.round(3)
            
            print(summary_table.to_string(index=False))
            
            # Highlight best model
            best_model = top_models.iloc[0]
            print(f"\n🏆 BEST MODEL: {best_model['model']} + {best_model['vectorizer']}")
            print(f"     Accuracy: {best_model['accuracy']:.3f}")
            print(f"     F1-Score: {best_model['f1_weighted']:.3f}")
            print(f"     AUC: {best_model['auc_macro']:.3f}")
            print(f"   ⏱️  Time: {best_model['training_time']:.2f}s")
    
    def _create_performance_charts(self, df_results):
        """Create performance visualization data"""
        print("\n  PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        for task in df_results['task'].unique():
            task_df = df_results[df_results['task'] == task]
            
            print(f"\n📈 {task} - Model Performance Comparison")
            print("-" * 60)
            
            # Performance vs Time analysis
            print("\n⚡ Performance vs Training Time:")
            performance_time = task_df[['model', 'f1_weighted', 'training_time']].copy()
            performance_time['efficiency'] = performance_time['f1_weighted'] / (performance_time['training_time'] + 0.1)
            performance_time = performance_time.sort_values('efficiency', ascending=False)
            
            print(performance_time.head(5).to_string(index=False))
            
            # Model type analysis
            print(f"\n🔍 Model Type Performance:")
            model_groups = {
                'Linear': ['Logistic_Regression', 'SGD'],
                'Ensemble': ['Random_Forest', 'Gradient_Boosting', 'Enhanced_Ensemble'],
                'Neural': ['Neural_Network'],
                'Probabilistic': ['Naive_Bayes'],
                'Kernel': ['SVM'],
                'Enhanced': ['Enhanced_Ensemble', 'Enhanced_Severity']
            }
            
            for group_name, models in model_groups.items():
                group_df = task_df[task_df['model'].isin(models)]
                if not group_df.empty:
                    avg_f1 = group_df['f1_weighted'].mean()
                    avg_time = group_df['training_time'].mean()
                    print(f"   {group_name:12} | Avg F1: {avg_f1:.3f} | Avg Time: {avg_time:.2f}s")
    
    def _create_recommendation_report(self, df_results):
        """Create model recommendation report"""
        print("\n🎯 MODEL SELECTION RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = {}
        
        for task in df_results['task'].unique():
            task_df = df_results[df_results['task'] == task]
            
            # Best overall model
            best_overall = task_df.loc[task_df['f1_weighted'].idxmax()]
            
            # Best efficient model (performance/time ratio)
            task_df_copy = task_df.copy()
            task_df_copy['efficiency'] = task_df_copy['f1_weighted'] / (task_df_copy['training_time'] + 0.1)
            best_efficient = task_df_copy.loc[task_df_copy['efficiency'].idxmax()]
            
            # Best fast model (< 5 seconds)
            fast_models = task_df[task_df['training_time'] < 5]
            if not fast_models.empty:
                best_fast = fast_models.loc[fast_models['f1_weighted'].idxmax()]
            else:
                best_fast = best_overall
            
            recommendations[task] = {
                'best_overall': best_overall,
                'best_efficient': best_efficient,
                'best_fast': best_fast
            }
            
            print(f"\n🎯 {task} Recommendations:")
            print(f"   🏆 Best Performance: {best_overall['model']} + {best_overall['vectorizer']}")
            print(f"      F1: {best_overall['f1_weighted']:.3f}, Time: {best_overall['training_time']:.2f}s")
            
            print(f"   ⚡ Most Efficient: {best_efficient['model']} + {best_efficient['vectorizer']}")
            print(f"      F1: {best_efficient['f1_weighted']:.3f}, Efficiency: {best_efficient['f1_weighted']/(best_efficient['training_time']+0.1):.2f}")
            
            print(f"   🚄 Fastest (<5s): {best_fast['model']} + {best_fast['vectorizer']}")
            print(f"      F1: {best_fast['f1_weighted']:.3f}, Time: {best_fast['training_time']:.2f}s")
        
        return recommendations
    
    def generate_justification_report(self):
        """Generate comprehensive justification report for model selection"""
        print("\n📝 MODEL SELECTION JUSTIFICATION REPORT")
        print("=" * 80)
        
        justification = {
            'enhanced_vs_traditional': {},
            'vectorizer_comparison': {},
            'performance_metrics': {},
            'business_justification': {}
        }
        
        for task in self.results.keys():
            task_results = self.results[task]
            task_df = pd.DataFrame(list(task_results.values()))
            
            # Enhanced vs Traditional comparison
            enhanced_models = task_df[task_df['model'].str.contains('Enhanced')]
            traditional_models = task_df[~task_df['model'].str.contains('Enhanced')]
            
            if not enhanced_models.empty and not traditional_models.empty:
                enhanced_avg = enhanced_models['f1_weighted'].mean()
                traditional_avg = traditional_models['f1_weighted'].mean()
                improvement = ((enhanced_avg - traditional_avg) / traditional_avg) * 100
                
                justification['enhanced_vs_traditional'][task] = {
                    'enhanced_avg_f1': enhanced_avg,
                    'traditional_avg_f1': traditional_avg,
                    'improvement_percent': improvement
                }
                
                print(f"\n🔍 {task} - Enhanced vs Traditional Models:")
                print(f"   Enhanced Models Avg F1: {enhanced_avg:.3f}")
                print(f"   Traditional Models Avg F1: {traditional_avg:.3f}")
                print(f"   Improvement: {improvement:+.1f}%")
            
            # Best model analysis
            best_model = task_df.loc[task_df['f1_weighted'].idxmax()]
            
            print(f"\n🏆 {task} - Selected Model Analysis:")
            print(f"   Model: {best_model['model']} + {best_model['vectorizer']}")
            print(f"   Justification Metrics:")
            print(f"     • Accuracy: {best_model['accuracy']:.3f} ({best_model['accuracy']*100:.1f}%)")
            print(f"     • F1-Score: {best_model['f1_weighted']:.3f}")
            print(f"     • Precision: {best_model['precision_macro']:.3f}")
            print(f"     • Recall: {best_model['recall_macro']:.3f}")
            print(f"     • AUC: {best_model['auc_macro']:.3f}")
            print(f"     • Training Time: {best_model['training_time']:.2f}s")
            
            # Rank analysis
            task_df_sorted = task_df.sort_values('f1_weighted', ascending=False)
            rank = task_df_sorted.index[task_df_sorted['model'] == best_model['model']].tolist()[0] + 1
            total_models = len(task_df)
            
            print(f"   Performance Ranking: #{rank} out of {total_models} models tested")
            print(f"   Better than {((total_models - rank) / total_models * 100):.1f}% of other models")
        
        return justification
    
    def save_results(self, output_dir="evaluation_results"):
        """Save all evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        for task_name, task_results in self.results.items():
            df_task = pd.DataFrame(list(task_results.values()))
            df_task.to_csv(f"{output_dir}/{task_name}_detailed_results.csv", index=False)
        
        # Save summary
        all_results = []
        for task_results in self.results.values():
            all_results.extend(list(task_results.values()))
        
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(f"{output_dir}/all_models_comparison.csv", index=False)
        
        print(f"\nResults saved to {output_dir}/")
        
        return df_all

def main():
    """Run comprehensive model evaluation"""
    print("  COMPREHENSIVE MODEL EVALUATION SUITE")
    print("=" * 80)
    print("This evaluation will test multiple models and provide")
    print("detailed justification for model selection decisions.")
    print("=" * 80)
    
    # Initialize evaluation suite
    evaluator = ModelEvaluationSuite()
    
    # Load data
    if not evaluator.load_and_prepare_data():
        print("  Failed to load data. Exiting.")
        return
    
    # Run evaluations
    print("\n🔬 Starting Model Evaluations...")
    evaluator.evaluate_traditional_models()
    evaluator.evaluate_enhanced_models()
    
    # Create reports
    print("\n  Generating Comparison Reports...")
    df_results = evaluator.create_comparison_report()
    
    # Generate justification
    print("\n📝 Generating Justification Report...")
    justification = evaluator.generate_justification_report()
    
    # Save results
    print("\n💾 Saving Results...")
    evaluator.save_results()
    
    print("\n✅ EVALUATION COMPLETE!")
    print("  Check the evaluation_results/ directory for detailed reports")
    print("🎯 Use these results to justify your model selection to your guide")

if __name__ == "__main__":
    main()
