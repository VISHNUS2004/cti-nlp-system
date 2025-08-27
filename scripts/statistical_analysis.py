"""
Statistical Analysis and Visualization for Model Justification
Creates charts and statistical tests to support model selection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def create_model_comparison_charts():
    """Create comprehensive visualization charts"""
    
    # Load results
    if not os.path.exists('evaluation_results/all_models_comparison.csv'):
        print("❌ Evaluation results not found. Please run comprehensive_model_evaluation.py first.")
        return
    
    df = pd.read_csv('evaluation_results/all_models_comparison.csv')
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Comparison by Task
    plt.subplot(3, 3, 1)
    task_performance = df.groupby('task')['f1_weighted'].mean().sort_values(ascending=False)
    bars = plt.bar(task_performance.index, task_performance.values, color=['#2E8B57', '#4169E1'])
    plt.title('Average F1-Score by Task', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Model Performance Distribution
    plt.subplot(3, 3, 2)
    model_performance = df.groupby('model')['f1_weighted'].mean().sort_values(ascending=False).head(8)
    bars = plt.bar(range(len(model_performance)), model_performance.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(model_performance))))
    plt.title('Top 8 Models by Average F1-Score', fontsize=14, fontweight='bold')
    plt.ylabel('F1-Score')
    plt.xticks(range(len(model_performance)), model_performance.index, rotation=45, ha='right')
    
    # Highlight SGD
    for i, (model, score) in enumerate(model_performance.items()):
        if 'SGD' in model:
            bars[i].set_color('#FFD700')  # Gold color for SGD
            plt.text(i, score + 0.005, '★ SELECTED', ha='center', va='bottom', 
                    fontweight='bold', color='red', fontsize=10)
    
    # 3. Training Time vs Performance
    plt.subplot(3, 3, 3)
    plt.scatter(df['training_time'], df['f1_weighted'], alpha=0.6, s=60)
    
    # Highlight SGD models
    sgd_data = df[df['model'] == 'SGD']
    plt.scatter(sgd_data['training_time'], sgd_data['f1_weighted'], 
               color='red', s=100, label='SGD (Selected)', marker='*')
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('F1-Score')
    plt.title('Training Time vs Performance', fontsize=14, fontweight='bold')
    plt.legend()
    
    # Add efficiency frontier line
    efficient_models = df.nlargest(5, 'f1_weighted')
    plt.plot(efficient_models['training_time'], efficient_models['f1_weighted'], 
             'r--', alpha=0.5, label='Top Performers')
    
    # 4. Vectorizer Comparison
    plt.subplot(3, 3, 4)
    vectorizer_performance = df.groupby('vectorizer')['f1_weighted'].mean().sort_values(ascending=False)
    bars = plt.bar(vectorizer_performance.index, vectorizer_performance.values, 
                   color=['#FF6B35', '#F7931E', '#FFD23F'])
    plt.title('Vectorizer Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Average F1-Score')
    plt.xticks(rotation=45)
    
    # Highlight Count vectorizer
    for i, (vec, score) in enumerate(vectorizer_performance.items()):
        if 'Count' in vec:
            bars[i].set_color('#FFD700')
            plt.text(i, score + 0.005, '★ SELECTED', ha='center', va='bottom', 
                    fontweight='bold', color='red', fontsize=10)
    
    # 5. Model Type Analysis
    plt.subplot(3, 3, 5)
    model_types = {
        'Linear': ['Logistic_Regression', 'SGD'],
        'Ensemble': ['Random_Forest', 'Gradient_Boosting'],
        'Kernel': ['SVM'],
        'Probabilistic': ['Naive_Bayes'],
        'Neural': ['Neural_Network'],
        'Enhanced': ['Enhanced_Ensemble', 'Enhanced_Severity']
    }
    
    type_performance = {}
    for type_name, models in model_types.items():
        type_data = df[df['model'].isin(models)]
        if not type_data.empty:
            type_performance[type_name] = type_data['f1_weighted'].mean()
    
    sorted_types = dict(sorted(type_performance.items(), key=lambda x: x[1], reverse=True))
    colors = ['#FFD700' if 'Linear' in k else '#87CEEB' for k in sorted_types.keys()]
    
    bars = plt.bar(sorted_types.keys(), sorted_types.values(), color=colors)
    plt.title('Model Type Performance', fontsize=14, fontweight='bold')
    plt.ylabel('Average F1-Score')
    plt.xticks(rotation=45)
    
    # 6. Accuracy vs F1-Score Correlation
    plt.subplot(3, 3, 6)
    plt.scatter(df['accuracy'], df['f1_weighted'], alpha=0.6, s=60)
    
    # Add correlation line
    correlation = np.corrcoef(df['accuracy'], df['f1_weighted'])[0, 1]
    plt.plot(df['accuracy'], np.poly1d(np.polyfit(df['accuracy'], df['f1_weighted'], 1))(df['accuracy']), 
             'r--', alpha=0.8)
    
    plt.xlabel('Accuracy')
    plt.ylabel('F1-Score')
    plt.title(f'Accuracy vs F1-Score (r={correlation:.3f})', fontsize=14, fontweight='bold')
    
    # 7. Cross-Validation Stability
    plt.subplot(3, 3, 7)
    cv_data = df[df['cv_f1_mean'] > 0]  # Only models with CV data
    if not cv_data.empty:
        plt.errorbar(range(len(cv_data)), cv_data['cv_f1_mean'], 
                    yerr=cv_data['cv_f1_std'], fmt='o', capsize=5)
        plt.xticks(range(len(cv_data)), 
                  [f"{row['model'][:8]}\n{row['vectorizer'][:8]}" for _, row in cv_data.iterrows()], 
                  rotation=45, ha='right')
        plt.title('Cross-Validation Stability', fontsize=14, fontweight='bold')
        plt.ylabel('CV F1-Score ± Std')
    else:
        plt.text(0.5, 0.5, 'No CV Data Available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Cross-Validation Stability', fontsize=14, fontweight='bold')
    
    # 8. Performance Distribution by Task
    plt.subplot(3, 3, 8)
    df_clean = df[df['f1_weighted'] > 0]  # Remove zero scores
    task_names = df_clean['task'].unique()
    
    for i, task in enumerate(task_names):
        task_data = df_clean[df_clean['task'] == task]['f1_weighted']
        plt.hist(task_data, alpha=0.7, label=task, bins=15)
    
    plt.xlabel('F1-Score')
    plt.ylabel('Frequency')
    plt.title('F1-Score Distribution by Task', fontsize=14, fontweight='bold')
    plt.legend()
    
    # 9. Model Efficiency (Performance/Time)
    plt.subplot(3, 3, 9)
    df['efficiency'] = df['f1_weighted'] / (df['training_time'] + 0.001)  # Avoid division by zero
    efficiency_top = df.nlargest(10, 'efficiency')
    
    bars = plt.bar(range(len(efficiency_top)), efficiency_top['efficiency'], 
                   color=plt.cm.plasma(np.linspace(0, 1, len(efficiency_top))))
    plt.title('Model Efficiency (F1/Time)', fontsize=14, fontweight='bold')
    plt.ylabel('Efficiency Score')
    plt.xticks(range(len(efficiency_top)), 
              [f"{row['model'][:8]}" for _, row in efficiency_top.iterrows()], 
              rotation=45, ha='right')
    
    # Highlight SGD
    for i, (_, row) in enumerate(efficiency_top.iterrows()):
        if 'SGD' in row['model']:
            bars[i].set_color('#FFD700')
            plt.text(i, row['efficiency'] + 0.1, '★', ha='center', va='bottom', 
                    fontweight='bold', color='red', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/model_comparison_charts.png', dpi=300, bbox_inches='tight')
    plt.savefig('evaluation_results/model_comparison_charts.pdf', bbox_inches='tight')
    print("✅ Visualization saved as 'model_comparison_charts.png' and '.pdf'")
    
    return fig

def statistical_analysis():
    """Perform statistical analysis for model justification"""
    
    if not os.path.exists('evaluation_results/all_models_comparison.csv'):
        print("❌ Evaluation results not found.")
        return
    
    df = pd.read_csv('evaluation_results/all_models_comparison.csv')
    
    print("\n📊 STATISTICAL ANALYSIS REPORT")
    print("=" * 60)
    
    # 1. Best Model Statistics
    best_model = df.loc[df['f1_weighted'].idxmax()]
    print(f"\n🏆 BEST MODEL STATISTICS:")
    print(f"   Model: {best_model['model']} + {best_model['vectorizer']}")
    print(f"   Task: {best_model['task']}")
    print(f"   F1-Score: {best_model['f1_weighted']:.4f}")
    print(f"   Accuracy: {best_model['accuracy']:.4f}")
    print(f"   Training Time: {best_model['training_time']:.4f}s")
    
    # 2. Statistical Significance Tests
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    
    # Compare SGD vs other models
    sgd_scores = df[df['model'] == 'SGD']['f1_weighted']
    other_scores = df[df['model'] != 'SGD']['f1_weighted']
    
    if len(sgd_scores) > 1 and len(other_scores) > 1:
        # T-test
        t_stat, p_value = stats.ttest_ind(sgd_scores, other_scores)
        print(f"   SGD vs Others T-test: t={t_stat:.4f}, p={p_value:.4f}")
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(sgd_scores, other_scores, alternative='two-sided')
        print(f"   Mann-Whitney U test: U={u_stat:.4f}, p={u_p_value:.4f}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((sgd_scores.var() + other_scores.var()) / 2)
        cohens_d = (sgd_scores.mean() - other_scores.mean()) / pooled_std
        print(f"   Effect Size (Cohen's d): {cohens_d:.4f}")
    
    # 3. Performance Distribution Analysis
    print(f"\n📊 PERFORMANCE DISTRIBUTION:")
    print(f"   Mean F1-Score: {df['f1_weighted'].mean():.4f}")
    print(f"   Median F1-Score: {df['f1_weighted'].median():.4f}")
    print(f"   Std Deviation: {df['f1_weighted'].std():.4f}")
    print(f"   Min F1-Score: {df['f1_weighted'].min():.4f}")
    print(f"   Max F1-Score: {df['f1_weighted'].max():.4f}")
    
    # 4. Correlation Analysis
    print(f"\n🔗 CORRELATION ANALYSIS:")
    numeric_cols = ['accuracy', 'f1_weighted', 'training_time', 'auc_macro']
    correlation_matrix = df[numeric_cols].corr()
    
    print(f"   Accuracy vs F1-Score: {correlation_matrix.loc['accuracy', 'f1_weighted']:.4f}")
    print(f"   F1-Score vs Training Time: {correlation_matrix.loc['f1_weighted', 'training_time']:.4f}")
    print(f"   Accuracy vs Training Time: {correlation_matrix.loc['accuracy', 'training_time']:.4f}")
    
    # 5. Model Ranking Analysis
    print(f"\n🏅 MODEL RANKING:")
    model_ranking = df.groupby('model')['f1_weighted'].mean().sort_values(ascending=False)
    
    for i, (model, score) in enumerate(model_ranking.head(5).items(), 1):
        print(f"   #{i}: {model} - F1: {score:.4f}")
    
    # 6. Confidence Intervals
    print(f"\n📏 CONFIDENCE INTERVALS (95%):")
    for task in df['task'].unique():
        task_data = df[df['task'] == task]['f1_weighted']
        mean_score = task_data.mean()
        sem = stats.sem(task_data)
        ci = stats.t.interval(0.95, len(task_data)-1, loc=mean_score, scale=sem)
        print(f"   {task}: {mean_score:.4f} ± {sem:.4f} (CI: {ci[0]:.4f} - {ci[1]:.4f})")
    
    return df

def create_summary_table():
    """Create publication-ready summary table"""
    
    if not os.path.exists('evaluation_results/all_models_comparison.csv'):
        print("❌ Evaluation results not found.")
        return
    
    df = pd.read_csv('evaluation_results/all_models_comparison.csv')
    
    # Create summary for top models
    summary_data = []
    
    # Group by model and get best performance for each
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        best_config = model_data.loc[model_data['f1_weighted'].idxmax()]
        
        summary_data.append({
            'Model': model,
            'Best_Vectorizer': best_config['vectorizer'],
            'Accuracy': f"{best_config['accuracy']:.3f}",
            'F1_Score': f"{best_config['f1_weighted']:.3f}",
            'Precision': f"{best_config['precision_macro']:.3f}",
            'Recall': f"{best_config['recall_macro']:.3f}",
            'AUC': f"{best_config['auc_macro']:.3f}",
            'Training_Time': f"{best_config['training_time']:.3f}s",
            'Task': best_config['task']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('F1_Score', ascending=False)
    
    # Save as CSV and display
    summary_df.to_csv('evaluation_results/model_summary_table.csv', index=False)
    
    print("\n📋 PUBLICATION-READY SUMMARY TABLE")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    
    # Create LaTeX table format
    latex_table = summary_df.head(10).to_latex(index=False, escape=False)
    with open('evaluation_results/model_summary_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"\n💾 Summary table saved as:")
    print(f"   📊 CSV: evaluation_results/model_summary_table.csv")
    print(f"   📄 LaTeX: evaluation_results/model_summary_table.tex")
    
    return summary_df

def main():
    """Run complete analysis and visualization"""
    print("🎯 CREATING COMPREHENSIVE MODEL JUSTIFICATION PACKAGE")
    print("=" * 70)
    
    # Create visualizations
    print("\n1️⃣ Creating visualization charts...")
    fig = create_model_comparison_charts()
    
    # Statistical analysis
    print("\n2️⃣ Performing statistical analysis...")
    df = statistical_analysis()
    
    # Summary table
    print("\n3️⃣ Creating summary tables...")
    summary_df = create_summary_table()
    
    print(f"\n✅ COMPLETE JUSTIFICATION PACKAGE READY!")
    print(f"📁 All files saved in 'evaluation_results/' directory:")
    print(f"   📊 model_comparison_charts.png/pdf - Visual comparisons")
    print(f"   📋 model_summary_table.csv - Summary statistics")
    print(f"   📄 model_summary_table.tex - LaTeX table for papers")
    print(f"   📈 Statistical analysis printed above")
    
    print(f"\n🎓 FOR YOUR ACADEMIC GUIDE:")
    print(f"   📊 Show the visualization charts for visual proof")
    print(f"   📋 Present the summary table for detailed metrics")
    print(f"   📈 Reference statistical significance tests")
    print(f"   📄 Include LaTeX table in thesis/report")

if __name__ == "__main__":
    main()
