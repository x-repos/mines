"""
Comprehensive Model Comparison with Heatmap Visualization
Tests multiple models and creates accuracy heatmap for comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("="*70)
print("COMPREHENSIVE MODEL COMPARISON WITH HEATMAP VISUALIZATION")
print("="*70)

# 1. Load and prepare data
print("\n1. Loading and preparing data...")
df = pd.read_csv("./multilingual_mobile_app_reviews_2025.csv")
print(f"Dataset shape: {df.shape}")

# Data preprocessing function
def prepare_data(df, include_text=False):
    """Prepare data with option to include text features"""
    df_prep = df.copy()
    df_prep.fillna("Unknown", inplace=True)
    
    # Date processing
    df_prep["review_date"] = pd.to_datetime(df_prep["review_date"], errors="coerce")
    df_prep["review_year"] = df_prep["review_date"].dt.year.fillna(0).astype(int)
    df_prep["review_month"] = df_prep["review_date"].dt.month.fillna(0).astype(int)
    df_prep["review_day"] = df_prep["review_date"].dt.day.fillna(0).astype(int)
    df_prep.drop(columns=["review_date"], inplace=True)
    
    # Separate features and target
    target = "verified_purchase"
    
    if include_text:
        # Clean text
        def clean_text(text):
            if pd.isna(text):
                return ""
            import re
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        df_prep['review_text_cleaned'] = df_prep['review_text'].apply(clean_text)
        
        # Extract text features
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
        text_features = tfidf.fit_transform(df_prep['review_text_cleaned']).toarray()
        
        # Numerical features
        X_num = df_prep.drop(columns=[target, 'review_text', 'review_text_cleaned'])
        
        # Encode categorical features
        for col in X_num.columns:
            if X_num[col].dtype == "object" or X_num[col].dtype == "bool":
                X_num[col] = X_num[col].astype(str)
                le = LabelEncoder()
                X_num[col] = le.fit_transform(X_num[col])
        
        # Combine text and numerical features
        X = np.hstack([text_features, X_num.values])
    else:
        # Only numerical features
        X = df_prep.drop(columns=[target])
        if 'review_text' in X.columns:
            X = X.drop(columns=['review_text'])
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype == "bool":
                X[col] = X[col].astype(str)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
    
    # Encode target
    y = df_prep[target]
    if y.dtype == "object" or y.dtype == "bool":
        y = y.astype(str)
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
    
    return X, y

# 2. Define models to test
print("\n2. Defining models for comparison...")

# Basic models
basic_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF)": SVC(kernel='rbf', random_state=42),
    "SVM (Linear)": SVC(kernel='linear', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Advanced models
advanced_models = {
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "Linear DA": LinearDiscriminantAnalysis(),
    "Quadratic DA": QuadraticDiscriminantAnalysis(),
    "Ridge Classifier": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42, max_iter=1000),
    "Elastic Net": ElasticNet(random_state=42, max_iter=1000),
}

# Optimized models
optimized_models = {
    "Optimized LR": LogisticRegression(max_iter=2000, C=0.1, solver='liblinear', random_state=42),
    "Optimized RF": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
    "Optimized GB": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42),
    "Optimized SVM": SVC(C=0.1, kernel='rbf', random_state=42),
    "Optimized KNN": KNeighborsClassifier(n_neighbors=7, weights='distance'),
}

# 3. Test different data configurations
print("\n3. Testing different data configurations...")

data_configs = {
    "Numerical Only": (False, None),
    "With Text Features": (True, None),
    "Scaled (Standard)": (False, StandardScaler()),
    "Scaled (Robust)": (False, RobustScaler()),
    "Scaled (MinMax)": (False, MinMaxScaler()),
}

# Store all results
all_results = {}

for config_name, (include_text, scaler) in data_configs.items():
    print(f"\nTesting configuration: {config_name}")
    
    # Prepare data
    X, y = prepare_data(df, include_text=include_text)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply scaling if specified
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Test all model categories
    all_models = {**basic_models, **advanced_models, **optimized_models}
    
    config_results = {}
    
    for model_name, model in all_models.items():
        try:
            # Handle different model types
            if hasattr(model, 'predict_proba') or model_name in ['Ridge Classifier', 'Lasso', 'Elastic Net']:
                if model_name in ['Ridge Classifier', 'Lasso', 'Elastic Net']:
                    # These are regression models, need classification wrapper
                    from sklearn.linear_model import RidgeClassifier, LogisticRegression
                    if model_name == 'Ridge Classifier':
                        model = RidgeClassifier(random_state=42)
                    else:
                        # Skip pure regression models for classification
                        continue
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred) * 100
                config_results[model_name] = accuracy
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred) * 100
                config_results[model_name] = accuracy
                
        except Exception as e:
            print(f"  Error with {model_name}: {str(e)[:50]}...")
            config_results[model_name] = 0.0
    
    all_results[config_name] = config_results
    
    # Print top 3 for this configuration
    top_3 = sorted(config_results.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3 for {config_name}:")
    for model, acc in top_3:
        print(f"    {model}: {acc:.2f}%")

# 4. Create comprehensive results DataFrame
print("\n4. Creating comprehensive results matrix...")

# Create DataFrame for heatmap
results_df = pd.DataFrame(all_results).fillna(0)

# Sort by best overall performance
results_df['Mean_Accuracy'] = results_df.mean(axis=1)
results_df = results_df.sort_values('Mean_Accuracy', ascending=False)
results_df = results_df.drop('Mean_Accuracy', axis=1)

print("Results matrix created with shape:", results_df.shape)

# 5. Create visualizations
print("\n5. Creating visualizations...")

# Set up the plot style
plt.style.use('default')
sns.set_palette("viridis")

# Create main heatmap
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Main heatmap (all models and configurations)
ax1 = axes[0, 0]
sns.heatmap(results_df, annot=True, fmt='.1f', cmap='RdYlBu_r', 
            center=75, vmin=50, vmax=85, ax=ax1, cbar_kws={'label': 'Accuracy (%)'})
ax1.set_title('Model Performance Heatmap - All Configurations', fontsize=14, fontweight='bold')
ax1.set_xlabel('Data Configuration')
ax1.set_ylabel('Model')

# Top 10 models heatmap
ax2 = axes[0, 1]
top_10_models = results_df.head(10)
sns.heatmap(top_10_models, annot=True, fmt='.1f', cmap='RdYlBu_r', 
            center=75, vmin=70, vmax=85, ax=ax2, cbar_kws={'label': 'Accuracy (%)'})
ax2.set_title('Top 10 Models Performance', fontsize=14, fontweight='bold')
ax2.set_xlabel('Data Configuration')
ax2.set_ylabel('Model')

# Best configuration comparison
ax3 = axes[1, 0]
config_means = results_df.mean().sort_values(ascending=False)
config_means.plot(kind='bar', ax=ax3, color='skyblue', edgecolor='navy')
ax3.set_title('Average Performance by Configuration', fontsize=14, fontweight='bold')
ax3.set_xlabel('Configuration')
ax3.set_ylabel('Average Accuracy (%)')
ax3.tick_params(axis='x', rotation=45)

# Best models comparison
ax4 = axes[1, 1]
model_means = results_df.mean(axis=1).head(10)
model_means.plot(kind='barh', ax=ax4, color='lightcoral', edgecolor='darkred')
ax4.set_title('Top 10 Models by Average Performance', fontsize=14, fontweight='bold')
ax4.set_xlabel('Average Accuracy (%)')
ax4.set_ylabel('Model')

plt.tight_layout()
plt.savefig('figures/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Create detailed comparison table
print("\n6. Detailed Results Summary")
print("="*70)

print(f"\nBest performing models across all configurations:")
print("-" * 50)
for i, (model, row) in enumerate(results_df.head(10).iterrows(), 1):
    avg_acc = row.mean()
    best_config = row.idxmax()
    best_acc = row.max()
    print(f"{i:2d}. {model:25} | Avg: {avg_acc:.2f}% | Best: {best_acc:.2f}% ({best_config})")

print(f"\nBest configuration for each data setup:")
print("-" * 50)
for config in results_df.columns:
    best_model = results_df[config].idxmax()
    best_acc = results_df[config].max()
    print(f"{config:20}: {best_model:25} ({best_acc:.2f}%)")

# Overall best result
overall_best_acc = results_df.max().max()
best_config = results_df.max().idxmax()
best_model_series = results_df[best_config]
best_model = best_model_series.idxmax()

print(f"\n🏆 OVERALL BEST RESULT:")
print(f"   Model: {best_model}")
print(f"   Configuration: {best_config}")
print(f"   Accuracy: {overall_best_acc:.2f}%")

# Save detailed results to CSV
results_df.to_csv('figures/model_comparison_results.csv')
print(f"\n📄 Detailed results saved to 'figures/model_comparison_results.csv'")
print(f"📊 Visualization saved to 'figures/comprehensive_model_comparison.png'")

print(f"\n✅ Analysis complete! Check the figures folder for detailed visualizations.")