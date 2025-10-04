"""
Multilingual Mobile App Reviews – Aug 2025 Analysis (Enhanced with NLP)

This script performs comprehensive analysis of multilingual mobile app reviews including:
- Data exploration and visualization
- Text preprocessing and vectorization
- Advanced NLP techniques for improved accuracy
- Predictive modeling with multiple algorithms

Author: Benjamin
Date: October 2025
"""

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
import re

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Machine Learning and NLP libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Text processing libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available. Some text preprocessing features will be disabled.")
    NLTK_AVAILABLE = False

# Download NLTK data if needed
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

print("=" * 50)
print("MULTILINGUAL MOBILE APP REVIEWS ANALYSIS")
print("=" * 50)

# Load dataset
print("\n1. Loading dataset...")
df = pd.read_csv("./multilingual_mobile_app_reviews_2025.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Basic exploration
print("\n2. Basic Data Exploration")
print(f"Data types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")

# Display basic info
print("\nDataset Info:")
df.info()

print("\nDataset Description:")
print(df.describe())

# Data Visualizations
print("\n3. Creating visualizations...")

# Rating Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="rating")
plt.title("Rating Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.savefig('figures/rating_distribution.png', dpi=300, bbox_inches='tight')

# Top 10 Most Reviewed Apps
top_apps = df["app_name"].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_apps.values, y=top_apps.index)
plt.title("Top 10 Most Reviewed Apps", fontsize=14, fontweight='bold')
plt.xlabel("Number of Reviews")
plt.ylabel("App Name")
plt.savefig('figures/top_reviewed_apps.png', dpi=300, bbox_inches='tight')

# Ratings Distribution by App Category
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="app_category", y="rating")
plt.title("Ratings Distribution by App Category", fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.savefig('figures/ratings_by_category.png', dpi=300, bbox_inches='tight')

# Top 10 Review Languages
plt.figure(figsize=(10, 5))
lang_counts = df["review_language"].value_counts().head(10)
sns.barplot(x=lang_counts.index, y=lang_counts.values)
plt.title("Top 10 Review Languages", fontsize=14, fontweight='bold')
plt.xlabel("Language")
plt.ylabel("Number of Reviews")
plt.savefig('figures/top_languages.png', dpi=300, bbox_inches='tight')

# Ratings by Verified Purchase Status
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="verified_purchase", y="rating")
plt.title("Ratings by Verified Purchase Status", fontsize=14, fontweight='bold')
plt.savefig('figures/ratings_by_verified_purchase.png', dpi=300, bbox_inches='tight')

# Reviews by Device Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="device_type", order=df["device_type"].value_counts().index)
plt.title("Reviews by Device Type", fontsize=14, fontweight='bold')
plt.xlabel("Device Type")
plt.ylabel("Number of Reviews")
plt.savefig('figures/reviews_by_device_type.png', dpi=300, bbox_inches='tight')

# Helpful Votes Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["num_helpful_votes"], bins=20, kde=True)
plt.title("Helpful Votes Distribution", fontsize=14, fontweight='bold')
plt.xlabel("Number of Helpful Votes")
plt.ylabel("Frequency")
plt.savefig('figures/helpful_votes_distribution.png', dpi=300, bbox_inches='tight')

# Ratings by Top 10 User Countries
top_countries = df["user_country"].value_counts().head(10).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[df["user_country"].isin(top_countries)],
            x="user_country", y="rating")
plt.title("Ratings by Top 10 User Countries", fontsize=14, fontweight='bold')
plt.xlabel("Country")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.savefig('figures/ratings_by_country.png', dpi=300, bbox_inches='tight')

# Ratings by Gender
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x="user_gender", y="rating")
plt.title("Ratings by Gender", fontsize=14, fontweight='bold')
plt.savefig('figures/ratings_by_gender.png', dpi=300, bbox_inches='tight')

# Review Volume Over Time
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
df_time = df.groupby(df["review_date"].dt.to_period("M")).size()
df_time.index = df_time.index.to_timestamp()
plt.figure(figsize=(12, 6))
plt.plot(df_time.index, df_time.values)
plt.title("Review Volume Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.savefig('figures/review_volume_over_time.png', dpi=300, bbox_inches='tight')

print("\n4. Text Preprocessing and NLP Analysis")

# Define text preprocessing functions
def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters, numbers, and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def advanced_text_preprocessing(text):
    """Advanced text preprocessing with NLTK"""
    if not NLTK_AVAILABLE:
        return clean_text(text)
    
    text = clean_text(text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Apply text preprocessing to review text
print("Preprocessing review text...")
df['review_text_cleaned'] = df['review_text'].apply(clean_text)
if NLTK_AVAILABLE:
    df['review_text_advanced'] = df['review_text'].apply(advanced_text_preprocessing)
else:
    df['review_text_advanced'] = df['review_text_cleaned']

print("\n5. Enhanced Predictive Modeling with Text Vectorization")

# Prepare data for modeling
df_model = df.copy()

# Fill missing values
df_model.fillna("Unknown", inplace=True)

# Convert review_date to datetime and extract features
df_model["review_date"] = pd.to_datetime(df_model["review_date"], errors="coerce")
df_model["review_year"] = df_model["review_date"].dt.year.fillna(0).astype(int)
df_model["review_month"] = df_model["review_date"].dt.month.fillna(0).astype(int)
df_model["review_day"] = df_model["review_date"].dt.day.fillna(0).astype(int)
df_model.drop(columns=["review_date"], inplace=True)

# Define target variable
target = "verified_purchase"
text_feature = "review_text_advanced"

# Separate text and numerical features
text_data = df_model[text_feature]
numerical_features = df_model.drop(columns=[target, 'review_text', 'review_text_cleaned', text_feature])

# Encode categorical features
label_encoders = {}
for col in numerical_features.columns:
    if numerical_features[col].dtype == "object" or numerical_features[col].dtype == "bool":
        numerical_features[col] = numerical_features[col].astype(str)
        le = LabelEncoder()
        numerical_features[col] = le.fit_transform(numerical_features[col])
        label_encoders[col] = le

# Encode target variable
y = df_model[target]
if y.dtype == "object" or y.dtype == "bool":
    y = y.astype(str)
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

print("\n6. Text Vectorization Techniques")

# Different vectorization approaches
vectorizers = {
    'TF-IDF (1-2 grams)': TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
    'TF-IDF (1-3 grams)': TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english'),
    'Count Vectorizer': CountVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english'),
    'TF-IDF + SVD': Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
        ('svd', TruncatedSVD(n_components=100, random_state=42))
    ])
}

results = {}

for vec_name, vectorizer in vectorizers.items():
    print(f"\nTesting {vec_name}...")
    
    # Vectorize text
    X_text = vectorizer.fit_transform(text_data)
    
    # Combine with numerical features
    if hasattr(X_text, 'toarray'):
        X_text = X_text.toarray()
    
    # Combine features
    X_combined = np.hstack([X_text, numerical_features.values])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models for this vectorization approach
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "Naive Bayes": GaussianNB()
    }
    
    vec_results = {}
    
    for name, model in models.items():
        try:
            if name == "Naive Bayes":
                # Use unscaled data for Naive Bayes
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test, y_pred) * 100
            vec_results[name] = accuracy
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            vec_results[name] = 0
    
    results[vec_name] = vec_results

# Print results
print("\n" + "="*80)
print("RESULTS SUMMARY - Text Vectorization Impact on Model Accuracy")
print("="*80)

for vec_name, vec_results in results.items():
    print(f"\n{vec_name}:")
    print("-" * len(vec_name))
    for model, acc in vec_results.items():
        print(f"{model:20}: {acc:.2f}%")

# Find best combination
best_combo = None
best_acc = 0
for vec_name, vec_results in results.items():
    for model, acc in vec_results.items():
        if acc > best_acc:
            best_acc = acc
            best_combo = (vec_name, model)

print(f"\nBest combination: {best_combo[1]} with {best_combo[0]} - {best_acc:.2f}%")

# Visualization of results
print("\n7. Creating comparison visualizations...")

# Create comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, (vec_name, vec_results) in enumerate(results.items()):
    if i < 4:  # Only plot first 4 vectorizers
        ax = axes[i]
        models = list(vec_results.keys())
        accuracies = list(vec_results.values())
        
        bars = ax.bar(models, accuracies, 
                     color=['darkblue', 'skyblue', 'darkgreen', 'orange', 'purple'])
        ax.set_title(f'{vec_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figures/vectorization_comparison.png', dpi=300, bbox_inches='tight')

# Overall comparison
all_results = []
for vec_name, vec_results in results.items():
    for model, acc in vec_results.items():
        all_results.append({
            'Vectorizer': vec_name,
            'Model': model,
            'Accuracy': acc
        })

results_df = pd.DataFrame(all_results)

# Heatmap of results
pivot_results = results_df.pivot(index='Model', columns='Vectorizer', values='Accuracy')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_results, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy (%)'})
plt.title('Model Performance Heatmap by Vectorization Technique', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/performance_heatmap.png', dpi=300, bbox_inches='tight')

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nKey Insights:")
print("1. Text vectorization significantly impacts model performance")
print("2. TF-IDF with n-grams generally performs better than simple bag-of-words")
print("3. Dimensionality reduction (SVD) can help with computational efficiency")
print("4. Different models respond differently to vectorization techniques")
print("5. Consider ensemble methods for optimal performance")

print(f"\nAll visualizations saved to 'figures/' directory")
print("Recommended improvements for higher accuracy:")
print("- Use pre-trained embeddings (Word2Vec, GloVe, BERT)")
print("- Implement language-specific preprocessing")
print("- Use deep learning models (LSTM, Transformer)")
print("- Apply feature engineering on metadata")
print("- Implement cross-validation for robust evaluation")