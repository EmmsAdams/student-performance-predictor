"""
Student Performance Predictor
A machine learning project demonstrating complete ML workflow
Developer: Emmelyn Adams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_explore_data():
    """Load and explore the dataset"""
    print("="*60)
    print("STUDENT PERFORMANCE PREDICTION")
    print("Machine Learning Project by Emmelyn Adams")
    print("="*60)
    print("\n📊 Loading dataset...")
    
    # Generate sample dataset (in real scenario, load from CSV)
    # This demonstrates the complete workflow
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'study_hours': np.random.randint(0, 40, n_samples),
        'attendance_rate': np.random.randint(50, 100, n_samples),
        'previous_grade': np.random.randint(40, 100, n_samples),
        'assignments_completed': np.random.randint(0, 20, n_samples),
        'class_participation': np.random.randint(1, 6, n_samples),
        'sleep_hours': np.random.randint(4, 10, n_samples),
        'extracurricular': np.random.randint(0, 3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable (Pass/Fail) based on features
    # Pass if weighted combination of factors is above threshold
    df['performance_score'] = (
        df['study_hours'] * 0.3 +
        df['attendance_rate'] * 0.25 +
        df['previous_grade'] * 0.2 +
        df['assignments_completed'] * 2 +
        df['class_participation'] * 3 +
        df['sleep_hours'] * 1.5
    )
    
    # Add some randomness
    df['performance_score'] += np.random.normal(0, 15, n_samples)
    
    # Create binary outcome
    df['outcome'] = (df['performance_score'] > df['performance_score'].median()).astype(int)
    df['outcome_label'] = df['outcome'].map({1: 'Pass', 0: 'Fail'})
    
    # Drop intermediate score
    df = df.drop('performance_score', axis=1)
    
    print("✅ Dataset loaded successfully!")
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 2}")
    print(f"Samples: {df.shape[0]}")
    
    return df

def explore_data(df):
    """Perform exploratory data analysis"""
    print("\n" + "="*60)
    print("📈 EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print("\n🔍 Dataset Overview:")
    print(df.head())
    
    print("\n📊 Statistical Summary:")
    print(df.describe())
    
    print("\n🎯 Class Distribution:")
    print(df['outcome_label'].value_counts())
    print(f"\nClass balance: {df['outcome'].value_counts(normalize=True).round(3)}")
    
    # Check for missing values
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found")
    else:
        print(missing[missing > 0])
    
    return df

def preprocess_data(df):
    """Preprocess data for machine learning"""
    print("\n" + "="*60)
    print("⚙️ DATA PREPROCESSING")
    print("="*60)
    
    # Separate features and target
    feature_columns = ['study_hours', 'attendance_rate', 'previous_grade', 
                       'assignments_completed', 'class_participation', 
                       'sleep_hours', 'extracurricular']
    
    X = df[feature_columns]
    y = df['outcome']
    
    print(f"\n✅ Features selected: {len(feature_columns)}")
    print(f"Features: {', '.join(feature_columns)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns

def train_models(X_train, y_train):
    """Train multiple ML models"""
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING")
    print("="*60)
    
    models = {}
    
    # Random Forest
    print("\n🌲 Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    print("✅ Random Forest trained")
    
    # Logistic Regression
    print("\n📈 Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    print("✅ Logistic Regression trained")
    
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"📈 {name} Results")
        print(f"{'='*60}")
        
        # Training accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        # Testing accuracy
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n✅ Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"✅ Testing Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, test_pred, 
                                     target_names=['Fail', 'Pass']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'predictions': test_pred,
            'confusion_matrix': cm
        }
    
    return results

def visualize_results(results, y_test, feature_columns, models):
    """Create visualizations of results"""
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Model Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Student Performance Prediction - Model Evaluation', 
                 fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    model_names = list(results.keys())
    train_accs = [results[m]['train_accuracy'] for m in model_names]
    test_accs = [results[m]['test_accuracy'] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_accs, width, label='Training', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_accs, width, label='Testing', alpha=0.8)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=15)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # Confusion matrices
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx, 1] if idx == 0 else axes[1, 0]
        cm = result['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Fail', 'Pass'],
                    yticklabels=['Fail', 'Pass'])
        ax.set_title(f'{name}\nConfusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[1, 1].barh(range(len(importances)), importances[indices], alpha=0.8)
        axes[1, 1].set_yticks(range(len(importances)))
        axes[1, 1].set_yticklabels([feature_columns[i] for i in indices])
        axes[1, 1].set_xlabel('Importance')
        axes[1, 1].set_title('Random Forest\nFeature Importance')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved as 'model_evaluation_results.png'")
    
    plt.show()

def print_summary(results):
    """Print final summary"""
    print("\n" + "="*60)
    print("🎯 FINAL SUMMARY")
    print("="*60)
    
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    
    print(f"\n🏆 Best Performing Model: {best_model[0]}")
    print(f"   Test Accuracy: {best_model[1]['test_accuracy']:.4f} ({best_model[1]['test_accuracy']*100:.2f}%)")
    
    print("\n📊 All Model Results:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Training Accuracy: {result['train_accuracy']:.4f}")
        print(f"  Testing Accuracy:  {result['test_accuracy']:.4f}")
        print(f"  Difference:        {abs(result['train_accuracy'] - result['test_accuracy']):.4f}")
    
    print("\n" + "="*60)
    print("✅ Analysis Complete!")
    print("="*60)
    
    print("\n💡 Key Learnings from this project:")
    print("   • Implemented complete ML pipeline from data to evaluation")
    print("   • Compared multiple classification algorithms")
    print("   • Used proper train-test split for validation")
    print("   • Evaluated models using multiple metrics")
    print("   • Created comprehensive visualizations")
    
    print("\n📚 Technologies Used:")
    print("   • pandas: Data manipulation")
    print("   • scikit-learn: Machine learning algorithms")
    print("   • matplotlib & seaborn: Data visualization")
    print("   • NumPy: Numerical computations")

def main():
    """Main execution function"""
    
    # 1. Load and explore data
    df = load_and_explore_data()
    df = explore_data(df)
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, feature_columns = preprocess_data(df)
    
    # 3. Train models
    models = train_models(X_train, y_train)
    
    # 4. Evaluate models
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # 5. Visualize results
    visualize_results(results, y_test, feature_columns, models)
    
    # 6. Print summary
    print_summary(results)
    
    print("\n" + "="*60)
    print("Project by Emmelyn Adams")
    print("Computer Science Student, Northumbria University")
    print("Demonstrating ML fundamentals and data science workflow")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
