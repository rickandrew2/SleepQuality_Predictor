import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')

def clean_dataset(data):
    """
    Clean the dataset by removing duplicates, handling missing values, and adding realistic noise
    """
    # Remove exact duplicates
    data = data.drop_duplicates()
    
    # Handle missing values in Sleep Disorder column
    data['Sleep Disorder'] = data['Sleep Disorder'].fillna('None')
    
    # Add small random noise to numeric columns to make data more realistic
    numeric_cols = ['Sleep Duration', 'Physical Activity Level', 'Stress Level', 
                   'Heart Rate', 'Daily Steps', 'Age']
    
    for col in numeric_cols:
        # Add noise proportional to the standard deviation of each column
        noise = np.random.normal(0, data[col].std() * 0.05, size=len(data))
        data[col] = data[col] + noise
    
    # Round numeric columns to maintain realistic precision
    data['Sleep Duration'] = data['Sleep Duration'].round(1)
    data['Physical Activity Level'] = data['Physical Activity Level'].round(0)
    data['Stress Level'] = data['Stress Level'].round(0)
    data['Heart Rate'] = data['Heart Rate'].round(0)
    data['Daily Steps'] = data['Daily Steps'].round(0)
    
    # Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    
    return data

def engineer_features(data):
    """
    Create new features to improve model performance
    """
    # Create age groups
    data['Age_Group'] = pd.cut(data['Age'], 
                              bins=[0, 30, 40, 50, 60, 100],
                              labels=['<30', '30-40', '40-50', '50-60', '60+'])
    
    # Create sleep quality score
    data['Sleep_Quality_Score'] = data['Sleep Duration'] * data['Quality of Sleep'] / 10
    
    # Create stress-sleep interaction
    data['Stress_Sleep_Interaction'] = data['Stress Level'] * data['Sleep Duration']
    
    # Create physical activity categories
    data['Activity_Level'] = pd.cut(data['Physical Activity Level'],
                                  bins=[0, 30, 50, 70, 100],
                                  labels=['Low', 'Moderate', 'High', 'Very High'])
    
    # Create heart rate zones
    data['Heart_Rate_Zone'] = pd.cut(data['Heart Rate'],
                                    bins=[0, 60, 70, 80, 90, 200],
                                    labels=['Very Low', 'Low', 'Normal', 'High', 'Very High'])
    
    return data

def load_kaggle_dataset():
    """
    Load the Kaggle dataset
    """
    data = pd.read_csv('../data/Sleep_health_and_lifestyle_dataset.csv')
    return data

def show_feature_importance(rf_model, feature_names, top_n=10):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    print("\nTop Feature Importances (Random Forest):")
    for i in indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")
    # Plot
    plt.figure(figsize=(8, 5))
    plt.title("Top Feature Importances (Random Forest)")
    plt.barh([feature_names[i] for i in indices[::-1]], importances[indices[::-1]])
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

# Function to tune probability thresholds for each class
def tune_thresholds(model, X_test, y_test, le, thresholds=None):
    from sklearn.metrics import classification_report
    proba = model.predict_proba(X_test)
    n_classes = proba.shape[1]
    if thresholds is None:
        thresholds = [0.5] * n_classes
    preds = np.zeros_like(y_test)
    for i, row in enumerate(proba):
        # Assign class with highest probability above its threshold, else default to argmax
        above = [j for j, p in enumerate(row) if p >= thresholds[j]]
        if above:
            preds[i] = above[np.argmax([row[j] for j in above])]
        else:
            preds[i] = np.argmax(row)
    print("\nClassification Report with Custom Thresholds:")
    print(classification_report(y_test, preds, target_names=le.classes_))
    return preds

def plot_roc_curves(models, X_test, y_test, le):
    """
    Plot ROC curves for each class and model
    """
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r']
    
    for i, (name, model) in enumerate(models.items()):
        y_score = model.predict_proba(X_test)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for j in range(len(le.classes_)):
            fpr[j], tpr[j], _ = roc_curve(y_test == j, y_score[:, j])
            roc_auc[j] = auc(fpr[j], tpr[j])
            
            plt.plot(fpr[j], tpr[j], color=colors[i], alpha=0.3,
                    label=f'{name} - {le.classes_[j]} (AUC = {roc_auc[j]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curves(models, X_test, y_test, le):
    """
    Plot Precision-Recall curves for each class and model
    """
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r']
    
    for i, (name, model) in enumerate(models.items()):
        y_score = model.predict_proba(X_test)
        
        # Compute Precision-Recall curve for each class
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        for j in range(len(le.classes_)):
            precision[j], recall[j], _ = precision_recall_curve(y_test == j, y_score[:, j])
            avg_precision[j] = average_precision_score(y_test == j, y_score[:, j])
            
            plt.plot(recall[j], precision[j], color=colors[i], alpha=0.3,
                    label=f'{name} - {le.classes_[j]} (AP = {avg_precision[j]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Class')
    plt.legend(loc="lower left")
    plt.show()

def train_model():
    """
    Train and evaluate the models with improved handling of class imbalance
    """
    # Load and preprocess data
    data = load_kaggle_dataset()
    
    # Clean the dataset
    data = clean_dataset(data)
    
    # Engineer new features
    data = engineer_features(data)
    
    # Prepare features and target
    X = data.drop(['Sleep Disorder', 'Person ID'], axis=1)  # Remove Person ID
    y = data['Sleep Disorder']
    
    # Encode categorical variables in X
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Encode target variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Split the data with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Calculate class weights
    class_weights = dict(zip(np.unique(y_train), 
                           len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))))
    
    # Define base models
    rf_model = RandomForestClassifier(class_weight=class_weights, random_state=42)
    
    # Define parameter grids
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Train Random Forest with GridSearchCV
    print("\nTraining Random Forest...")
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    rf_grid.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    results = {}
    models = {
        'Random Forest': rf_grid.best_estimator_
    }
    
    for name, model in models.items():
        pred = model.predict(X_test_scaled)
        results[name] = {
            'accuracy': accuracy_score(y_test, pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, pred),
            'f1': f1_score(y_test, pred, average='weighted'),
            'classification_report': classification_report(y_test, pred, target_names=le.classes_),
            'confusion_matrix': confusion_matrix(y_test, pred)
        }
    
    # Plot ROC curves
    plot_roc_curves(models, X_test_scaled, y_test, le)
    
    # Plot Precision-Recall curves
    plot_precision_recall_curves(models, X_test_scaled, y_test, le)
    
    # Show feature importance for Random Forest
    show_feature_importance(rf_grid.best_estimator_, X.columns)
    
    return results, models, le, scaler

def plot_results(results):
    """
    Plot model performance metrics
    """
    # Create directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot accuracy and F1 score
    metrics = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'F1 Score': [r['f1'] for r in results.values()]
    })
    
    plt.figure(figsize=(10, 6))
    metrics.set_index('Model').plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/model_performance.png')
    plt.close()
    
    # Plot confusion matrices
    for name, result in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train model and get results
    results, models, le, scaler = train_model()
    
    # Print results
    for name, result in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {result['accuracy']:.3f}")
        print(f"F1 Score: {result['f1']:.3f}")
        print("\nClassification Report:")
        print(result['classification_report'])

    # Save models
    joblib.dump(models['Random Forest'], '../models/random_forest_model.joblib')
    joblib.dump(scaler, '../models/scaler.joblib')
    joblib.dump(le, '../models/label_encoder.joblib')

    # Plot results
    plot_results(results) 