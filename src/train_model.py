import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def clean_dataset(data):
    """
    Clean the dataset by removing duplicates and adding realistic noise
    """
    # Remove exact duplicates
    data = data.drop_duplicates()
    
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
    
    return data

def load_kaggle_dataset():
    """
    Load and preprocess the Sleep Health and Lifestyle Dataset
    """
    try:
        # Load the dataset
        data = pd.read_csv('../data/Sleep_health_and_lifestyle_dataset.csv')
        
        # Clean the dataset
        data = clean_dataset(data)
        
        # Handle outliers for numeric features
        for col in ['Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Age']:
            lower = data[col].quantile(0.01)
            upper = data[col].quantile(0.99)
            data[col] = data[col].clip(lower, upper)
        
        # Use separate encoders for categorical variables
        gender_encoder = LabelEncoder()
        bmi_encoder = LabelEncoder()
        data['Gender'] = gender_encoder.fit_transform(data['Gender'])
        data['BMI Category'] = bmi_encoder.fit_transform(data['BMI Category'])
        
        # Feature engineering: bin Age and Daily Steps
        data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['<30', '30-40', '40-50', '50-60', '60+'])
        data['StepsGroup'] = pd.cut(data['Daily Steps'], bins=[0, 4000, 7000, 10000, 20000], labels=['Low', 'Medium', 'High', 'Very High'])
        agegroup_encoder = LabelEncoder()
        stepsgroup_encoder = LabelEncoder()
        data['AgeGroup'] = agegroup_encoder.fit_transform(data['AgeGroup'].astype(str))
        data['StepsGroup'] = stepsgroup_encoder.fit_transform(data['StepsGroup'].astype(str))
        
        # Select features for prediction (removed potentially leaking features)
        features = [
            'Sleep Duration',
            'Physical Activity Level',
            'Stress Level',
            'BMI Category',
            'Heart Rate',
            'Daily Steps',
            'Age',
            'Gender',
            'AgeGroup',
            'StepsGroup'
        ]
        
        X = data[features]
        y = data['Quality of Sleep']
        
        return X, y, gender_encoder, bmi_encoder, agegroup_encoder, stepsgroup_encoder
        
    except FileNotFoundError:
        print("Dataset file not found. Please make sure 'Sleep_health_and_lifestyle_dataset.csv' is in the project directory.")
        return None, None, None, None, None, None

def train_model():
    # Load and preprocess data
    X, y, gender_encoder, bmi_encoder, agegroup_encoder, stepsgroup_encoder = load_kaggle_dataset()
    
    if X is None or y is None:
        return
    
    # Print class distribution in the full dataset
    print("\nClass distribution in full dataset:")
    print(y.value_counts(normalize=True))
    
    # Create a more rigorous train/validation/test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of total data
        random_state=42,
        stratify=y_temp
    )
    
    print("\nData split sizes:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # First try a simple logistic regression model
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    
    # Evaluate logistic regression
    lr_val_pred = lr_model.predict(X_val_scaled)
    lr_val_acc = accuracy_score(y_val, lr_val_pred)
    lr_val_f1 = f1_score(y_val, lr_val_pred, average='weighted')
    
    print("\nLogistic Regression Performance (Validation):")
    print(f"Accuracy: {lr_val_acc:.3f}")
    print(f"F1 Score: {lr_val_f1:.3f}")
    print("\nClassification Report (Validation):\n", classification_report(y_val, lr_val_pred))
    
    # Now train Random Forest with reduced complexity
    print("\nTraining Random Forest model...")
    param_grid = {
        'n_estimators': [50, 100],  # Reduced number of trees
        'max_depth': [3, 5, 7],     # Limited depth to prevent overfitting
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'class_weight': ['balanced']
    }
    
    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search with cross-validation
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    
    # Get best model
    best_model = grid.best_estimator_
    print("\nBest parameters:", grid.best_params_)
    
    # Evaluate on validation set
    val_pred = best_model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average='weighted')
    
    print("\nRandom Forest Performance (Validation):")
    print(f"Accuracy: {val_acc:.3f}")
    print(f"F1 Score: {val_f1:.3f}")
    print("\nClassification Report (Validation):\n", classification_report(y_val, val_pred))
    
    # Final evaluation on test set
    test_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    print("\nFinal Model Performance (Test):")
    print(f"Accuracy: {test_acc:.3f}")
    print(f"F1 Score: {test_f1:.3f}")
    print("\nClassification Report (Test):\n", classification_report(y_test, test_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../static/confusion_matrix_final.png')
    plt.close()
    
    # Feature importance analysis
    plt.figure(figsize=(12, 6))
    importances = best_model.feature_importances_
    feat_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [feat_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('../static/feature_importances.png')
    plt.close()
    
    # SHAP values for better feature importance interpretation
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_scaled, feature_names=feat_names, plot_type="bar")
    plt.title('Feature Importances (SHAP)')
    plt.tight_layout()
    plt.savefig('../static/feature_importances_shap.png')
    plt.close()
    
    # Save models and encoders
    joblib.dump(best_model, '../models/sleep_model.joblib')
    joblib.dump(lr_model, '../models/sleep_model_lr.joblib')
    joblib.dump(scaler, '../models/scaler.joblib')
    joblib.dump(gender_encoder, '../models/gender_encoder.joblib')
    joblib.dump(bmi_encoder, '../models/bmi_encoder.joblib')
    joblib.dump(agegroup_encoder, '../models/agegroup_encoder.joblib')
    joblib.dump(stepsgroup_encoder, '../models/stepsgroup_encoder.joblib')
    
    print("\nModels, scaler, and encoders saved successfully!")

if __name__ == "__main__":
    train_model() 