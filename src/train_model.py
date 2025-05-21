import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_kaggle_dataset():
    """
    Load and preprocess the Sleep Health and Lifestyle Dataset
    """
    try:
        # Load the dataset
        data = pd.read_csv('../data/Sleep_health_and_lifestyle_dataset.csv')
        
        # Handle outliers for numeric features
        for col in ['Sleep Duration', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Age']:
            lower = data[col].quantile(0.01)
            upper = data[col].quantile(0.99)
            data[col] = data[col].clip(lower, upper)
        
        # Use separate encoders for Gender and BMI Category
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
        
        # Select features for prediction
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
        y = data['Quality of Sleep']  # Use the real column as target
        
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
    
    # Split into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    
    # Print class distribution in train and test sets
    print("\nClass distribution in training set:")
    print(y_train.value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(y_test.value_counts(normalize=True))
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model tuning with GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced']  # Added balanced class weights
    }
    
    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Perform multiple random splits for validation
    n_splits = 3  # Reduced from 5 to 3 due to class imbalance
    best_scores = []
    best_params_list = []
    
    for i in range(n_splits):
        print(f"\nValidation Split {i+1}/{n_splits}")
        # Create a new train-test split for each iteration
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            X, y, 
            test_size=0.3,
            random_state=i,
            stratify=y
        )
        
        # Scale features for this split
        scaler_i = StandardScaler()
        X_train_scaled_i = scaler_i.fit_transform(X_train_i)
        X_test_scaled_i = scaler_i.transform(X_test_i)
        
        # Grid search for this split
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring='f1_weighted',  # Changed to weighted F1 score
            n_jobs=-1
        )
        grid.fit(X_train_scaled_i, y_train_i)
        
        # Store results
        best_scores.append(grid.best_score_)
        best_params_list.append(grid.best_params_)
        
        # Evaluate on test set
        model_i = grid.best_estimator_
        test_pred_i = model_i.predict(X_test_scaled_i)
        test_acc_i = accuracy_score(y_test_i, test_pred_i)
        test_f1_i = f1_score(y_test_i, test_pred_i, average='weighted')
        
        print(f"Best params for split {i+1}:", grid.best_params_)
        print(f"Best CV score for split {i+1}:", grid.best_score_)
        print(f"Test accuracy for split {i+1}:", test_acc_i)
        print(f"Test F1 score for split {i+1}:", test_f1_i)
        print("\nClassification Report (Test):\n", classification_report(y_test_i, test_pred_i))
        
        # Plot confusion matrix for this split
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test_i, test_pred_i)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Split {i+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'../static/confusion_matrix_split_{i+1}.png')
        plt.close()
    
    # Print summary of all splits
    print("\nSummary of all validation splits:")
    print("Best CV scores:", best_scores)
    print("Mean CV score:", np.mean(best_scores))
    print("Std CV score:", np.std(best_scores))
    
    # Train final model on full training set with best parameters
    best_params = max(best_params_list, key=lambda x: best_scores[best_params_list.index(x)])
    print("\nFinal best parameters:", best_params)
    
    final_model = RandomForestClassifier(**best_params, random_state=42)
    final_model.fit(X_train_scaled, y_train)
    
    # Evaluate final model
    train_pred = final_model.predict(X_train_scaled)
    test_pred = final_model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_f1 = f1_score(y_train, train_pred, average='weighted')
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    print(f"\nFinal Model Performance:")
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Testing Accuracy: {test_acc:.3f}")
    print(f"Training F1 Score: {train_f1:.3f}")
    print(f"Testing F1 Score: {test_f1:.3f}")
    print("\nClassification Report (Test):\n", classification_report(y_test, test_pred))
    
    # Plot confusion matrix for final model
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Final Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('../static/confusion_matrix_final.png')
    plt.close()
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    importances = final_model.feature_importances_
    feat_names = X.columns
    sorted_idx = np.argsort(importances)[::-1]
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), [feat_names[i] for i in sorted_idx], rotation=45, ha='right')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.savefig('../static/feature_importances.png')
    plt.close()
    
    # Cross-validation for robust accuracy estimate
    scores = cross_val_score(final_model, scaler.transform(X), y, cv=cv, scoring='f1_weighted')
    print(f"\n3-Fold Cross-Validation F1 scores: {scores}")
    print(f"Mean CV F1 Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    
    # Save model, scaler, and all encoders
    joblib.dump(final_model, '../models/sleep_model.joblib')
    joblib.dump(scaler, '../models/scaler.joblib')
    joblib.dump(gender_encoder, '../models/gender_encoder.joblib')
    joblib.dump(bmi_encoder, '../models/bmi_encoder.joblib')
    joblib.dump(agegroup_encoder, '../models/agegroup_encoder.joblib')
    joblib.dump(stepsgroup_encoder, '../models/stepsgroup_encoder.joblib')
    
    print("\nModel, scaler, and encoders saved successfully!")

if __name__ == "__main__":
    train_model() 