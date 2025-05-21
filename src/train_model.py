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
import argparse
from sklearn.cluster import KMeans

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
    
    # --- New interaction features ---
    data['HRxActivity'] = data['Heart Rate'] * data['Physical Activity Level']
    data['StepsxAge'] = data['Daily Steps'] * data['Age']
    data['StressxHR'] = data['Stress Level'] * data['Heart Rate']
    data['ActivityxSleep'] = data['Physical Activity Level'] * data['Sleep Duration']
    
    # --- KMeans cluster feature ---
    cluster_features = data[[
        'Sleep Duration', 'Physical Activity Level', 'Stress Level',
        'Heart Rate', 'Daily Steps', 'Age']].copy()
    cluster_features = cluster_features.fillna(cluster_features.mean())
    cluster_scaler = StandardScaler()
    cluster_scaled = cluster_scaler.fit_transform(cluster_features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['Behavior_Cluster'] = kmeans.fit_predict(cluster_scaled)
    # Save the scaler and kmeans for use in the Flask app
    joblib.dump(cluster_scaler, os.path.join('models', 'cluster_scaler.joblib'))
    joblib.dump(kmeans, os.path.join('models', 'cluster_kmeans.joblib'))
    
    return data

def load_kaggle_dataset(dataset_path=None):
    """
    Load the Kaggle dataset
    """
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Sleep_health_and_lifestyle_dataset.csv')
    data = pd.read_csv(dataset_path)
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

def train_and_save_models(dataset_path=None):
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    data = load_kaggle_dataset(dataset_path)
    data = clean_dataset(data)
    data = engineer_features(data)

    # Group Quality of Sleep into Low, Medium, High
    bins = [2, 5, 7, 10]
    labels = ['Low', 'Medium', 'High']
    data['Quality_Group'] = pd.cut(data['Quality of Sleep'], bins=bins, labels=labels, right=True, include_lowest=True)

    # Features to use (remove Person ID, Quality of Sleep, Sleep Disorder, Quality_Group, Blood Pressure, Occupation)
    feature_cols = [col for col in data.columns if col not in ['Person ID', 'Quality of Sleep', 'Sleep Disorder', 'Quality_Group', 'Blood Pressure', 'Occupation']]
    print('FEATURE COLS:', feature_cols)
    X = data[feature_cols]

    # Save encoders for all categorical features with consistent naming
    encoders = {
        'gender': LabelEncoder(),
        'bmi': LabelEncoder(),
        'agegroup': LabelEncoder(),
        'activitylevel': LabelEncoder(),
        'heartratezone': LabelEncoder()
    }
    
    # Transform and save encoders
    X['Gender'] = encoders['gender'].fit_transform(X['Gender'])
    X['BMI Category'] = encoders['bmi'].fit_transform(X['BMI Category'])
    X['Age_Group'] = encoders['agegroup'].fit_transform(X['Age_Group'])
    X['Activity_Level'] = encoders['activitylevel'].fit_transform(X['Activity_Level'])
    X['Heart_Rate_Zone'] = encoders['heartratezone'].fit_transform(X['Heart_Rate_Zone'])
    
    # Save all encoders
    for name, encoder in encoders.items():
        joblib.dump(encoder, os.path.join(models_dir, f'{name}_encoder.joblib'))

    # Encode remaining categorical variables
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # --- Model 1: Predict Quality of Sleep Group ---
    y_group = data['Quality_Group']
    le_group = LabelEncoder()
    y_group_enc = le_group.fit_transform(y_group)
    joblib.dump(le_group, os.path.join(models_dir, 'quality_group_encoder.joblib'))

    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y_group_enc, test_size=0.2, random_state=42, stratify=y_group_enc)
    scaler_g = StandardScaler()
    X_train_g_scaled = scaler_g.fit_transform(X_train_g)
    X_test_g_scaled = scaler_g.transform(X_test_g)

    smote_g = SMOTE(random_state=42, k_neighbors=1)
    X_train_g_bal, y_train_g_bal = smote_g.fit_resample(X_train_g_scaled, y_train_g)

    rf_group = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid_g = {'n_estimators': [100, 200], 'max_depth': [10, 15], 'min_samples_split': [2, 5]}
    grid_g = GridSearchCV(rf_group, param_grid_g, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_g.fit(X_train_g_bal, y_train_g_bal)

    y_pred_g = grid_g.predict(X_test_g_scaled)
    print("\n=== Quality Group Prediction ===")
    print(f"Accuracy: {accuracy_score(y_test_g, y_pred_g):.3f}")
    print(f"F1 Score: {f1_score(y_test_g, y_pred_g, average='weighted'):.3f}")
    print(classification_report(y_test_g, y_pred_g, target_names=le_group.classes_))

    # Save quality group model and scaler
    joblib.dump(grid_g.best_estimator_, os.path.join(models_dir, 'quality_group_model.joblib'))
    joblib.dump(scaler_g, os.path.join(models_dir, 'quality_group_scaler.joblib'))

    # --- Model 2: Predict Sleep Disorder ---
    y_disorder = data['Sleep Disorder'].fillna('None')
    le_disorder = LabelEncoder()
    y_disorder_enc = le_disorder.fit_transform(y_disorder)

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_disorder_enc, test_size=0.2, random_state=42, stratify=y_disorder_enc)
    scaler_d = StandardScaler()
    X_train_d_scaled = scaler_d.fit_transform(X_train_d)
    X_test_d_scaled = scaler_d.transform(X_test_d)

    smote_d = SMOTE(random_state=42, k_neighbors=2)
    X_train_d_bal, y_train_d_bal = smote_d.fit_resample(X_train_d_scaled, y_train_d)

    rf_disorder = RandomForestClassifier(class_weight='balanced_subsample', random_state=42)
    param_grid_d = {'n_estimators': [100, 200], 'max_depth': [10, 15], 'min_samples_split': [2, 5]}
    grid_d = GridSearchCV(rf_disorder, param_grid_d, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_d.fit(X_train_d_bal, y_train_d_bal)

    y_pred_d = grid_d.predict(X_test_d_scaled)
    print("\n=== Sleep Disorder Prediction ===")
    print(f"Accuracy: {accuracy_score(y_test_d, y_pred_d):.3f}")
    print(f"F1 Score: {f1_score(y_test_d, y_pred_d, average='weighted'):.3f}")
    print(classification_report(y_test_d, y_pred_d, target_names=le_disorder.classes_))

    # Save sleep disorder model and scaler
    joblib.dump(grid_d.best_estimator_, os.path.join(models_dir, 'sleep_disorder_model.joblib'))
    joblib.dump(scaler_d, os.path.join(models_dir, 'sleep_disorder_scaler.joblib'))
    joblib.dump(le_disorder, os.path.join(models_dir, 'sleep_disorder_encoder.joblib'))

    # --- Feature Importance Plot for Quality Group Model ---
    importances = grid_g.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances for Quality Group Model')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'quality_group_feature_importance.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and save sleep models.')
    parser.add_argument('--dataset', type=str, default=None, help='Path to the dataset CSV file')
    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train model and get results
    train_and_save_models(args.dataset) 