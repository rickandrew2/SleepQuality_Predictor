# Sleep Quality Predictor

A Flask web application that predicts sleep quality based on various lifestyle and health factors.

## Features

- Predicts sleep quality score based on multiple factors:
  - Sleep Duration
  - Physical Activity Level
  - Stress Level
  - BMI Category
  - Heart Rate
  - Daily Steps
  - Age
  - Gender
- Uses machine learning models trained on real sleep health data
- Provides quality classification (Poor to Outstanding)
- Modern, responsive UI
- Real-time predictions
- Visualizes feature importance (traditional and SHAP)

## Sleep Quality Scale

The model predicts sleep quality on a scale from 4 to 9, where:
- 4: Poor - Indicates significant sleep issues that need attention
- 5: Fair - Suggests room for improvement in sleep habits
- 6: Good - Represents adequate sleep quality
- 7: Very Good - Shows healthy sleep patterns
- 8: Excellent - Indicates optimal sleep quality
- 9: Outstanding - Represents exceptional sleep quality

This scale is based on the original dataset measurements and represents the actual sleep quality assessments. The model uses a Random Forest Classifier trained on real sleep health data to make predictions.

### Input Ranges and Categories

The model accepts the following input ranges:
- Sleep Duration: 4-10 hours
- Physical Activity Level: 0-120 minutes per day
- Stress Level: 1-10 scale
- BMI Categories: Underweight, Normal, Overweight, Obese
- Heart Rate: 40-100 beats per minute
- Daily Steps: 0-10,000 steps
- Age: 18-80 years
- Gender: Male, Female

### Model Training and Evaluation

**How the Model Was Trained and Validated**

- **Data Split:** The dataset was split into training (60%), validation (20%), and test (20%) sets using stratified sampling to preserve class distribution.
- **Data Cleaning:** Duplicate entries were removed and small random noise was added to numeric features to improve robustness. Outliers were clipped to the 1st and 99th percentiles.
- **Feature Engineering:** Categorical variables were encoded, and age and daily steps were binned into groups. Features that could leak the target variable were excluded.
- **Baseline Model:** A Logistic Regression model with balanced class weights was trained as a baseline.
- **Main Model:** A Random Forest Classifier was trained with reduced complexity (limited tree depth, more samples per leaf, balanced class weights).
- **Hyperparameter Tuning:** 5-fold stratified cross-validation was used on the training set to select the best hyperparameters for the Random Forest.
- **Validation:** The best model was evaluated on the validation set for model selection.
- **Final Evaluation:** The selected model was evaluated on the held-out test set to report final performance metrics.
- **Interpretability:** Feature importance was analyzed using both traditional Random Forest importances and SHAP values.

- **Data Cleaning:**
  - Removes duplicate entries from the dataset
  - Adds realistic noise to numeric features for robustness
  - Handles outliers by clipping extreme values
- **Feature Engineering:**
  - Encodes categorical variables
  - Bins age and daily steps into groups
  - Removes features that could leak the target variable
- **Modeling:**
  - Uses both Logistic Regression (as a baseline) and Random Forest Classifier (main model)
  - Reduces model complexity to prevent overfitting (limited tree depth, more samples per leaf)
  - Implements a robust train/validation/test split (60%/20%/20%)
  - Uses 5-fold stratified cross-validation for hyperparameter tuning
- **Performance:**
  - Achieves realistic accuracy and F1 scores (e.g., ~96% test accuracy, not 100%)
  - Reports class-wise performance and confusion matrix
- **Feature Importance:**
  - Generates and saves both traditional and SHAP feature importance plots for interpretability

Performance may vary depending on the dataset and random seed.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your health and lifestyle information:
   - Sleep Duration (hours)
   - Physical Activity Level (minutes)
   - Stress Level (1-10)
   - BMI Category
   - Heart Rate (bpm)
   - Daily Steps
   - Age
   - Gender

2. Click "Predict Sleep Quality" to get your sleep quality classification and score

## Technical Details

- Built with Flask 2.0.1
- Uses scikit-learn for machine learning
- Implements feature scaling and categorical encoding
- Trained on real sleep health and lifestyle dataset
- Includes data cleaning, exploration, and model training scripts
- Generates feature importance plots using both Random Forest and SHAP

## Note

This application uses a machine learning model trained on real sleep health and lifestyle data. The model takes into account various factors that can affect sleep quality and provides a comprehensive assessment based on these inputs. The code and app have been updated to ensure robust, realistic, and interpretable predictions. 