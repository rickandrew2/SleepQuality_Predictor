# Sleep Quality & Disorder Predictor

A Flask web application that predicts both sleep quality (as a category) and the likelihood/type of sleep disorder based on various lifestyle and health factors, with actionable lifestyle insights.

## Features

- Predicts **sleep quality category** (High, Medium, Low) based on multiple factors
- Predicts **sleep disorder** (None, Insomnia, Sleep Apnea)
- Uses machine learning models trained on real sleep health data
- Provides clear, actionable lifestyle insights based on your results
- Modern, responsive UI
- Real-time predictions
- Visualizes feature importance (traditional and SHAP)

## Sleep Quality Categories

The model predicts sleep quality as one of three categories:
- **High:** Excellent sleep quality. Keep it up!
- **Medium:** Good, but could improve.
- **Low:** Needs attention. Try to improve your sleep habits.

This grouping is based on the original dataset's sleep quality scores, but is more robust and user-friendly than a raw numeric score. The model uses a Random Forest Classifier trained on real sleep health data to make predictions.

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

### Feature Engineering

- **Interaction Features:**
  - Heart Rate × Physical Activity
  - Daily Steps × Age
  - Stress Level × Heart Rate
  - Physical Activity × Sleep Duration
- **Cluster Feature:**
  - Behavioral cluster (using KMeans on key lifestyle metrics)
- **Categorical Encoding:**
  - Gender, BMI Category, Age Group, Activity Level, Heart Rate Zone
- **Other:**
  - Age group, activity level, heart rate zone, sleep quality score (engineered), stress-sleep interaction

### Model Training and Evaluation

- **Data Split:** 80% training, 20% test, stratified by class
- **Data Cleaning:** Duplicate removal, noise addition, outlier handling
- **Feature Engineering:** As above
- **Model 1:** Random Forest Classifier for **Sleep Quality Category** (High/Medium/Low)
- **Model 2:** Random Forest Classifier for **Sleep Disorder** (None, Insomnia, Sleep Apnea)
- **Hyperparameter Tuning:** 3-fold stratified cross-validation
- **Final Evaluation:** Metrics reported on held-out test set
- **Interpretability:** Feature importance via Random Forest and SHAP

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

3. Train the models:
```bash
python src/train_model.py
```

4. Run the application:
```bash
python src/app.py
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

2. Click "Predict Sleep Quality" to get:
   - Your sleep quality **category** (High, Medium, Low)
   - Your predicted sleep disorder (if any)
   - **Lifestyle Insights** tailored to your sleep quality result

## Technical Details

- Built with Flask 2.x
- Uses scikit-learn for machine learning
- Implements feature scaling, clustering, and categorical encoding
- Trained on real sleep health and lifestyle dataset
- Includes data cleaning, exploration, and model training scripts
- Generates feature importance plots using both Random Forest and SHAP
- Predicts both sleep quality (as a category) and sleep disorder in a single app
- UI dynamically updates lifestyle insights based on your predicted sleep quality

## Note

This application uses machine learning models trained on real sleep health and lifestyle data. The models take into account various factors that can affect sleep quality and sleep disorders, providing a comprehensive assessment and actionable lifestyle insights. The code and app have been updated to ensure robust, realistic, and interpretable predictions for both sleep quality and sleep disorder. 