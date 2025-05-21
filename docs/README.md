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
- Uses machine learning model trained on real sleep health data
- Provides quality classification (Poor to Outstanding)
- Modern, responsive UI
- Real-time predictions

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

### Model Performance

The model has been trained and validated using:
- Stratified K-Fold Cross-Validation (3 folds)
- Balanced class weights to handle class imbalance
- Feature scaling for numerical inputs
- Categorical encoding for nominal variables

The model provides both a numerical score and a quality classification, along with personalized recommendations for improving sleep quality.

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
- Includes data exploration and model training scripts

## Note

This application uses a machine learning model trained on real sleep health and lifestyle data. The model takes into account various factors that can affect sleep quality and provides a comprehensive assessment based on these inputs. 