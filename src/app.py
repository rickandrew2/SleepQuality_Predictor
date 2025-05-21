from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('../models/sleep_model.joblib')
scaler = joblib.load('../models/scaler.joblib')
gender_encoder = joblib.load('../models/gender_encoder.joblib')
bmi_encoder = joblib.load('../models/bmi_encoder.joblib')
agegroup_encoder = joblib.load('../models/agegroup_encoder.joblib')
stepsgroup_encoder = joblib.load('../models/stepsgroup_encoder.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form
        sleep_duration = float(request.form['sleep_duration'])
        physical_activity = float(request.form['physical_activity'])
        stress_level = float(request.form['stress_level'])
        bmi_category = request.form['bmi_category']
        heart_rate = float(request.form['heart_rate'])
        daily_steps = float(request.form['daily_steps'])
        age = float(request.form['age'])
        gender = request.form['gender']

        # Encode categorical variables
        gender_encoded = gender_encoder.transform([gender])[0]
        bmi_encoded = bmi_encoder.transform([bmi_category])[0]

        # Create age group and steps group
        age_bins = [0, 30, 40, 50, 60, 100]
        age_labels = ['<30', '30-40', '40-50', '50-60', '60+']
        age_group = pd.cut([age], bins=age_bins, labels=age_labels)[0]
        age_group_encoded = agegroup_encoder.transform([age_group])[0]

        steps_bins = [0, 4000, 7000, 10000, 20000]
        steps_labels = ['Low', 'Medium', 'High', 'Very High']
        steps_group = pd.cut([daily_steps], bins=steps_bins, labels=steps_labels)[0]
        steps_group_encoded = stepsgroup_encoder.transform([steps_group])[0]

        # Create feature array (order must match model training)
        features = np.array([[
            sleep_duration,
            physical_activity,
            stress_level,
            bmi_encoded,
            heart_rate,
            daily_steps,
            age,
            gender_encoded,
            age_group_encoded,
            steps_group_encoded
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction (classifier)
        predicted_class = int(model.predict(features_scaled)[0])
        # Compute continuous score using predict_proba
        class_probs = model.predict_proba(features_scaled)[0]
        class_labels = model.classes_
        score = float(np.dot(class_probs, class_labels))
        score_rounded = round(score, 2)
        # Map class to quality label
        quality_map = {
            4: "Poor",
            5: "Fair",
            6: "Good",
            7: "Very Good",
            8: "Excellent",
            9: "Outstanding"
        }
        # Use the closest class for quality and recommendations
        closest_class = int(round(score))
        quality = quality_map.get(closest_class, "Unknown")
        # Recommendations based on score
        recommendations_map = {
            4: "Try to increase your sleep duration and reduce stress. Consider more physical activity.",
            5: "Aim for more consistent sleep and moderate your stress levels.",
            6: "Maintain your current habits, but small improvements in activity or stress could help.",
            7: "Great job! Keep up your healthy habits.",
            8: "Excellent! Continue your routine for optimal sleep.",
            9: "Outstanding! You have excellent sleep hygiene."
        }
        recommendations = recommendations_map.get(closest_class, "Keep monitoring your sleep and health habits.")
        return jsonify({
            'prediction': predicted_class,
            'quality': quality,
            'score': score_rounded,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 