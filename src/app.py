from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load models, scalers, and encoders for both tasks
rf_quality = joblib.load('models/quality_group_model.joblib')
scaler_q = joblib.load('models/quality_group_scaler.joblib')
le_quality = joblib.load('models/quality_group_encoder.joblib')

rf_disorder = joblib.load('models/sleep_disorder_model.joblib')
scaler_d = joblib.load('models/sleep_disorder_scaler.joblib')
le_disorder = joblib.load('models/sleep_disorder_encoder.joblib')

gender_encoder = joblib.load('models/gender_encoder.joblib')
bmi_encoder = joblib.load('models/bmi_encoder.joblib')
agegroup_encoder = joblib.load('models/agegroup_encoder.joblib')
activitylevel_encoder = joblib.load('models/activitylevel_encoder.joblib')
heartratezone_encoder = joblib.load('models/heartratezone_encoder.joblib')

cluster_scaler = joblib.load('models/cluster_scaler.joblib')
cluster_kmeans = joblib.load('models/cluster_kmeans.joblib')

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

        # --- Feature Engineering to match training ---
        # Age_Group
        if age < 30:
            age_group = '<30'
        elif age < 40:
            age_group = '30-40'
        elif age < 50:
            age_group = '40-50'
        elif age < 60:
            age_group = '50-60'
        else:
            age_group = '60+'
        age_group_encoded = agegroup_encoder.transform([age_group])[0]

        # Sleep_Quality_Score (set to 7 as neutral, not used in prediction)
        sleep_quality_score = sleep_duration * 7 / 10

        # Stress_Sleep_Interaction
        stress_sleep_interaction = stress_level * sleep_duration

        # Activity_Level
        if physical_activity < 30:
            activity_level = 'Low'
        elif physical_activity < 50:
            activity_level = 'Moderate'
        else:
            activity_level = 'High'  # Cap at 'High' to match encoder
        activity_level_encoded = activitylevel_encoder.transform([activity_level])[0]

        # Heart_Rate_Zone
        if heart_rate < 70:
            heart_rate_zone = 'Low'
        elif heart_rate < 80:
            heart_rate_zone = 'Normal'
        else:
            heart_rate_zone = 'High'  # Cap at 'High' to match encoder
        heart_rate_zone_encoded = heartratezone_encoder.transform([heart_rate_zone])[0]

        # Encoded categorical features
        gender_encoded = gender_encoder.transform([gender])[0]
        bmi_encoded = bmi_encoder.transform([bmi_category])[0]

        # New interaction features
        hr_x_activity = heart_rate * physical_activity
        steps_x_age = daily_steps * age
        stress_x_hr = stress_level * heart_rate
        activity_x_sleep = physical_activity * sleep_duration

        # KMeans cluster feature (use saved scaler and kmeans from training)
        cluster_features = np.array([[sleep_duration, physical_activity, stress_level, heart_rate, daily_steps, age]])
        cluster_scaled = cluster_scaler.transform(cluster_features)
        behavior_cluster = cluster_kmeans.predict(cluster_scaled)[0]

        # Build the feature vector in the correct order
        features = np.array([[
            gender_encoded,           # Gender
            age,                      # Age
            sleep_duration,           # Sleep Duration
            physical_activity,        # Physical Activity Level
            stress_level,             # Stress Level
            bmi_encoded,              # BMI Category
            heart_rate,               # Heart Rate
            daily_steps,              # Daily Steps
            age_group_encoded,        # Age_Group
            sleep_quality_score,      # Sleep_Quality_Score
            stress_sleep_interaction, # Stress_Sleep_Interaction
            activity_level_encoded,   # Activity_Level
            heart_rate_zone_encoded,  # Heart_Rate_Zone
            hr_x_activity,            # HRxActivity
            steps_x_age,              # StepsxAge
            stress_x_hr,              # StressxHR
            activity_x_sleep,         # ActivityxSleep
            behavior_cluster          # Behavior_Cluster
        ]])

        # --- Predict Quality Group ---
        features_q = scaler_q.transform(features)
        pred_quality = rf_quality.predict(features_q)
        pred_quality_label = le_quality.inverse_transform(pred_quality)[0]

        # --- Predict Sleep Disorder ---
        features_d = scaler_d.transform(features)
        pred_disorder = rf_disorder.predict(features_d)
        pred_disorder_label = le_disorder.inverse_transform(pred_disorder)[0]

        # Recommendations based on quality group
        recommendations_map = {
            'Low': "Try to increase your sleep duration and reduce stress. Consider more physical activity.",
            'Medium': "Aim for more consistent sleep and moderate your stress levels.",
            'High': "Great job! Keep up your healthy habits."
        }
        recommendations = recommendations_map.get(pred_quality_label, "Keep monitoring your sleep and health habits.")

        return jsonify({
            'quality': pred_quality_label,
            'disorder': pred_disorder_label,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 