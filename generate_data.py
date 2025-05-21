import pandas as pd
import numpy as np
from datetime import datetime

# Read existing data
df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')

# Function to generate new data
def generate_new_data(n_new_records):
    # Define possible values based on existing data
    genders = ['Male', 'Female']
    occupations = ['Software Engineer', 'Doctor', 'Nurse', 'Teacher', 'Engineer', 
                  'Accountant', 'Lawyer', 'Salesperson', 'Manager', 'Scientist']
    bmi_categories = ['Normal', 'Normal Weight', 'Overweight', 'Obese']
    sleep_disorders = ['None', 'Sleep Apnea', 'Insomnia']
    
    # Generate new records
    new_data = []
    
    for i in range(n_new_records):
        # Generate age (following existing distribution)
        age = np.random.normal(45, 10)  # Mean age around 45
        age = max(25, min(65, int(age)))  # Keep age between 25-65
        
        # Generate gender
        gender = np.random.choice(genders)
        
        # Generate occupation (with some gender-based probabilities)
        if gender == 'Female':
            occupation = np.random.choice(['Nurse', 'Teacher', 'Accountant', 'Doctor', 'Engineer', 'Manager', 'Scientist'], 
                                        p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05])
        else:
            occupation = np.random.choice(['Doctor', 'Engineer', 'Software Engineer', 'Lawyer', 'Salesperson', 'Manager', 'Teacher'], 
                                        p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.05, 0.05])
        
        # Generate sleep duration (following normal distribution)
        sleep_duration = round(np.random.normal(7.2, 0.8), 1)
        sleep_duration = max(5.5, min(9.0, sleep_duration))
        
        # Generate quality of sleep (1-10)
        quality_of_sleep = int(np.random.normal(7, 1.5))
        quality_of_sleep = max(1, min(10, quality_of_sleep))
        
        # Generate physical activity level (0-100)
        physical_activity = int(np.random.normal(60, 20))
        physical_activity = max(30, min(100, physical_activity))
        
        # Generate stress level (1-10)
        stress_level = int(np.random.normal(6, 1.5))
        stress_level = max(1, min(10, stress_level))
        
        # Generate BMI category
        bmi_category = np.random.choice(bmi_categories, p=[0.3, 0.3, 0.3, 0.1])
        
        # Generate blood pressure based on BMI
        if bmi_category in ['Overweight', 'Obese']:
            systolic = np.random.randint(130, 145)
            diastolic = np.random.randint(85, 95)
        else:
            systolic = np.random.randint(115, 130)
            diastolic = np.random.randint(75, 85)
        blood_pressure = f"{systolic}/{diastolic}"
        
        # Generate heart rate
        heart_rate = np.random.randint(65, 85)
        
        # Generate daily steps
        daily_steps = np.random.randint(3000, 10001)
        
        # Generate sleep disorder based on other factors
        if sleep_duration < 6.5 and stress_level > 7:
            sleep_disorder = np.random.choice(['Sleep Apnea', 'Insomnia'], p=[0.6, 0.4])
        elif bmi_category in ['Overweight', 'Obese'] and sleep_duration < 7:
            sleep_disorder = 'Sleep Apnea'
        else:
            sleep_disorder = np.random.choice(['None', 'Sleep Apnea', 'Insomnia'], p=[0.7, 0.15, 0.15])
        
        # Create new record
        new_record = {
            'Person ID': len(df) + i + 1,
            'Gender': gender,
            'Age': age,
            'Occupation': occupation,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'BMI Category': bmi_category,
            'Blood Pressure': blood_pressure,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Sleep Disorder': sleep_disorder
        }
        
        new_data.append(new_record)
    
    return pd.DataFrame(new_data)

# Generate new data
n_new_records = 900 - len(df)
new_df = generate_new_data(n_new_records)

# Combine with existing data
combined_df = pd.concat([df, new_df], ignore_index=True)

# Save to CSV
combined_df.to_csv('data/Sleep_health_and_lifestyle_dataset.csv', index=False)

print(f"Generated {n_new_records} new records. Total records: {len(combined_df)}") 