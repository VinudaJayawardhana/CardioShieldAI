import streamlit as st
import pickle
import pandas as pd
import numpy as np
from fractions import Fraction
import base64
import os
import plotly.graph_objects as go


# Load model
with open('linear_reg_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Background setup
@st.cache_data
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

set_background(r'C:\Users\Vinuda\OneDrive\Desktop\Heart Disease AI\Heart Disease AI\heart.png')

st.title("CardioShield Heart Attack Prediction")
st.write("Enter details to predict heart attack risk (Make sure you have a lab report e.g., Full Body checkup report)")

# Multilingual toggle
language = st.selectbox("🌐 Choose Language", ["English", "සිංහල", "தமிழ்"])
def translate(text):
    if language == "සිංහල":
        return "හෘදයාබාධ අවදානම"
    elif language == "தமிழ்":
        return "இதய நோய் அபாயம்"
    return text

# User inputs
age = st.number_input('Age of the patient', min_value=0, max_value=100)
cholesterol = st.number_input('Cholesterol Level', min_value=0, max_value=500)
heart_rate = st.number_input('Heart Rate of patient', min_value=0.0, max_value=200.0, step=0.1)
diabetes = st.selectbox('Do you have Diabetes?', ['Yes', 'No'])
family_his = st.text_input('Do you have any family history of Heart Attack? Type Yes or No')
smoking = st.selectbox('Are you smoking?', ['Yes', 'No'])
obese = st.selectbox('Are you obese?', ['Yes', 'No'])
alco_consume = st.selectbox('Are you consuming alcohol?', ['Yes', 'No'])
exercise_hours = st.number_input('Exercise Hours per Week', min_value=0, max_value=100)
pre_heart_prob = st.text_input('Do you have any previous heart problems? Type Yes or No')
medi_use = st.selectbox('Have you used any medication before for any heart problems?', ['Yes', 'No'])
stress_level = st.slider('Enter your stress level (1-10): 1-Minimum stress and 10-Maximum stress', min_value=1, max_value=10)
sedentary_hours = st.number_input('Sedentary Hours per Day', min_value=0, max_value=24)
income_level = st.number_input('Enter your income level', min_value=0)
bmi = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0)
tri = st.number_input('Enter triglycerides level (type of fat)', min_value=0)
act_days = st.number_input('Enter Number of days do activities:', min_value=0, max_value=7)
sleep_hours = st.number_input('Enter sleep hours per day:', min_value=0, max_value=24)
blood_pressure = st.text_input('Blood Pressure (e.g., 120/80)')
sex = st.selectbox('Sex', ['Male', 'Female'])
diet = st.selectbox('Diet', ['Vegetarian', 'Non-Vegetarian', 'Vegan'])

# Encodings
medi_use_encoded = 1 if medi_use == 'Yes' else 0
pre_heart_prob_encoded = 1 if pre_heart_prob == 'Yes' else 0
family_his_encoded = 1 if family_his == 'Yes' else 0
alco_consume_encoded = 1 if alco_consume == 'Yes' else 0
obese_encoded = 1 if obese == 'Yes' else 0
smoking_encoded = 1 if smoking == 'Yes' else 0
diabetes_encoded = 1 if diabetes == 'Yes' else 0

def fraction_to_float(fraction_str):
    try:
        return float(Fraction(fraction_str))
    except ValueError:
        return None

blood_pressure_float = fraction_to_float(blood_pressure)

# Input DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'Cholesterol': [cholesterol],
    'Heart Rate': [heart_rate],
    'Diabetes': [diabetes_encoded],
    'Family History': [family_his_encoded],
    'Smoking': [smoking_encoded],
    'Obesity': [obese_encoded],
    'Alcohol Consumption': [alco_consume_encoded],
    'ExerciseHours': [exercise_hours],
    'Previous Heart Problems': [pre_heart_prob_encoded],
    'Medication Use': [medi_use_encoded],
    'Stress Level': [stress_level],
    'SedentaryHours': [sedentary_hours],
    'Income': [income_level],
    'BMI': [bmi],
    'Triglycerides': [tri],
    'Physical Activity Days Per Week': [act_days],
    'Sleep Hours Per Day': [sleep_hours],
    'BloodPressure_Float': [blood_pressure_float],
    'Sex_Female': [1 if sex == 'Female' else 0],
    'Sex_Male': [1 if sex == 'Male' else 0],
    'Diet_encoded': [0 if diet == 'Vegetarian' else 1 if diet == 'Non-Vegetarian' else 2],
})

# Prediction
if st.button('Predict'):
    prediction = loaded_model.predict(input_data)
    st.write(f"Predicted Heart Attack Risk: {prediction[0]:.2f}")

    # Risk Level
    if prediction[0] < 0.5:
        risk_level = 'High'
    elif prediction[0] <= 0.7:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    st.write(f"{translate('Heart Attack Risk Level')}: {risk_level}")

    # Gauge Chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction[0],
        title={'text': "Heart Attack Risk"},
        gauge={
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.5], 'color': "red"},
                {'range': [0.5, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "green"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'value': prediction[0]}
        }
    ))
    st.plotly_chart(gauge_fig)

    # Bar Chart
    bar_chart_labels = ['Blood Pressure', 'BMI', 'Exercise Hours', 'Cholesterol', 'Triglycerides']
    bar_chart_values = [blood_pressure_float, bmi, exercise_hours, cholesterol, tri] 

    st.write("### Visual Representation of Key Health Metrics")
    bar_fig = go.Figure(data=[go.Bar(
        x=bar_chart_labels,
        y=bar_chart_values,
        marker=dict(color=['red', 'blue', 'green', 'orange', 'purple'])
    )])
    bar_fig.update_layout(
        xaxis=dict(title='Health Metrics'),
        yaxis=dict(title='Values'),
        title='Key Health Metrics for Heart Attack Risk'
    )
    st.plotly_chart(bar_fig)

    # Radar Chart
    radar_labels = ['BMI', 'Exercise Hours', 'Cholesterol', 'Triglycerides', 'Stress Level']
    radar_values = [bmi, exercise_hours, cholesterol, tri, stress_level]
    healthy_ranges = [22, 5, 180, 150, 3]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_labels,
        fill='toself',
        name='Your Metrics'
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=healthy_ranges,
        theta=radar_labels,
        fill='toself',
        name='Healthy Range'
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Health Metrics vs Healthy Ranges"
    )
    st.plotly_chart(radar_fig)

    # Feature Importance
    if hasattr(loaded_model, 'coef_'):
        st.write("### 🔍 Feature Importance")
        if len(loaded_model.coef_.shape) > 1:
            weights = loaded_model.coef_[0]
        else:
            weights = loaded_model.coef_
        importance_df = pd.DataFrame({
            'Feature': input_data.columns,
            'Weight': weights
        }).sort_values(by='Weight', key=abs, ascending=False)
        st.dataframe(importance_df)

    # Health Score
    score = 100

    if bmi > 25:
        score -= 10
    if cholesterol > 200:
        score -= 10
    if tri > 150:
        score -= 10
    if exercise_hours < 2:
        score -= 10
    if blood_pressure_float and blood_pressure_float > 120:
        score -= 10
    if stress_level > 7:
        score -= 10

    st.write(f"🏅 Your Cardio Health Score: **{score}/100**")
    if score >= 80:
        st.success("Excellent! Keep up the healthy habits.")
    elif score >= 60:
        st.warning("Moderate risk. Consider improving your lifestyle.")
    else:
        st.error("High risk. Please consult a healthcare provider.")

    # Lifestyle Planner
    st.write("### 🗓️ Weekly Lifestyle Planner")
    if exercise_hours < 2:
        st.write("- 🏃 Add 3×30 min brisk walks this week.")
    if sleep_hours < 6:
        st.write("- 😴 Aim for 7–8 hours of sleep per night.")
    if stress_level > 7:
        st.write("- 🧘 Try 10 mins of daily meditation or breathing exercises.")
    if sedentary_hours > 8:
        st.write("- 🚶 Take short movement breaks every hour.")

    # Recommendations based on original logic
    recommendations = []
    if blood_pressure_float and blood_pressure_float > 120:
        recommendations.append("Maintain healthy blood pressure through diet, exercise, and medication.")
    if bmi > 25:
        recommendations.append("Consider a balanced diet and regular physical activity to manage weight.")
    if exercise_hours < 2:
        recommendations.append("Increase physical activity to at least 150 minutes per week.")
    if cholesterol > 200:
        recommendations.append("Monitor your cholesterol levels and consider dietary changes and medication.")

    if recommendations:
        st.write("### Recommendations to reduce heart attack risk:")
        for rec in recommendations:
            st.write(f"- {rec}")

    # Optional Lab Report Upload
    st.write("### 📄 Optional: Upload Lab Report")
    uploaded_file = st.file_uploader(
        "Upload your full body checkup report (PDF or image)",
        type=["pdf", "png", "jpg"]
    )
    if uploaded_file:
        st.info("File uploaded successfully. OCR parsing will be supported in future versions.")