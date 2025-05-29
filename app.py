import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('framingham.csv')
    return data

data = load_data()

# Preprocess the data
def preprocess_data(data):
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes'])
    
    # Normalize continuous variables
    scaler = StandardScaler()
    continuous_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    data[continuous_cols] = scaler.fit_transform(data[continuous_cols])
    
    return data, scaler

# Train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Preprocess the data and train the model
preprocessed_data, scaler = preprocess_data(data)
X = preprocessed_data.drop('TenYearCHD', axis=1)
y = preprocessed_data['TenYearCHD']
model = train_model(X, y)

# Save the model and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Custom CSS for the header bar
header_css = """
<style>
.header-bar {
    top: 0;
    left: 0;
    width: 100%; 
    height: 20%;
    background-color: #1f2b5b;
    color: white;
    text-align: center;
    line-height: 60px;
    font-size: 24px;
    font-weight: bold;
    z-index: 1000; 
    border-radius: 0;
}
body {
    padding-top: 60px; 
}
</style>
"""

# Inject custom CSS for the header
st.markdown(header_css, unsafe_allow_html=True)

# Header bar content
header_content = """
<div class="header-bar">
    HeartCast: CVD Risk Prediction
</div>
"""

# Render the header bar
st.markdown(header_content, unsafe_allow_html=True)

# Description below the header
st.markdown("<br>", unsafe_allow_html=True)
st.write("""
This web app predicts the 10-year risk of cardiovascular disease (CVD) based on patient data.
Please enter the patient's information below:
""")

# Terms and Conditions Checklist
st.markdown("### Terms and Conditions")
st.write("""
By using this application, you agree to the following:
- The predictions provided by this app are for educational purposes only.
- This app does not provide medical advice, diagnosis, or treatment.
- Always consult with a healthcare professional for medical concerns.
- Your data will not be stored or shared and will only be used for the purpose of generating predictions during this session.
""")

# Add a required checkbox
agree_to_terms = st.checkbox("I agree to the terms and conditions.")

# Input fields
if agree_to_terms:
    st.markdown("### Patient Information")
    age = st.number_input('Age', min_value=18, max_value=100)
    male = st.selectbox('Sex', ['Male', 'Female']) == 'Male'
    current_smoker = st.checkbox('Current Smoker')
    cigs_per_day = st.number_input('Cigarettes per Day', min_value=0, max_value=100)
    bp_meds = st.checkbox('On Blood Pressure Medication')
    prevalent_stroke = st.checkbox('Prevalent Stroke')
    prevalent_hyp = st.checkbox('Prevalent Hypertension')
    diabetes = st.checkbox('Diabetes')
    tot_chol = st.number_input('Total Cholesterol', min_value=100, max_value=600)
    sys_bp = st.number_input('Systolic Blood Pressure', min_value=80, max_value=300)
    dia_bp = st.number_input('Diastolic Blood Pressure', min_value=40, max_value=200)
    bmi = st.number_input('BMI', min_value=15, max_value=50)
    heart_rate = st.number_input('Heart Rate', min_value=40, max_value=200)
    glucose = st.number_input('Glucose', min_value=40, max_value=400)

    # Prediction button
    if st.button('Predict CVD Risk'):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'male': [male],
            'currentSmoker': [current_smoker],
            'cigsPerDay': [cigs_per_day],
            'BPMeds': [bp_meds],
            'prevalentStroke': [prevalent_stroke],
            'prevalentHyp': [prevalent_hyp],
            'diabetes': [diabetes],
            'totChol': [tot_chol],
            'sysBP': [sys_bp],
            'diaBP': [dia_bp],
            'BMI': [bmi],
            'heartRate': [heart_rate],
            'glucose': [glucose]
        })
        
        # Perform one-hot encoding
        input_data = pd.get_dummies(input_data, columns=['male', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes'])
        
        # Ensure all columns from training are present
        for col in X.columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[X.columns]
        
        # Scale the input data
        continuous_cols = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
        input_data[continuous_cols] = scaler.transform(input_data[continuous_cols])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        # Display result
        st.subheader('Prediction Result:')

        # Categorize the risk level based on probability
        if probability >= 0.7:
            risk_level = 'High'
        elif 0.4 <= probability < 0.7:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        # Display the risk level
        st.write(f'Risk Level: **{risk_level}**')

        # Display the probability
        st.write(f'Probability of developing CVD: {probability:.2%}')
else:
    st.warning("You must agree to the terms and conditions to proceed.")

# Custom CSS for footer
footer_css = """
<style>
.footer {
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #1f2b5b; 
    color: white;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
    height: 10vh;
}
</style>
"""

# Inject custom CSS
st.markdown(footer_css, unsafe_allow_html=True)

# Footer content
footer_content = """
<div class="footer">
    <p>Copyright © 2025 All rights reserved<br>
    GROUP 8 (BSCS-3A) - Software Engineering 2</p>
</div>
"""

# Render the footer
st.markdown(footer_content, unsafe_allow_html=True)
