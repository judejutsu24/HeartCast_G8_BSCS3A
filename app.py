import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
from datetime import datetime

def get_health_recommendation(risk_level):
    if risk_level.startswith('Low risk'):
        return (
            "🟢 Low Risk:\n"
            "- Maintain your healthy lifestyle.\n"
            "- Continue regular physical activity (at least 150 minutes per week).\n"
            "- Eat a balanced diet rich in fruits, vegetables, and whole grains.\n"
            "- Avoid smoking and limit alcohol consumption.\n"
            "- Keep a healthy weight and monitor your blood pressure and cholesterol annually."
        )
    elif risk_level.startswith('Borderline risk'):
        return (
            "🟡 Borderline Risk:\n"
            "- Consider lifestyle improvements such as increasing exercise and improving your diet.\n"
            "- Reduce salt, sugar, and saturated fat intake.\n"
            "- Monitor your blood pressure, cholesterol, and blood sugar more frequently.\n"
            "- If you smoke, seek help to quit.\n"
            "- Discuss your risk with your healthcare provider for possible early interventions."
        )
    elif risk_level.startswith('Intermediate risk'):
        return (
            "🟠 Intermediate Risk:\n"
            "- Consult your healthcare provider for a personalized risk reduction plan.\n"
            "- You may need medications to control blood pressure, cholesterol, or blood sugar.\n"
            "- Adopt a heart-healthy diet (DASH or Mediterranean diet recommended).\n"
            "- Engage in regular physical activity and aim for a healthy weight.\n"
            "- Avoid tobacco and limit alcohol.\n"
            "- Manage stress and get regular check-ups."
        )
    else:
        return (
            "🔴 High Risk:\n"
            "- Immediate consultation with a healthcare professional is strongly recommended.\n"
            "- Medications may be necessary to control your risk factors (blood pressure, cholesterol, diabetes).\n"
            "- Strictly follow a heart-healthy diet and exercise plan as advised by your doctor.\n"
            "- Stop smoking and avoid secondhand smoke.\n"
            "- Monitor your health closely and attend all medical appointments.\n"
            "- Consider joining a cardiac rehabilitation program if recommended."
        )
    
def create_pdf_report(risk_level, probability, raw_patient_info, recommendation):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("HeartCast - CVD Risk Assessment Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Patient Information:</b>", styles['Heading2']))

    for label, value in raw_patient_info.items():
        story.append(Paragraph(f"{label}: {value}", styles['Normal']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Prediction Result:</b>", styles['Heading2']))
    story.append(Paragraph(f"Risk Level: {risk_level}", styles['Normal']))
    story.append(Paragraph(f"Probability of developing CVD: {probability:.2%}", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Health Recommendation:</b>", styles['Heading2']))
    for line in recommendation.split('\n'):
        story.append(Paragraph(line, styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("This prediction is for educational purposes only and should not be used as a substitute for professional medical advice.", styles['Italic']))
    doc.build(story)
    buffer.seek(0)
    return buffer

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
This app predicts your 10-year risk of cardiovascular disease (CVD) using a machine learning model based on the Framingham Heart Study. 
Enter your information below to receive your CVD risk and health recommendations.
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
    age = st.number_input('Age', min_value=20, max_value=79)
    male = st.selectbox('Sex', ['Male', 'Female']) == 'Male'
    
    # Education level input with descriptions (without numeric prefixes)
    education_levels = [
        'No High School',
        'HS Degree',
        'Some College',
        'College Degree'
    ]
    education = st.selectbox(
        'Education Level',
        options=education_levels
    )

    # Map the selected description back to a numeric value
    education_mapping = {
        'No High School': 1,
        'HS Degree': 2,
        'Some College': 3,
        'College Degree': 4
    }
    education = education_mapping[education]  # Convert the selected description to its numeric value
    
    current_smoker = st.checkbox('Current Smoker')
    cigs_per_day = st.number_input('Cigarettes per Day', min_value=0, max_value=100)
    bp_meds = st.checkbox('On Blood Pressure Medication')
    prevalent_stroke = st.checkbox('Prevalent Stroke')
    prevalent_hyp = st.checkbox('Prevalent Hypertension')
    diabetes = st.checkbox('Diabetes')
    tot_chol = st.number_input('Total Cholesterol (mg/dL)', min_value=100, max_value=600)
    sys_bp = st.number_input('Systolic Blood Pressure (mgHg)', min_value=80, max_value=300)
    dia_bp = st.number_input('Diastolic Blood Pressure (mgHg)', min_value=40, max_value=200)
    bmi = st.number_input('BMI', min_value=15.0, max_value=50.0)
    heart_rate = st.number_input('Heart Rate (BPM)', min_value=40, max_value=200)
    glucose = st.number_input('Glucose (mg/dL)', min_value=40, max_value=400)

    # Prediction button
    if st.button('Predict CVD Risk'):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'education': [education],
            'currentSmoker': [int(current_smoker)],
            'cigsPerDay': [cigs_per_day],
            'BPMeds': [int(bp_meds)],
            'prevalentStroke': [int(prevalent_stroke)],
            'prevalentHyp': [int(prevalent_hyp)],
            'diabetes': [int(diabetes)],
            'totChol': [tot_chol],
            'sysBP': [sys_bp],
            'diaBP': [dia_bp],
            'BMI': [bmi],
            'heartRate': [heart_rate],
            'glucose': [glucose],
            'male': [int(male)]
        })

        # Collect raw patient info for the report
        raw_patient_info = {
            'Age': age,
            'Sex': 'Male' if male else 'Female',
            'Education Level': education,  # This is the label, e.g., 'HS Degree'
            'Current Smoker': 'Yes' if current_smoker else 'No',
            'Cigarettes per Day': cigs_per_day,
            'On Blood Pressure Medication': 'Yes' if bp_meds else 'No',
            'Prevalent Stroke': 'Yes' if prevalent_stroke else 'No',
            'Prevalent Hypertension': 'Yes' if prevalent_hyp else 'No',
            'Diabetes': 'Yes' if diabetes else 'No',
            'Total Cholesterol (mg/dL)': tot_chol,
            'Systolic Blood Pressure (mgHg)': sys_bp,
            'Diastolic Blood Pressure (mgHg)': dia_bp,
            'BMI': bmi,
            'Heart Rate (BPM)': heart_rate,
            'Glucose (mg/dL)': glucose
        }

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
        if probability < 0.05:
            risk_level = 'Low risk (<5%)'
        elif 0.05 <= probability < 0.075:
            risk_level = 'Borderline risk (5% to 7.4%)'
        elif 0.075 <= probability < 0.20:
            risk_level = 'Intermediate risk (7.5% to 19.9%)'
        else:
            risk_level = 'High risk (≥20%)'

        # Display the risk level
        st.write(f'Risk Level: **{risk_level}**')

        # Display the probability
        st.write(f'Probability of developing CVD: {probability:.2%}')

        # Provide specific health recommendations based on risk level
        st.markdown("### Health Recommendations")

        if risk_level.startswith('Low risk'):
            st.info(
                "🟢 **Low Risk:**\n"
                "- Maintain your healthy lifestyle.\n"
                "- Continue regular physical activity (at least 150 minutes per week).\n"
                "- Eat a balanced diet rich in fruits, vegetables, and whole grains.\n"
                "- Avoid smoking and limit alcohol consumption.\n"
                "- Keep a healthy weight and monitor your blood pressure and cholesterol annually."
            )
        elif risk_level.startswith('Borderline risk'):
            st.warning(
                "🟡 **Borderline Risk:**\n"
                "- Consider lifestyle improvements such as increasing exercise and improving your diet.\n"
                "- Reduce salt, sugar, and saturated fat intake.\n"
                "- Monitor your blood pressure, cholesterol, and blood sugar more frequently.\n"
                "- If you smoke, seek help to quit.\n"
                "- Discuss your risk with your healthcare provider for possible early interventions."
            )
        elif risk_level.startswith('Intermediate risk'):
            st.warning(
                "🟠 **Intermediate Risk:**\n"
                "- Consult your healthcare provider for a personalized risk reduction plan.\n"
                "- You may need medications to control blood pressure, cholesterol, or blood sugar.\n"
                "- Adopt a heart-healthy diet (DASH or Mediterranean diet recommended).\n"
                "- Engage in regular physical activity and aim for a healthy weight.\n"
                "- Avoid tobacco and limit alcohol.\n"
                "- Manage stress and get regular check-ups."
            )
        else:  # High risk
            st.error(
                "🔴 **High Risk:**\n"
                "- Immediate consultation with a healthcare professional is strongly recommended.\n"
                "- Medications may be necessary to control your risk factors (blood pressure, cholesterol, diabetes).\n"
                "- Strictly follow a heart-healthy diet and exercise plan as advised by your doctor.\n"
                "- Stop smoking and avoid secondhand smoke.\n"
                "- Monitor your health closely and attend all medical appointments.\n"
                "- Consider joining a cardiac rehabilitation program if recommended."
            )

        # Get health recommendation text
        recommendation = get_health_recommendation(risk_level)

        # Generate PDF report including recommendation
        pdf_buffer = create_pdf_report(risk_level, probability, raw_patient_info, recommendation)

        # Provide download link for the PDF report
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="cvd_risk_report.pdf",
            mime="application/pdf",
            key="download-pdf"
        )

        # Set a flag in session state to indicate a prediction was made
        st.session_state['prediction_made'] = True

    # Show "Test Again" button only if a prediction was made
    if st.session_state.get('prediction_made', False):
        if st.button('Test Again'):
            # Clear all session state variables
            st.session_state.clear()
            # Explicitly uncheck the terms and conditions checkbox
            if "I agree to the terms and conditions." in st.session_state:
                del st.session_state["I agree to the terms and conditions."]
            # Stop execution so the app resets to the start on next interaction
            st.rerun()
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
