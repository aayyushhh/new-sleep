import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('your_model.pkl')

st.title('Sleep Quality Predictor')

st.header('User Input')

# Input features with specified ranges
sr = st.number_input('Sleep Rating (0-100)', min_value=0, max_value=100, value=50)
rr = st.number_input('Restfulness Rating (0-50)', min_value=0, max_value=50, value=25)
t = st.number_input('Temperature (°C)', min_value=0.0, max_value=100.0, value=25.0)
lm = st.number_input('Light in Room (0-50)', min_value=0.0, max_value=50.0, value=25.0)
bo = st.number_input('Background Noise (0-100)', min_value=0.0, max_value=100.0, value=50.0)
rem = st.number_input('REM Sleep (%)', min_value=0.0, max_value=100.0, value=20.0)
hr = st.number_input('Heart Rate (bpm)', min_value=0, max_value=100, value=75)
stress_level = st.number_input('Stress Level (0-10)', min_value=0, max_value=10, value=5)

if st.button('Predict Sleep Quality'):
    # Create a feature vector from user input
    input_data = np.array([sr, rr, t, lm, bo, rem, hr, stress_level])

    # Reshape the data for prediction
    input_data_reshaped = input_data.reshape(1, -1)

    # Make a prediction
    prediction = model.predict(input_data_reshaped)

    st.header('Prediction Result')  # Display result on the main screen

    if prediction[0] == 1:
        st.write('Good sleep')
    else:
        st.write('Bad sleep')
