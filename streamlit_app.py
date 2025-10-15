# Streamlit Dashboard for Accident Hotspot Visualization

import streamlit as st
import pandas as pd
import joblib

st.title('🚦 Road Accident Hotspot Predictor')
st.write('Predicting accident-prone areas using AI')

model = joblib.load('accident_rf_model.joblib')
uploaded_file = st.file_uploader('Upload traffic data CSV', type='csv')

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    preds = model.predict(data)
    data['Predicted_Risk'] = preds
    st.dataframe(data.head())
