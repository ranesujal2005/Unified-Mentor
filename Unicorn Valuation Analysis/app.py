
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and feature names
model = joblib.load("unicorn_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.set_page_config(page_title=" Unicorn Valuation Predictor", layout="centered")
st.title(" Unicorn Startup Valuation Predictor")

# Extract dropdown values from feature names
countries = sorted([f.replace("Country_", "") for f in feature_names if f.startswith("Country_")])
sectors = sorted([f.replace("Sector_", "") for f in feature_names if f.startswith("Sector_")])
cities = sorted([f.replace("City_", "") for f in feature_names if f.startswith("City_")])
years = sorted([int(f.replace("Year_", "")) for f in feature_names if f.startswith("Year_")])
months = sorted([int(f.replace("Month_", "")) for f in feature_names if f.startswith("Month_")])

# Sidebar user inputs
st.sidebar.header("Input Features")
country = st.sidebar.selectbox("Country", countries)
sector = st.sidebar.selectbox("Sector", sectors)
city = st.sidebar.selectbox("City", cities)
year = st.sidebar.selectbox("Year", years)
month = st.sidebar.selectbox("Month", months)

# Build input vector
input_df = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

input_fields = {
    f"Country_{country}": 1,
    f"Sector_{sector}": 1,
    f"City_{city}": 1,
    f"Year_{year}": 1,
    f"Month_{month}": 1
}

for col, val in input_fields.items():
    if col in input_df.columns:
        input_df.at[0, col] = val

# Display active features for debugging
st.subheader(" Active Features")
st.dataframe(input_df.T[input_df.T[0] != 0])

# Prediction
if st.button("Predict Valuation"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f" Estimated Valuation: ${prediction:.2f} Billion")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
