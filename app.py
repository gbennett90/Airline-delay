import streamlit as st
import pandas as pd
import pickle

st.title("Airline Delay Predictor")

# Load model once
@st.cache_resource
def load_model():
    with open("linreg_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# User inputs for prediction (example inputs, adjust as needed)
airline = st.selectbox("Airline", ["CO", "US", "AA", "AS"])
flight = st.number_input("Flight Number", min_value=1, max_value=9999, value=100)
day_of_week = st.slider("Day of Week", 1, 7, 3)
time = st.number_input("Time (hour)", 0, 23, 15)
length = st.number_input("Flight Length (minutes)", 10, 500, 200)

# Encode categorical feature airline manually or with your method
airline_map = {"CO": 0, "US": 1, "AA": 2, "AS": 3}
airline_encoded = airline_map.get(airline, 0)

# Prepare input dataframe for prediction
input_df = pd.DataFrame({
    "Airline": [airline_encoded],
    "Flight": [flight],
    "DayOfWeek": [day_of_week],
    "Time": [time],
    "Length": [length]
})

if st.button("Predict Delay"):
    prediction = model.predict(input_df)
    st.write(f"Predicted delay: {prediction[0]:.2f} minutes")
