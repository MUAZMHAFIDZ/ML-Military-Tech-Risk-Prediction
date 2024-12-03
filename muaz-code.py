import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

df_military_tech = pd.read_csv('military_tech.csv')

with open('failure_risk_model.sav', 'rb') as file:
    pipeline = pickle.load(file)

if 'page' not in st.session_state:
    st.session_state.page = "Home"
st.sidebar.title("Select Page")
home_button = st.sidebar.button("Home")
dataset_button = st.sidebar.button("Dataset")
graph_button = st.sidebar.button("Graph")
prediction_button = st.sidebar.button("Prediction")

if home_button:
    st.session_state.page = "Home"
elif dataset_button:
    st.session_state.page = "Dataset"
elif graph_button:
    st.session_state.page = "Graph"
elif prediction_button:
    st.session_state.page = "Prediction"

if st.session_state.page == "Home":
    st.title("Failure Risk Prediction By Muaz")
    st.subheader("Welcome!!!")
    st.image("https://cdn.stocksnap.io/img-thumbs/960w/nature-landscape_EQ7JVUWATQ.jpg", caption="Military Tech Risk")

elif st.session_state.page == "Dataset":
    st.write("This is the dataset:")
    st.dataframe(df_military_tech)

elif st.session_state.page == "Graph":
    st.subheader("Visualization of Parameters")
    st.write("1. Operating Hours")
    plt.figure(figsize=(8, 4))
    plt.plot(df_military_tech['Operating Hours'], label="Operating Hours")
    plt.title("Operating Hours Distribution")
    plt.xlabel("Index")
    plt.ylabel("Operating Hours")
    plt.legend()
    st.pyplot(plt)
    st.write("2. Temperature (Celsius)")
    plt.figure(figsize=(8, 4))
    plt.plot(df_military_tech['Temperature (Celsius)'], color='orange', label="Temperature (Celsius)")
    plt.title("Temperature (Celsius) Distribution")
    plt.xlabel("Index")
    plt.ylabel("Temperature (Celsius)")
    plt.legend()
    st.pyplot(plt)
    st.write("3. Vibration (m/s^2)")
    plt.figure(figsize=(8, 4))
    plt.plot(df_military_tech['Vibration (m/s^2)'], color='green', label="Vibration (m/s^2)")
    plt.title("Vibration (m/s^2) Distribution")
    plt.xlabel("Index")
    plt.ylabel("Vibration (m/s^2)")
    plt.legend()
    st.pyplot(plt)

elif st.session_state.page == "Prediction":
    st.title("Failure Risk Prediction By Muaz")

    operating_hours = st.number_input("Operating Hours", min_value=0, value=100)
    temperature = st.number_input("Temperature (Celsius)", min_value=-50, value=25)
    vibration = st.number_input("Vibration (m/s^2)", min_value=0.0, value=0.5, step=0.1)
    maintenance_score = st.number_input("Maintenance Score", min_value=0, max_value=100, value=75)
    critical_components = st.number_input("Critical Components", min_value=1, value=5)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            'Operating Hours': [operating_hours],
            'Temperature (Celsius)': [temperature],
            'Vibration (m/s^2)': [vibration],
            'Maintenance Score': [maintenance_score],
            'Critical Components': [critical_components]
        })
        
        predicted_risk = pipeline.predict(input_data)[0]
        
        st.success(f"Predicted Failure Risk: {predicted_risk:.2f}%")
        