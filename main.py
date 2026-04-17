import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/performance_model.pkl")

st.title("Employee Performance Predictor")

st.write("Enter employee details:")

# Inputs
age = st.slider("Age", 20, 60, 30)
experience = st.slider("Experience", 1, 20, 5)
salary = st.number_input("Salary", 20000, 150000, 50000)
training_hours = st.slider("Training Hours", 10, 100, 40)

department = st.selectbox("Department", ["HR", "IT", "Sales", "Finance"])
education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])

# Encode manually (same as training)
dept_map = {"HR": 0, "IT": 1, "Sales": 2, "Finance": 3}
edu_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}

# Prediction button
if st.button("Predict Performance"):

    input_data = pd.DataFrame({
        'Age': [age],
        'Experience': [experience],
        'Salary': [salary],
        'Training_Hours': [training_hours],
        'Department': [dept_map[department]],
        'Education': [edu_map[education]]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("High Performer 🚀")
    else:
        st.error("Low Performer ⚠️")