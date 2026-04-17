import streamlit as st
import pandas as pd
import joblib

st.title("🔮 AI Performance Prediction")

model = joblib.load("models/performance_model.pkl")

st.markdown("### Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 60, 30)
    experience = st.slider("Experience", 1, 20, 5)
    salary = st.number_input("Salary", 20000, 150000, 50000)

with col2:
    training_hours = st.slider("Training Hours", 10, 100, 40)
    department = st.selectbox("Department", ["HR", "IT", "Sales", "Finance"])
    education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])

dept_map = {"HR": 0, "IT": 1, "Sales": 2, "Finance": 3}
edu_map = {"Bachelors": 0, "Masters": 1, "PhD": 2}

if st.button("🚀 Predict Performance"):

    input_data = pd.DataFrame({
        'Age': [age],
        'Experience': [experience],
        'Salary': [salary],
        'Training_Hours': [training_hours],
        'Department': [dept_map[department]],
        'Education': [edu_map[education]]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("### 📊 Prediction Result")

    if prediction == 1:
        st.success(f"High Performer 🚀 (Confidence: {round(probability*100,2)}%)")
    else:
        st.error(f"Needs Improvement ⚠️ (Confidence: {round(probability*100,2)}%)")

    st.markdown("### 🧠 Explanation")

    st.info("""
    Performance is influenced by:
    - Experience level  
    - Training hours  
    - Salary competitiveness  
    """)