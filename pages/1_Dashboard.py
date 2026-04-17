import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 HR Dashboard")

data = pd.read_csv("data/employees.csv")

# Filters
dept_filter = st.selectbox("Filter by Department", ["All"] + list(data["Department"].unique()))

if dept_filter != "All":
    data = data[data["Department"] == dept_filter]

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(data))
col2.metric("Avg Salary", int(data["Salary"].mean()))
col3.metric("Avg Experience", int(data["Experience"].mean()))

# Charts
fig = px.histogram(data, x="Salary", title="Salary Distribution")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.scatter(data, x="Experience", y="Salary", color="Performance",
                  title="Experience vs Salary")
st.plotly_chart(fig2, use_container_width=True)