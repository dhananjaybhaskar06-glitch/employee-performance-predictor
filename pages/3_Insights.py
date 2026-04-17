import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Insights")

data = pd.read_csv("data/employees.csv")

fig = px.box(data, x="Department", y="Salary", title="Salary by Department")
st.plotly_chart(fig)

fig2 = px.box(data, x="Education", y="Performance", title="Performance by Education")
st.plotly_chart(fig2)