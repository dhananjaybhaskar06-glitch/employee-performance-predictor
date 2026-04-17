import streamlit as st

st.set_page_config(page_title="HR Analytics Pro", layout="wide")

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
h1, h2, h3 {
    color: #00C9A7;
}
.stButton>button {
    background-color: #00C9A7;
    color: black;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏢 HR Analytics & Performance Intelligence System")

st.write("AI-powered system for predicting and analyzing employee performance.")