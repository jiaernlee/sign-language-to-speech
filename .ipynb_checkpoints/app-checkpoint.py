import streamlit as st
from introduction import intro_page
from predictor import prediction_page

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Predictor"], index=0)

# Render the selected page
if page == "Introduction":
    intro_page()
elif page == "Predictor":
    prediction_page()
