import streamlit as st
from introduction import intro_page
from predictor import prediction_page
from eda import eda_page

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Predictor"], index=0)

# Render the selected page
if page == "Introduction":
    intro_page()
elif page == "Predictor":
    prediction_page()
elif page == "EDA":
    eda_page()
