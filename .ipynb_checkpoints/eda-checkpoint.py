import streamlit as st

def eda_page():
    st.title("EDA Visualizations")

    st.subheader("1. Class distribution in Training set")
    st.image("images/1.png")

    st.subheader("2. Pixel intensity distribution")
    st.image("images/2.png")

    st.subheader("3. Mean pixel intensity per class")
    st.image("images/3.png")

    st.subheader("4. Sample images")
    st.image("images/4.png")

