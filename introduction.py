import streamlit as st

def intro_page():
    st.title("Welcome to the Sign Language Recognition App")
    
    st.markdown("""
    ### Purpose
    This project aims to break communication barriers and promote inclusivity by enabling real-time translation of American Sign Language (ASL) gestures into speech using cutting-edge machine learning technology.

    ### Features
    - **Real-Time Gesture Recognition:** Detects static ASL gestures with high accuracy.
    - **Text-to-Speech Integration:** Converts recognized gestures into speech for seamless communication.
    - **User-Friendly Interface:** Interactive and accessible dashboard for exploration.

    ### Motivation
    People with hearing and speech impairments face significant challenges in day-to-day communication. This project provides an innovative solution to enhance accessibility and foster inclusive interactions.

    ### Explore the App
    Navigate through the sections using the sidebar to learn more about the project and experience the functionality.

    """)