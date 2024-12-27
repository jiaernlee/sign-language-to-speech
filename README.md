# Sign Language to Speech Recognition
This project aims to bridge the communication gap for individuals using sign language by translating American Sign Language (ASL) gestures A to Z into text and converting it into speech using a trained deep learning model. The application integrates a webcam feed to capture hand gestures, a pre-trained convolutional neural network (CNN) for gesture recognition, and text-to-speech (TTS) functionality for vocalizing the detected text.

# Features

- Real-time hand gesture detection using Mediapipe.
 - Classification of ASL gestures (A-Z, `space`, `del`, `nothing`) using a CNN
 - Text-to-Speech (TTS) integration for spoken output. 
 - Interactive Streamlit dashboard for easy usage. 

## Installation

1. **Clone the repository** :
```bash 
git clone https://github.com/jiaernlee/sign-language-to-speech 
cd sign-language-to-speech
```
2. **Install dependencies** : 
```bash
pip install -r requirements.txt
```
3. **Download the ASL dataset** :
-   Place the training dataset in `data/asl_alphabet_train/`.
-   Place the testing dataset in `data/asl_alphabet_test/`.

4.  **Ensure you have a trained model** :
-   Use the pre-trained model (`models/sign_language_model.h5`) provided, or train your own.

## Usage
 1. **Run the Streamlit app** :
```bash 
streamlit run app.py
```
 2. **Navigation** :
Navigate through the sidebar to learn more about the project and experience the functionality.  
 3. **Using the webcam**: 
-   Start the webcam using the "Start Webcam" button.
-   Perform ASL gestures in front of the camera.
-   View the recognized text in the "Detected Text" section.
-   Use the "Clear Text" button to reset the detected text.
-   Press the "Convert to Speech" button to hear the text spoken aloud.

## Dataset
The dataset used in this project is the **[ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)** from kaggle, which includes images of hands performing ASL signs for A-Z, `space`, `del`, and `nothing`. Each image is resized to `128x128` pixels and normalized for input to the CNN.
