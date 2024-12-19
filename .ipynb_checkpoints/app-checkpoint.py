import streamlit as st
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from gtts import gTTS
import os
from datetime import datetime

# Initialize model in session state if not already present
if 'model' not in st.session_state:
    try:
        model = tf.keras.models.load_model('sign_language_model.h5')
        st.session_state.model = model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Class labels
class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G',
    7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'K', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

OUTPUT_FOLDER = "outputs"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1 
)

# Preprocess hand region for the model
def preprocess_hand_region(hand_image):
    image_size = (128, 128)
    hand_image = cv2.resize(hand_image, image_size)
    hand_image = hand_image / 255.0
    hand_image = np.expand_dims(hand_image, axis=0)
    return hand_image

def speak_text_with_gtts(text):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(OUTPUT_FOLDER, f"output_audio_{timestamp}.mp3")
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    st.audio(filename, format="audio/mp3")

# Streamlit UI
st.title("Sign Language Recognition with TTS")

if "detected_text" not in st.session_state:
    st.session_state["detected_text"] = ""

if "run_webcam" not in st.session_state:
    st.session_state["run_webcam"] = False

# Columns for layout
col1, col2 = st.columns([2, 1])

# Webcam in Left Column
with col1:
    st.subheader("Webcam Feed")
    if st.button("Start/Stop Webcam"):
        st.session_state["run_webcam"] = not st.session_state["run_webcam"]
    frame_window = st.image([])

# Detected Text and Controls in Right Column
with col2:
    st.session_state["detected_text"] = st.text_area(
        "Detected Text",
        value=st.session_state["detected_text"],
        height=200,
        key="text_input",
        on_change=lambda: setattr(st.session_state, "detected_text", st.session_state.text_input)
    )

    # Clear Text Button
    if st.button("Clear Text"):
        st.session_state["detected_text"] = ""

    if st.button("Speak Text"):
        if st.session_state["detected_text"]:
            speak_text_with_gtts(st.session_state["detected_text"])
        else:
            st.warning("No text to speak!")

# Webcam Processing
if st.session_state["run_webcam"]:
    cap = cv2.VideoCapture(0)

    while st.session_state["run_webcam"]:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam. Check if it's connected.")
            break

        # Flip frame for a mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks with Mediapipe
        results = hands.process(rgb_frame)
        prediction = "" 
        confidence = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate bounding box
                h, w, c = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

                # Add padding to bounding box
                padding = 20
                x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                hand_region = frame[y_min:y_max, x_min:x_max]
                if hand_region.size > 0:
                    preprocessed_hand = preprocess_hand_region(hand_region)
    
                    # Use model from session state for predictions
                if st.session_state.model is not None:
                    predictions = st.session_state.model.predict(preprocessed_hand, verbose=0)
                    predicted_class = np.argmax(predictions)
                    confidence = predictions[0][predicted_class]
                    
                    if confidence >= 0.8: 
                        prediction = class_labels[predicted_class]
                        
                        # Append prediction to detected text
                        if prediction == "space":
                            st.session_state["detected_text"] += " "
                        elif prediction == "del":
                            st.session_state["detected_text"] = st.session_state["detected_text"][:-1]
                        elif prediction != "nothing":
                            st.session_state["detected_text"] += prediction


        # Display prediction on the frame
        cv2.putText(frame, f"Prediction: {prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, f"Confidence: {confidence}", (10, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Stream the frame in the Streamlit app
        frame_window.image(frame, channels="BGR")

    cap.release()
    cv2.destroyAllWindows()

# Close Mediapipe resources
hands.close()