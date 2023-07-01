import cv2
import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import os

# Dictionary to store face embeddings along with name
known_faces = dict()

def capture_images(name):
    cap = cv2.VideoCapture(0)

    images = []
    for _ in range(5):
        ret, frame = cap.read()
        # Convert to RGB
        frame = frame[:, :, ::-1]
        # Get face embeddings
        resp_obj = DeepFace.analyze(frame, actions=['embedding'])
        images.append(resp_obj["embedding"])

    cap.release()
    cv2.destroyAllWindows()

    # Compute face embeddings and store in known_faces
    known_faces[name] = images

def identify_faces(frame):
    # Get face embeddings
    resp_obj = DeepFace.analyze(frame, actions=['embedding'])

    name = 'Human'
    if "embedding" in resp_obj:
        # Compare with known faces
        for known_name, known_embeddings in known_faces.items():
            for known_embedding in known_embeddings:
                dist = np.linalg.norm(known_embedding - resp_obj["embedding"])
                if dist < 0.1:  # threshold may need to be tuned
                    name = known_name
                    break

    return name

def main():
    st.title("Face Recognition App")
    st.header("Capture Images")

    user_input = st.text_input("Enter your name", "")
    capture_button = st.button("Capture Images")
    if capture_button:
        capture_images(user_input)

    st.header("Live Feed")
    start_button = st.button("Start Webcam")
    placeholder = st.empty()

    if start_button:
        # Open webcam
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            name = identify_faces(frame)
            placeholder.text(name)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
