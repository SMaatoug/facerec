import cv2
import dlib
import face_recognition
import streamlit as st
from PIL import Image
import numpy as np
import os

# Set up face detector and predictor (used for face landmarks)
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # provide path to your .dat file

# Dictionary to store face encodings along with name
known_faces = dict()

def capture_images(name):
    cap = cv2.VideoCapture(0)

    images = []
    for _ in range(5):
        ret, frame = cap.read()
        # Convert to RGB
        frame = frame[:, :, ::-1]
        # Get face encodings
        faces = face_recognition.face_locations(frame)
        if len(faces) == 0:
            st.write("No face detected. Try again.")
        else:
            # Assuming only one face, get that face
            top, right, bottom, left = faces[0]
            face_image = frame[top:bottom, left:right]
            images.append(face_image)

    cap.release()
    cv2.destroyAllWindows()

    # Compute face encodings and store in known_faces
    encodings = [face_recognition.face_encodings(img)[0] for img in images]
    known_faces[name] = encodings

def identify_faces(frame):
    # Detect faces
    faces = face_recognition.face_locations(frame)
    
    for top, right, bottom, left in faces:
        # Get face encodings
        face_encodings = face_recognition.face_encodings(frame, [(top, right, bottom, left)])

        name = 'Human'
        for face_encoding in face_encodings:
            # Compare with known faces
            for known_name, known_encodings in known_faces.items():
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                if True in matches:
                    name = known_name
                    break

            # Draw box and name
            frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            frame = cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    return frame

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
            frame = identify_faces(frame)
            placeholder.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
