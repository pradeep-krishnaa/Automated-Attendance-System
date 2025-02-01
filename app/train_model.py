# app/train_model.py
import face_recognition
import os
import json
import pickle
import numpy as np

def train_model():
    # Load student data from JSON
    with open("data/students.json", "r") as f:
        students = json.load(f)
    
    known_face_encodings = []
    known_face_names = []
    
    print("Training model...")
    
    for student in students:
        student_id = student["student_id"]
        student_name = student["name"]
        
        # Look for student images
        for i in range(10):  # We capture 10 images per student
            image_path = f"data/images/{student_id}_{i}.jpg"
            if os.path.exists(image_path):
                # Load and encode face
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(student_name)
                    print(f"Processed image {i+1} for {student_name}")
    
    # Save the encodings
    if known_face_encodings:
        with open("models/trained_model.pkl", "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        print("Model trained successfully!")
    else:
        print("No faces found in the images!")

# Train the model
if __name__ == "__main__":
    train_model()