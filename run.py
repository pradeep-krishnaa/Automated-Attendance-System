from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import face_recognition
import pickle
import csv
from datetime import datetime
import numpy as np
import json

app = Flask(__name__)

# Load the trained model and students data
with open("models/trained_model.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

with open("data/students.json", "r") as f:
    students_data = json.load(f)

# Convert the list of students into a dictionary for easy lookup
students_dict = {student['student_id']: student for student in students_data}

# Function to mark attendance
def mark_attendance(student_id, student_name):
    with open("data/attendance.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([student_id, student_name, datetime.now()])

# Global variable to store recognized student details
recognized_student = None

def generate_frames():
    global recognized_student
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Debug logs
        print(f"Face locations: {face_locations}")
        print(f"Face encodings: {face_encodings}")

        # Convert back to BGR for OpenCV display
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        recognized_student = None

        if face_locations and face_encodings:  # Check if faces are detected
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare the face with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                # Use the known face with the smallest distance
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        student_id = known_face_names[best_match_index]
                        # Look up student information from students_dict
                        student_info = students_dict.get(student_id, {"name": "Unknown"})
                        name = student_info.get("name", "Unknown")
                        recognized_student = {
                            "student_id": student_id,
                            "student_name": name
                        }

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Draw the name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame.")
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in frame conversion: {e}")
            continue

    cap.release()
    print("Video feed stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance_route():
    data = request.json
    student_id = data.get("student_id")
    student_name = data.get("student_name")
    if student_id and student_name:
        mark_attendance(student_id, student_name)
        return jsonify({"status": "success", "message": "Attendance marked successfully!"})
    return jsonify({"status": "error", "message": "Invalid data!"})

@app.route('/attendance')
def attendance():
    records = []
    with open("data/attendance.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3:
                records.append({"student_id": row[0], "student_name": row[1], "timestamp": row[2]})
            else:
                # Handle the case where row does not have enough elements
                print(f"Warning: Row does not have enough elements: {row}")
    return render_template('attendance.html', records=records)

@app.route('/get_recognized_student')
def get_recognized_student():
    global recognized_student
    if recognized_student:
        return jsonify(recognized_student)
    else:
        return jsonify({"student_id": None, "student_name": None})

if __name__ == '__main__':
    app.run(debug=True)