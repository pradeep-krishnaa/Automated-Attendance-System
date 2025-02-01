# app/capture_images.py
import cv2
import os
import json

def capture_images(student_id, student_name):
    # Create the data/images directory if it doesn't exist
    if not os.path.exists("data/images"):
        os.makedirs("data/images")

    cap = cv2.VideoCapture(0)
    count = 0
    
    print(f"Capturing images for {student_name}...")
    print("Press SPACE to capture an image")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display counter
        cv2.putText(frame, f"Captured: {count}/10", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Capture Images (SPACE to capture, Q to quit)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Space key to capture
            # Save the image
            image_name = f"data/images/{student_id}_{count}.jpg"
            cv2.imwrite(image_name, frame)
            print(f"Captured image {count + 1}/10")
            count += 1
            
        if key == ord('q') or count >= 10:  # Quit if 'q' pressed or 10 images captured
            break

    cap.release()
    cv2.destroyAllWindows()

    # Store student details in a JSON file
    student_data = {"student_id": student_id, "name": student_name}
    try:
        with open("data/students.json", "r") as f:
            students = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        students = []
    
    students.append(student_data)
    with open("data/students.json", "w") as f:
        json.dump(students, f, indent=4)
    
    print(f"Captured {count} images for {student_name}")

if __name__ == "__main__":
    student_id = input("Enter student ID: ")
    student_name = input("Enter student name: ")
    capture_images(student_id, student_name)