# app/view_attendance.py
import csv

def view_attendance():
    records = []
    with open("data/attendance.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            records.append({"student_id": row[0], "timestamp": row[1]})
    return records