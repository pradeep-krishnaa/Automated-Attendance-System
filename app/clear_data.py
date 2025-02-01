import os
import shutil

def clear_dataset():
    # Clear images directory
    if os.path.exists("data/images"):
        shutil.rmtree("data/images")
        os.makedirs("data/images")
    
    # Clear students.json
    if os.path.exists("data/students.json"):
        open("data/students.json", "w").close()
    
    # Clear trained model
    if os.path.exists("models/trained_model.pkl"):
        os.remove("models/trained_model.pkl")
        
    print("Dataset cleared successfully!")

if __name__ == "__main__":
    clear_dataset() 