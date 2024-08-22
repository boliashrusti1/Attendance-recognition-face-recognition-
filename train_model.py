#! /usr/bin/python

# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Define the dataset directory
dataset_dir = "dataset"

# Ensure the dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"[ERROR] Dataset directory '{dataset_dir}' does not exist.")
    exit(1)

print("[INFO] start processing faces...")

# Get the image paths
imagePaths = list(paths.list_images(dataset_dir))

# Initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# Loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # Extract the person name from the image path
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]

    # Load the input image and convert it from BGR (OpenCV ordering) to RGB (dlib ordering)
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Could not read image {imagePath}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb, model="hog")

    # Compute the facial embeddings for the faces
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the encodings
    for encoding in encodings:
        # Add each encoding + name to our set of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# Dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] processing complete.")
