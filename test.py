import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture(3)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()
