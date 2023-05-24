# Import necessary libraries
import os
import pickle
import mediapipe as mp
import cv2

# Initialize the MediaPipe Hands object for landmark extraction
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the image data is stored
IMAGE_DIR = './images'

# Create empty lists to store the extracted landmark data and corresponding labels
landmark_data = []
labels = []

# Loop through each subdirectory in the image directory
for subdir_name in os.listdir(IMAGE_DIR):
    subdir_path = os.path.join(IMAGE_DIR, subdir_name)
    # Loop through each image in the subdirectory
    for img_name in os.listdir(subdir_path):
        # Create an empty list to store the extracted x,y coordinates for each landmark
        coords = []
        # Load the image and convert it to RGB format
        img_path = os.path.join(subdir_path, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Use the MediaPipe Hands object to extract hand landmarks from the image
        results= hands.process(img_rgb)
        # If landmarks are detected in the image, extract the x,y coordinates for each landmark
        if results.multi_hand_landmarks:
            # Create empty lists to store the x,y coordinates for all landmarks and the minimum x,y values
            x_coords = []
            y_coords = []
            # Loop through each detected hand landmark
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Append the x,y coordinates to the respective lists
                    x_coords.append(x)
                    y_coords.append(y)
            # Normalize the x,y coordinates by subtracting the minimum value for each axis
            x_min = min(x_coords)
            y_min = min(y_coords)
            for i in range(len(x_coords)):
                x_norm = x_coords[i] - x_min
                y_norm = y_coords[i] - y_min
                coords.append(x_norm)
                coords.append(y_norm)
            # Append the extracted landmark data and corresponding label to the respective lists
            landmark_data.append(coords)
            labels.append(subdir_name)

# Save the extracted landmark data and corresponding labels to a pickled data file
data_file = open('data.pickle', 'wb')
pickle.dump({'data': landmark_data, 'labels': labels}, data_file)
data_file.close()
