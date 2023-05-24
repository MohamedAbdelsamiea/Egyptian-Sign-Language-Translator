# Import necessary libraries
import os
import cv2

# Define the directory where images will be saved
IMAGE_DIR = './images'
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

# Define the number of categories and the number of images to collect for each category
NUM_CATEGORIES = 3
IMAGES_PER_CATEGORY = 50

# Initialize a video capture object from the default camera
video_capture = cv2.VideoCapture(0)

# Loop through each category
for category in range(NUM_CATEGORIES):
    # Create a subdirectory for the category if it doesn't already exist
    if not os.path.exists(os.path.join(IMAGE_DIR, str(category))):
        os.makedirs(os.path.join(IMAGE_DIR, str(category)))

    # Display a message asking the user to start collecting images for this category
    print('Collecting data for category {}'.format(category))
    done = False
    while not done:
        # Capture a frame from the video stream
        ret, frame = video_capture.read()
        # Display a message asking the user to press a key to start collecting images
        cv2.putText(frame, 'Ready? Press any key to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        # Wait for the user to press a key
        if cv2.waitKey(25) != -1:
            done = True

    # Collect images for this category
    image_count = 0
    while image_count < IMAGES_PER_CATEGORY:
        # Capture a frame from the video stream
        ret, frame = video_capture.read()
        # Display the frame to the user
        cv2.imshow('frame', frame)
        # Wait for a short period of time
        cv2.waitKey(25)
        # Save the image to the appropriate subdirectory
        image_path = os.path.join(IMAGE_DIR, str(category), '{}.jpg'.format(image_count))
        cv2.imwrite(image_path, frame)
        # Increment the image count
        image_count += 1

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
