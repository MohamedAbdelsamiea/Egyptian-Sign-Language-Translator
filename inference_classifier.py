import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Load the trained Random Forest classifier model from a pickled data file
model_dict = pickle.load(open('./model.p', 'rb'))
rf_model = model_dict['model']

# Initialize a MediaPipe Hands object for hand landmark extraction
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define a dictionary to map classifier output values to their corresponding labels
label_dict = {0: 'أ', 1: 'ب', 2: 'ت', 3: 'ث', 4: 'ج', 5: 'ح',
               6: 'خ', 7: 'د', 8: 'ذ', 9: 'ر',10: 'ز', 11: 'س',
               12: 'ش', 13: 'ص', 14: 'ض', 15: 'ط', 16: 'ظ', 17: 'ع',
               18: 'غ', 19: 'ف', 20: 'ق', 21: 'ك', 22: 'ل',
               23: 'م', 24: 'ن', 25: 'ه', 26: 'و', 27: 'ي', 28: 'ي'}

# Initialize variables for word and sentence
word = ''
sentence = ''

# Define a function to add the predicted character to the word
def add_char():
    global word
    global label_var
    if label_var.get():
        word += label_var.get()
        label_var.set('')
        word_label.config(text=word)

# Define a function to delete the last character from the word
def delete_char():
    global word
    word = word[:-1]
    word_label.config(text=word)

# Define a function to add the current word to the sentence
def add_word():
    global word
    global sentence
    if word:
        sentence += word + ' '
        word = ''
        word_label.config(text=word)
        sentence_label.config(text=sentence)

# Define a function to delete the last word from the sentence
def delete_word():
    global sentence
    words = sentence.split()
    if words:
        sentence = ' '.join(words[:-1]) + ' '
        sentence_label.config(text=sentence)

# Define a function to copy the current word or sentence to the clipboard
def copy_to_clipboard():
    global word
    global sentence
    if word:
        root.clipboard_clear()
        root.clipboard_append(word)
        root.update()
    elif sentence:
        root.clipboard_clear()
        root.clipboard_append(sentence)
        root.update()

# Initialize the GUI
root = tk.Tk()
root.title('Hand Gesture Typing')

# Create a frame for the video stream
video_frame = tk.Frame(root)
video_frame.pack(side=tk.LEFT)

# Create a label for the word
word_label = tk.Label(root, text=word, font=('Arial', 26), fg='blue')
word_label.pack(side=tk.TOP, pady=10)

# Create a label for the sentence
sentence_label = tk.Label(root, text=sentence, font=('Arial', 26), fg='black')
sentence_label.pack(side=tk.TOP, pady=10)

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.LEFT, padx=20)

# Create a label for the predicted character
label_var = tk.StringVar()
label_var.set('')
predicted_label = tk.Label(button_frame, textvariable=label_var, font=('Arial', 26), fg='green')
predicted_label.pack(side=tk.TOP, pady=10)

# Create buttons for adding and deleting characters, words, and sentences, and copying to clipboard
add_char_button = tk.Button(button_frame, text='Add Character', command=add_char)
add_char_button.pack(side=tk.TOP, pady=10)

delete_char_button = tk.Button(button_frame, text='Delete Character', command=delete_char)
delete_char_button.pack(side=tk.TOP, pady=10)

add_word_button = tk.Button(button_frame, text='Add Word', command=add_word)
add_word_button.pack(side=tk.TOP, pady=10)

delete_word_button = tk.Button(button_frame, text='Delete Word', command=delete_word)
delete_word_button.pack(side=tk.TOP, pady=10)

copy_button = tk.Button(button_frame, text='Copy', command=copy_to_clipboard)
copy_button.pack(side=tk.TOP, pady=10)

# Create a label for the video stream
video_label = tk.Label(video_frame)
video_label.pack()

# Initialize a video capture object using the default camera
cap = cv2.VideoCapture(0)

# Define a function to update the GUI
def update_gui():
    global word
    global sentence

    # Create an empty list to store the extracted x,y coordinates for each landmark
    data = []
    x_coords = []
    y_coords = []

    # Capture a frame from the video stream
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Get the height and width of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB format using OpenCV
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the MediaPipe Hands object to extract hand landmarks from the frame
    results = hands.process(frame_rgb)

    # If hand landmarks are detected in the frame, draw them and extract their x,y coordinates
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_coords.append(x)
                y_coords.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                # Normalize the x,y coordinates by subtracting the minimum value for each axis
                data.append(x - min(x_coords))
                data.append(y - min(y_coords))

        # Calculate the bounding box coordinates for the hand
        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10
        x2 = int(max(x_coords) * W) - 10
        y2 = int(max(y_coords) * H) - 10

        # Use the trained Random Forest classifier model to predict the gesture being made based on the extracted landmark data
        prediction = rf_model.predict([np.array(data)])

        # Update the predicted label in the GUI
        label_var.set(label_dict[int(prediction[0])])

    # Convert the frame to PIL format and resize it to fit the video label
    frame_pil = Image.fromarray(frame_rgb)
    frame_pil = frame_pil.resize((640, 480), Image.ANTIALIAS)

    # Convert the PIL image to Tkinter format and display it in the video label
    video_img = ImageTk.PhotoImage(image=frame_pil)
    video_label.config(image=video_img)
    video_label.image = video_img

    # Call the update_gui function again after 10 milliseconds
    root.after(10, update_gui)

# Start the GUI update loop
update_gui()

# Start the Tkinter event loop
root.mainloop()
