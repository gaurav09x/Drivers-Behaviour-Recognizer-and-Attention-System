from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model("C:/Users/Asus/OneDrive/Desktop/Distracted drivers/mymodel.h5")

# Define the class labels
classes = {
    0: 'Safe driving',
    1: 'Texting - right',
    2: 'Talking on the phone - right',
    3: 'Texting - left',
    4: 'Talking on the phone - left',
    5: 'Operating the radio',
    6: 'Drinking',
    7: 'Reaching behind',
    8: 'Hair and makeup',
    9: 'Talking to passenger'
}

# Create a Tkinter window
root = Tk()
root.title('Driver Distraction Detection')
root.geometry('800x600')

# Create a function to open a file dialog
def open_file():
    # Ask the user to select a video file
    file_path = filedialog.askopenfilename(initialdir='/', title='Select Video',
                                           filetypes=(('Video files', '*.mp4'), ('All files', '*.*')))
    
    # Load the video using OpenCV
    video = cv2.VideoCapture(file_path)

    # Define the delay between frames (in milliseconds)
    delay = 50  # Adjust the value to control the playback speed

    # Create a function to process and display the next frame
    def process_next_frame():
        # Read the next frame
        ret, frame = video.read()

        if ret:
            # Preprocess the frame (e.g., resize, normalize)
            processed_frame = preprocess(frame)

            # Predict the class of the frame
            pred_class = predict_class(processed_frame)

            # Display the frame with the predicted class
            display_frame(frame, pred_class)

            # Schedule the next frame processing after the specified delay
            root.after(delay, process_next_frame)
        else:
            # Release the video capture and close the window when the video ends
            video.release()
            cv2.destroyAllWindows()

    # Start processing the first frame
    process_next_frame()

def preprocess(frame):
    # Preprocess the frame (e.g., resize, normalize)
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = processed_frame / 255.0  # Normalize pixel values
    processed_frame = np.expand_dims(processed_frame, axis=0)
    return processed_frame

def predict_class(frame):
    # Predict the class of the frame
    pred = model.predict(frame)
    pred_class = classes[np.argmax(pred)]
    return pred_class

def display_frame(frame, pred_class):
    # Display the frame with the predicted class
    cv2.putText(frame, f'Predicted Class: {pred_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('Driver Distraction Detection', frame)

# Create a button to open the file dialog
button = Button(root, text='Select Video', command=open_file)
button.pack()

# Run the Tkinter mainloop
root.mainloop()
