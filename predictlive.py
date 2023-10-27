import cv2
import numpy as np
from keras.models import load_model

# Load the trained CNN model
model = load_model('C:/Users/Asus/OneDrive/Desktop/Distracted drivers/mymodel.h5')

# Define the class labels
class_labels = [
    "Safe driving",
    "Texting - right hand",
    "Talking on the phone - right hand",
    "Texting - left hand",
    "Talking on the phone - left hand",
    "Operating the radio",
    "Drinking",
    "Reaching behind",
    "Hair and makeup",
    "Talking to passenger"
]

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier("C:/Users/Asus/OneDrive/Desktop/Distracted drivers/haarcascade_frontalface_default.xml")

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

while True:
    # Read the video stream frame by frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face_roi = gray[y:y+h, x:x+w]

        # Resize the face ROI to match the input size of the model
        face_roi = cv2.resize(face_roi, (224, 224))

        # Convert grayscale image to RGB
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

         # Normalize the face ROI
        face_roi = face_roi / 255.0

        # Reshape the face ROI to match the input shape of the model
        face_roi = np.reshape(face_roi, (1, 224, 224, 3))

        # Perform the prediction using the trained model
        prediction = model.predict(face_roi)
        label_index = np.argmax(prediction)
        label = class_labels[label_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Write the predicted label above the rectangle
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Distracted Driver Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()

# Destroy all windows
cv2.destroyAllWindows()
