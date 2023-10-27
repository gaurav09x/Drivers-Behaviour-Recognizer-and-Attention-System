from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
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

from PIL import ImageTk, Image


# Create a function to open a file dialog

def open_file():
    global file_path, img
    file_path = filedialog.askopenfilename()
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = ImageTk.PhotoImage(img)
    img_label.configure(image=img)
    img_label.image = img

def open_file():
    # Ask the user to select an image file
    file_path = filedialog.askopenfilename(initialdir='/', title='Select Image', 
                                           filetypes=(('JPEG files', '*.jpg'), ('PNG files', '*.png')))
    # Load the selected image
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img = ImageTk.PhotoImage(img)

    # Update the image label
    img_label.configure(image=img)
    img_label.image = img
    
    # Prepare the image for prediction
    new_img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
    new_img = tf.keras.utils.img_to_array(new_img)
    new_img = np.expand_dims(new_img, axis=0)
    new_img /= 255.
    
    # Predict the class of the image
    pred = model.predict(new_img)
    pred_class = classes[np.argmax(pred)]

    # Update the result label
    result_label.configure(text=f'Predicted Class: {pred_class}')

# Create a button to open the file dialog
button = Button(root, text='Select Image', command=open_file)
button.pack()

# Create a label to display the image
img_label = Label(root)
img_label.pack()

# Create a label to display the predicted class
result_label = Label(root, font=('Helvetica', 20))
result_label.pack()

# Run the Tkinter mainloop
root.mainloop()
