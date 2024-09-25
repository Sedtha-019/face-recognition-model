import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model from the saved file
model = tf.keras.models.load_model('C:/Users/sedth/PycharmProjects/pythonProject/pythonProject/model1.h5')

# Read an image using OpenCV
img = cv2.imread('D:/Screenshot 2024-09-09 160343.png')

# Convert the image from BGR (OpenCV format) to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib with a binary color map
plt.imshow(img, cmap=plt.cm.binary)

# Resize the image to match the input size of the model and normalize it
img_resized = cv2.resize(img, (300, 300))  # Resize the image to 300x300
img_array = np.array([img_resized]) / 255.0  # Normalize the image

# Make a prediction using the loaded model
prediction = model.predict(img_array)

# Get the index of the predicted class (class with the highest probability)
index = np.argmax(prediction)

# Define the class names (modify this list based on your dataset)
class_names = ['Rak sa', 'Sed']  # Replace with actual class names

# Print the predicted class
print(f'Prediction is {class_names[index]}')

# Show the image
plt.show()