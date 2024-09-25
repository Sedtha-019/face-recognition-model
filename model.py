import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.compat.tensorflow_stub.tensor_shape import matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import os
import warnings
import numpy as np

# Ignore any warnings for cleaner output
warnings.filterwarnings("ignore")

# Define the paths to training, validation, and test datasets
train_data = os.path.join("D:/Self_Program/D_Program/ML/Project_github/Face_recognise/train")
test_data = os.path.join("D:/Self_Program/D_Program/ML/Project_github/Face_recognise/test")
val_data = os.path.join("D:/Self_Program/D_Program/ML/Project_github/Face_recognise/Validation")

# Initialize data generators to load images and apply rescaling
train = ImageDataGenerator(rescale=1./255)
test = ImageDataGenerator(rescale=1./255)
Val = ImageDataGenerator(rescale=1./255)

# Load the training and validation data from their respective directories
model_train = train.flow_from_directory(train_data,
                                         target_size=(300, 300),  # Resize images to 300x300
                                         batch_size=2,  # Process 2 images at a time
                                         class_mode='categorical'  # Use categorical class labels
                                         )
model_Val = train.flow_from_directory(val_data,
                                         target_size=(300, 300),
                                         batch_size=2,
                                         class_mode='categorical'
                                         )

# Define a Sequential model with several Conv2D and MaxPooling layers
md = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),  # 16 filters, 3x3 kernel
    tf.keras.layers.MaxPool2D(2,2),  # Max pooling with 2x2 window

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),  # 32 filters, 3x3 kernel
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  # 64 filters, 3x3 kernel
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),  # Flatten the output for the dense layers

    tf.keras.layers.Dense(512, activation='relu'),  # Fully connected layer with 512 units
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer with 2 classes, using softmax activation
])

# Compile the model with categorical crossentropy loss and RMSprop optimizer
md.compile(
    loss='categorical_crossentropy',  # Suitable for multi-class classification
    optimizer=RMSprop(learning_rate=0.001),  # RMSprop with a learning rate of 0.001
    metrics=['accuracy']  # Track accuracy during training
)

# Train the model with the training and validation datasets
md.fit(model_train,
    steps_per_epoch=2,  # Run 2 batches per epoch
    epochs=200,  # Train for 200 epochs
    validation_data=model_Val,
    validation_steps=2  # Run 2 validation steps per epoch
)

# Uncomment the line below to save the trained model to a file
# md.save('pythonProject/model1.h5')

