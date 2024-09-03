import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt

# Define the paths to validation and test data
valid_path = 'C:/Users/vkesh/Downloads/currency/vaid/validation'
test_path = 'C:/Users/vkesh/Downloads/currency/test/test1'

# Check if the validation and test data exists
print("Validation Data Exists:", os.path.exists(valid_path))
print("Test Data Exists:", os.path.exists(test_path))

# Define image dimensions
IMAGE_SIZE = (224, 224)  # Standard image size for MobileNetV2
BATCH_SIZE = 16

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
    
)

# Load the pre-trained model
model = load_model('C:/Users/vkesh/Downloads/currency/model_Classifier.h5')

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Visualize some sample predictions
test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)
class_labels = test_generator.class_indices
inv_class_labels = {v: k for k, v in class_labels.items()}

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(test_images[i])
    true_label = inv_class_labels[np.argmax(test_labels[i])]
    predicted_label = inv_class_labels[np.argmax(predictions[i])]
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
    plt.axis('off')
plt.show()
