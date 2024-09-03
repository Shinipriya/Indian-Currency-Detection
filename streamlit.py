"""
import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
import tempfile
import subprocess
import pyttsx3

# Load the trained model
model_path = "model_Classifier.h5"
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(img_array):
    result = model.predict(img_array)
    return result

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("Keshu's Currency Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Perform prediction
    img_array = preprocess_image(uploaded_file)
    prediction = predict_image(img_array)
    currency_labels = ['1Dollar', '1Hundrednote', '1Thousandnote', '20note', '5note', '50note', 'Euro']
    predicted_label = currency_labels[np.argmax(prediction)]

    st.write("Classified Currency:", predicted_label)

    # Convert predicted label to speech
    text_to_speech("The classified currency is " + predicted_label)
"""
import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from keras.preprocessing import image
from keras.models import load_model
import pyttsx3

# Load the trained model
model_path = "C:/Users/vkesh/Downloads/currency/model_Classifier.h5"
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(img_array):
    result = model.predict(img_array)
    return result

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI

st.title("E_EYE -The Currency Classifier")
st.markdown("---")
# Style for title
st.markdown('<style>h1{color: #FF00FF; text-align: center;}</style>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    

    # Perform prediction
    img_array = preprocess_image(uploaded_file)
    prediction = predict_image(img_array)
    currency_labels = ['10note', '20note', '50note', '100note', '200note', '500note', '2000note']
    predicted_label = currency_labels[np.argmax(prediction)]

   # Style for output text
    st.markdown('<p style="text-align:center; font-size:24px; color:#FF00FF;">Classified Currency: {}</p>'.format(predicted_label), unsafe_allow_html=True)
    # Convert predicted label to speech
    text_to_speech(f"The classified currency is {predicted_label}")
