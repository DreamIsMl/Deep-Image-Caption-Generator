import os
import pickle
import string

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Load models and tokenizer
with open('F:\\class\\DeepLearning\\computer vision Project\\Image Caption Generator\\x_model.pkl', 'rb') as f:
    x_model = pickle.load(f)

model = load_model('F:\\class\\DeepLearning\\computer vision Project\\Image Caption Generator\\model')

with open('F:\\class\\DeepLearning\\computer vision Project\\Image Caption Generator\\features (1).pkl', 'rb') as f:
    features = pickle.load(f)

with open('F:\\class\\DeepLearning\\computer vision Project\\Image Caption Generator\\tokenizer (2).pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 35

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Preprocess the image once
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    return preprocess_input(image)

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)

        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Prediction function using preprocessed image
def predict(image_path):
    preprocessed_image = preprocess_image(image_path)
    features = x_model.predict(preprocessed_image, verbose=0)
    result = predict_caption(model, features, tokenizer, max_length)
    
    return result

# Streamlit app
st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Perform prediction when button is clicked
    if st.button("Generate Caption"):
        result = predict(uploaded_file)
        st.success(f"Generated Caption: {result}")
