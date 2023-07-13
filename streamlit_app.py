import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('rock_paper_scissors_cnn.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Read the image file
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    # Resize the image to match the input size of your model
    img = cv2.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    # Expand dimensions to match the shape expected by the model
    img = np.expand_dims(img, axis=0)
    return img

# Function to predict the gesture
def predict_gesture(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Streamlit app
def main():
    # Set page title
    st.title("Image Classification")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Perform prediction if an image is uploaded
    if uploaded_file is not None:
        # Predict gesture
        gesture = predict_gesture(uploaded_file)

        # Read the uploaded file again for display
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Display the uploaded image
        st.image(image, channels="RGB", use_column_width=True)

        # Display the predicted gesture
        if gesture == 0:
            st.write("You made a Rock!")
        elif gesture == 1:
            st.write("You made a Paper!")
        elif gesture == 2:
            st.write("You made Scissors!")

if __name__ == '__main__':
    main()
