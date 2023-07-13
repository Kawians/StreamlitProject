import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('rock_paper_scissors_cnn.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of your model
    resized_image = cv2.resize(image, (224, 224))
    # Convert the image to a numpy array
    array_image = np.array(resized_image)
    # Normalize the image
    normalized_image = array_image / 255.0
    # Expand dimensions to match the input shape of your model
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Function to make predictions
def make_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

# Streamlit app
def main():
    # Set page title
    st.title("Image Classification")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Perform prediction if an image is uploaded
    if uploaded_file is not None:
        # Read the uploaded file
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Display the uploaded image
        st.image(image, channels="RGB", use_column_width=True)

        # Make prediction on the uploaded image
        prediction = make_prediction(image)
        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
