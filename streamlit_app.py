import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('rock_paper_scissors_cnn.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of your model
    resized_image = cv2.resize(image, (128, 128))
    # Normalize the image
    normalized_image = array_image / 255.0
    return input_image
    
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
        # Read the uploaded file
        #image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        image = preprocess_image(uploaded_file)
        # Display the uploaded image
        st.image(image, channels="RGB", use_column_width=True)

        # Predict gesture
        gesture = predict_gesture(image)
        if gesture == 0:
            st.write("You made a Rock!")
        elif gesture == 1:
            st.write("You made a Paper!")
        elif gesture == 2:
            st.write("You made Scissors!")

if __name__ == '__main__':
    main()
