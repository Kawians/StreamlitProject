import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model("rock_paper_scissors_cnn.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to predict the gesture
def predict_gesture(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Streamlit app
def main():
    st.title("Rock Paper Scissors")

    # Camera setup
    camera = cv2.VideoCapture(0)

    if st.checkbox("Activate Camera"):
        st.write("Camera is active.")
        st.write("Get ready to make a gesture.")

        # Capture image
        if st.button("Capture Image"):
            _, frame = camera.read()

            # Display the captured image
            st.image(frame, channels="BGR")

            # Convert image to grayscale
            grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Predict gesture
            gesture = predict_gesture(grayscale_image)
            if gesture == 0:
                st.write("You made a Rock!")
            elif gesture == 1:
                st.write("You made a Paper!")
            elif gesture == 2:
                st.write("You made Scissors!")

    # Release the camera
    camera.release()

if __name__ == "__main__":
    main()
