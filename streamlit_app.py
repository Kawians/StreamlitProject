import streamlit as st
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model("rps_cnn.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = tf.image.resize(image, (150, 150))
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
    camera_active = st.checkbox("Activate Camera")

    if camera_active:
        st.write("Camera is active.")
        st.write("Get ready to make a gesture.")

        # Capture image after 3 seconds
        for i in range(3, 0, -1):
            st.write(f"Capturing image in {i} seconds...")
            time.sleep(1)

        frame = st.camera_input()

        # Display the captured image
        st.image(frame, channels="RGB")

        # Convert image to numpy array
        image_np = np.array(frame)

        # Predict gesture
        gesture = predict_gesture(image_np)
        if gesture == 0:
            st.write("You made a Rock!")
        elif gesture == 1:
            st.write("You made a Paper!")
        elif gesture == 2:
            st.write("You made Scissors!")

if _name_ == "_main_":
    main()
