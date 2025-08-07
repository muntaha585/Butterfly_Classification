import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Define the path to the saved model file
model_filename = 'best_butterfly_model.h5'

# Load the trained model
@st.cache_resource # Cache the model to avoid reloading on each rerun
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model(model_filename)

# Define image dimensions and class names (assuming NUM_CLASSES and class_indices are available from previous steps or can be loaded)
# If not available, you would need to load them or hardcode them if known.
# For this script, we'll assume they are defined or can be inferred.
# In a real-world scenario, you might save these along with the model.
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Assuming class_indices was saved or is available. Let's create a dummy one for demonstration
# Replace this with actual class indices if available
# Example: class_indices = {'species1': 0, 'species2': 1, ...}
# For this example, we'll retrieve from the generator if possible or use a placeholder
# A more robust solution would save/load class_indices.
if 'class_indices' in globals():
    idx_to_class = {v: k for k, v in class_indices.items()}
    species_list = [idx_to_class[i] for i in range(len(idx_to_class))]
elif model is not None and hasattr(model, 'class_names'): # Check if model itself has class names
     species_list = model.class_names
else:
    # Placeholder species list if class_indices is not available and model doesn't have them
    # In a real application, you must ensure class names are loaded correctly.
    st.warning("Class names not found. Using placeholder names. Please ensure 'class_indices' is available or load class names correctly.")
    species_list = [f'class_{i}' for i in range(model.outputs[0].shape[-1])] if model else []


# Set up the Streamlit application title and header
st.title("Butterfly Image Classification")
st.header("Upload an image of a butterfly for classification")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if model is not None:
        # Preprocess the image
        img = image.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Normalize

        # Make a prediction
        predictions = model.predict(img_array)

        # Get the predicted class probabilities
        probabilities = predictions[0]

        # Get the index of the class with the highest probability
        predicted_class_index = np.argmax(probabilities)

        # Get the predicted class name and confidence
        if species_list:
            predicted_class_name = species_list[predicted_class_index]
            confidence = probabilities[predicted_class_index] * 100

            # Display the predicted class and confidence
            st.subheader("Prediction:")
            st.write(f"Predicted Class: **{predicted_class_name}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

            # Optionally, display the top 3 probabilities
            st.subheader("Top 3 Predictions:")
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            for i in top_3_indices:
                class_name = species_list[i]
                prob = probabilities[i] * 100
                st.write(f"- {class_name}: {prob:.2f}%")
        else:
            st.warning("Could not retrieve class names.")