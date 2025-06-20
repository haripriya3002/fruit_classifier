import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ✅ Load trained model (.keras format recommended)
model = load_model("model/fruit_model.keras")
  # <- updated here

# Define class names (must match training folders)
class_names = ['apple', 'banana', 'orange']

# Streamlit UI
st.set_page_config(page_title="Fruit Classifier", layout="centered")
st.title(" Fruit Image Classifier")
st.write("Upload an image of an **apple🍎**, **banana🍌** or **orange🍊**.")

# Upload image
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    img = img.resize((224, 224))  # MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display results
    st.markdown(f"### ✅ Prediction: **{predicted_class.capitalize()}**")
    st.markdown(f"**Confidence:** {confidence * 100:.2f}%")

    # st.subheader("Class Probabilities:")
    # for i, prob in enumerate(predictions[0]):
    #     st.write(f"{class_names[i].capitalize()}: {prob * 100:.2f}%")
