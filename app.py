import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Model
model = tf.keras.models.load_model(
    "scene_classifier.keras"
)

# Class Names
class_names = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

st.title("🌍 Intel Natural Scene Classification")

st.write(
    "Upload an image and the model will predict the scene category."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image and convert to RGB
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Resize image
    #img = image.resize((224,224))
    img = image.resize((150,150))

    # Convert to numpy array
    img = np.array(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # MobileNetV2 preprocessing
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2f}%"
    )