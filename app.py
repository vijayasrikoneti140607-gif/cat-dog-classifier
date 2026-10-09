import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱"
)

st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload an image and let AI analyze it!")

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(
        weights="imagenet"
    )

model = load_model()

uploaded_file = st.file_uploader(
    "Choose a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image",
             use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Analyzing image..."):
            image = image.resize((224, 224))
            image_array = np.array(image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )

            predictions = model.predict(image_array, verbose=0)
            results = tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions, top=5
            )[0]

            cat_words = [
                "cat", "tabby", "siamese", "persian",
                "egyptian_cat", "tiger_cat"
            ]

            dog_words = [
                "dog", "retriever", "terrier", "shepherd",
                "poodle", "spaniel", "husky", "beagle",
                "rottweiler", "boxer", "bulldog", "chihuahua",
                "pug", "dalmatian", "doberman", "malamute",
                "collie", "schnauzer", "pinscher", "mastiff",
                "hound", "corgi", "samoyed", "whippet"
            ]

            label = results[0][1].lower()

            if any(word in label for word in cat_words):
                st.success("🐱 Prediction: CAT!")
            elif any(word in label for word in dog_words):
                st.success("🐶 Prediction: DOG!")
            else:
                st.warning(
                    "The image may not contain a cat or dog. "
                    f"AI's top prediction: {label.replace('_', ' ')}"
                )

            st.subheader("Top AI predictions")
            for _, name, score in results:
                st.write(
                    f"{name.replace('_', ' ').title()}: "
                    f"{score * 100:.2f}%"
                )
