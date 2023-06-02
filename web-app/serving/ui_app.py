import os
import streamlit as st
from PIL import Image

from serving.predictor import Predictor


@st.cache(hash_funcs={Predictor: lambda _: None})
def get_model() -> Predictor:
    return Predictor.default_from_model_registry(os.getenv("MODEL_ID"), os.getenv("MODEL_PATH"))


predictor = get_model()


def prediction():
    st.write("Classes:", predictor.get_classes_config())
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        result = predictor.predict(image)
        st.write("Prediction result:", result)


def main():
    st.header("UWG Garbage Classifier Demo App")
    prediction()


if __name__ == "__main__":
    main()