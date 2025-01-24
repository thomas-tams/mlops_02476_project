import os
import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2
from PIL import Image
import io
from typing import Optional, Dict, Any


@st.cache_resource
def get_backend_url() -> Optional[str]:
    """
    Get the URL of the backend service.

    Returns:
        Optional[str]: URL of the backend service or None if not found.
    """
    parent = "projects/our-brand-447716-f1/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "mlops-api":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def classify_image(files: Dict[str, Any], backend: str) -> Optional[Dict[str, Any]]:
    """
    Send the image to the backend for classification.

    Args:
        files (Dict[str, Any]): Dictionary containing the image file.
        backend (str): URL of the backend service.

    Returns:
        Optional[Dict[str, Any]]: JSON response from the backend or None if the request failed.
    """
    predict_url = f"{backend}/predict/"
    response = requests.post(predict_url, files=files, timeout=60)
    if response.status_code == 200:
        return response.json()
    return None


def preprocess_image(files: Dict[str, Any], backend: str) -> Optional[Image.Image]:
    """
    Send the image to the backend for preprocessing.

    Args:
        files (Dict[str, Any]): Dictionary containing the image file.
        backend (str): URL of the backend service.

    Returns:
        Optional[Image.Image]: Preprocessed image or None if the request failed.
    """
    predict_url = f"{backend}/preprocess/"
    response = requests.post(predict_url, files=files, timeout=10)
    if response.status_code == 200:
        preprocessed_image = Image.open(io.BytesIO(response.content))
        return preprocessed_image
    st.error("Error: " + response.text)
    return None


def main() -> None:
    """
    Main function of the Streamlit frontend.
    """
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service not found"
        raise ValueError(msg)

    st.title("Image Classification")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and uploaded_file.type == "image/png":
        image = Image.open(uploaded_file)
        converted_image = io.BytesIO()
        image.save(converted_image, format="JPEG")
        uploaded_file = converted_image
        uploaded_file.name = uploaded_file.name.replace(".png", ".jpg")

    if uploaded_file is not None:
        image = uploaded_file.read()
        files = {"file": (uploaded_file.getvalue())}
        result = classify_image(files=files, backend=backend)

        if result is not None:
            prediction = result["prediction"]
            probabilities = result["probabilities"]

            # show the image and prediction
            st.image(image, caption="Uploaded Image")
            st.write("Prediction:", prediction)

            # show preprocessed image (input for the model)
            preprocessed_image = preprocess_image(files=files, backend=backend)
            print(type(preprocessed_image))
            if preprocessed_image is not None:
                st.image(preprocessed_image, caption="Preprocessed image (input for the model)")

            # Create a bar chart of prediction probabilities
            data = {"Class": ["No cancer", "Cancer"], "Probability": probabilities[0]}
            print(data)
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
