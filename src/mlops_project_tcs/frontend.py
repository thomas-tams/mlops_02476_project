import os

import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2
from PIL import Image
import io


@st.cache_resource
def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/our-brand-447716-f1/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    print(services)
    for service in services:
        if service.name.split("/")[-1] == "mlops-api":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name


def classify_image(files, backend):
    """Send the image to the backend for classification."""
    predict_url = f"{backend}/predict/"
    response = requests.post(predict_url, files=files, timeout=60)
    print(response)
    if response.status_code == 200:
        return response.json()
    return None


def preprocess_image(files, backend):
    predict_url = f"{backend}/preprocess/"
    response = requests.post(predict_url, files=files, timeout=10)
    print(response)
    if response.status_code == 200:
        preprocessed_image = Image.open(io.BytesIO(response.content))
        return preprocessed_image
    st.error("Error: " + response.text)
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    backend = get_backend_url()
    backend = "http://127.0.0.1:8000"
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

            # Creat a bar chart of prediction probabilities
            data = {"Class": ["No cancer", "Cancer"], "Probability": probabilities[0]}
            print(data)
            df = pd.DataFrame(data)
            df.set_index("Class", inplace=True)
            st.bar_chart(df, y="Probability")
        else:
            st.write("Failed to get prediction")


if __name__ == "__main__":
    main()
