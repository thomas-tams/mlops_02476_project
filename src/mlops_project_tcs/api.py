from fastapi import FastAPI
from http import HTTPStatus
from typing import List
import numpy as np
from fastapi import File, UploadFile
import cv2
from mlops_project_tcs.evaluate import ONNXEvaluate
from contextlib import asynccontextmanager
from google.cloud import storage
import os
from PIL import Image


app = FastAPI()


def download_model_from_gcs(bucket_name: str, gcs_path: str, local_path: str):
    """
    Downloads a model from GCS to the local machine.

    Args:
        bucket_name (str): Name of the GCS bucket.
        gcs_path (str): Path to the model in the GCS bucket (e.g., 'models/best_model.onnx').
        local_path (str): Local path to save the model (e.g., 'best_model.onnx').
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Model successfully downloaded to {local_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global onnx_model
    print("Loading model")

    bucket_name = "mlops_dtu_model_onnx"
    model_path = "models/best_model_val_loss_0.5177.onnx"
    if not os.path.exists(model_path):
        download_model_from_gcs(bucket_name=bucket_name, gcs_path=model_path, local_path=model_path)
    onnx_model = ONNXEvaluate(onnx_model_path=model_path)

    yield

    print("Cleaning up")
    del onnx_model


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
async def predict(image_files: List[UploadFile] = File(...)):
    """
    Endpoint to handle multiple image uploads and return predictions for each.

    Args:
        image_files (List[UploadFile]): List of uploaded image files.

    Returns:
        dict: Dictionary containing predictions for each uploaded file.
    """
    predictions = {}

    for image_file in image_files:
        try:
            # Read and process each image file
            image = await image_file.read()
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Evaluate the image using the ONNX model
            res = onnx_model.evaluate_image(image=image)
            res = res.tolist()

            # Store the result in the dictionary
            predictions[image_file.filename] = {
                "result": res,
                "message": HTTPStatus.OK.phrase,
                "status-code": HTTPStatus.OK,
            }
        except Exception as e:
            # Handle errors for individual files
            predictions[image_file.filename] = {
                "result": None,
                "message": str(e),
                "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
            }

    return predictions
