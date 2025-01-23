from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from http import HTTPStatus
from mlops_project_tcs.evaluate import ONNXEvaluate
from contextlib import asynccontextmanager
from torchvision import transforms
import os
import anyio
import json
from PIL import Image
import torch
from pathlib import Path
import sys
import io
from typing import AsyncGenerator, Dict, Tuple, Union


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load and clean up model on startup and shutdown."""
    global onnx_model, classes
    print("Loading model")

    model_mount_path = Path(os.environ.get("MODEL_MOUNT_PATH", "/mnt/models"))
    model_path = model_mount_path / "models/best_model_val_loss_0.5177.onnx"

    if model_path.exists():
        onnx_model = ONNXEvaluate(onnx_model_path=model_path)
    else:
        print(f"Model path: {model_path} not present")
        sys.exit(1)

    async with await anyio.open_file("configs/predict_labels.json", "r") as f:
        content = await f.read()
        classes = json.loads(content)

    yield

    print("Cleaning up")
    del onnx_model, classes


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root() -> Dict[str, str]:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def predict_image(image_path: str) -> Tuple[torch.Tensor, str]:
    """Predict image class (or classes) given image path and return the result."""
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        output, cropped_image = onnx_model.evaluate_image(img)
    _, predicted_idx = torch.max(output, 1)
    return output, classes[str(predicted_idx.item())]


@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)) -> Dict[str, Union[str, float, list]]:
    """Classify image endpoint."""
    file_path = Path("data") / file.filename
    contents = await file.read()
    async with await anyio.open_file(file_path, "wb") as f:
        await f.write(contents)
    probabilities, prediction = predict_image(file_path)
    return {"file_path": str(file_path), "prediction": prediction, "probabilities": probabilities.tolist()}


@app.post("/preprocess/")
async def preprocess_image(file: UploadFile = File(...)) -> StreamingResponse:
    """Preprocess image and return the image."""
    try:
        # Open the uploaded image
        image = Image.open(io.BytesIO(await file.read()))

        tensor_image = onnx_model._preprocess_image(image)
        preprocessed_image = transforms.ToPILImage()(tensor_image)

        # Save the cropped image to a byte stream
        byte_stream = io.BytesIO()
        preprocessed_image.save(byte_stream, format=image.format)
        byte_stream.seek(0)

        # Return the cropped image
        return StreamingResponse(byte_stream, media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}
