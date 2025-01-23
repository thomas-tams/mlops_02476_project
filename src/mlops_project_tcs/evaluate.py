import onnxruntime as ort
import numpy as np
from PIL import Image
from mlops_project_tcs.crop_img import CropExtremePoints
import torch
from torchvision import transforms


class ONNXEvaluate:
    def __init__(self, onnx_model_path: str):
        """
        Initializes the ONNXEvaluate class with the path to the ONNX model.

        Args:
            onnx_model_path (str): Path to the ONNX model file.
        """
        self.cropper = CropExtremePoints(add_pixels_value=10, target_size=(224, 224))
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
        self.onnx_model_path = onnx_model_path
        self.session = None
        self._load_model()

    def _load_model(self):
        """
        Loads the ONNX model using ONNX Runtime.
        """
        try:
            self.session = ort.InferenceSession(self.onnx_model_path)
            print("Model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Error loading the ONNX model: {e}")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocesses an image to match the input requirements of the ONNX model.

        Args:
            image (PIL.Image.Image): The input image to be processed.

        Returns:
            torch.Tensor: Cropped and resized image.
        """
        # Crop and resize
        processed_image = self.cropper(image)
        processed_image = processed_image.float()
        processed_image = self.transform(processed_image)
        return processed_image

    def run_inference(self, image: torch.Tensor) -> torch.Tensor:
        """
        Runs inference on the given image using the loaded ONNX model.

        Args:
            image (torch.Tensor): The input image as a numpy array.

        Returns:
            torch.Tensor: Output of model inference.
        """
        if self.session is None:
            raise RuntimeError("Model is not loaded.")

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        # Convert image to batch-like input
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Run inference
        result = self.session.run([output_name], {input_name: image})

        return torch.tensor(result[0])

    def evaluate_image(self, image: Image.Image) -> torch.Tensor:
        """
        Evaluates an image through the model.

        Args:
            image (PIL.Image.Image): The input image to evaluate.

        Returns:
            res (torch.Tensor): The model's inference result.
            image_data (torch.Tensor): Cropped and resized image
        """
        image_data = self._preprocess_image(image)
        res = self.run_inference(image_data)
        return res, image_data
