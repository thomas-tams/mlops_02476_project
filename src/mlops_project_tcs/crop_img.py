import numpy as np
import torch
from PIL import Image
import cv2
import imutils
from typing import Union, Tuple


class CropExtremePoints:
    def __init__(self, add_pixels_value: int = 0, target_size: Tuple[int, int] = (224, 224)) -> None:
        """
        Initialize the CropExtremePoints class.

        Args:
            add_pixels_value (int): Number of pixels to expand the crop around the extreme points.
            target_size (tuple): Target size for the output image after cropping and resizing (default is 224x224).
        """
        self.add_pixels_value = add_pixels_value
        self.target_size = target_size

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Perform the cropping and resizing operation.

        Args:
            img (PIL Image, ndarray, or torch.Tensor): Input image.

        Returns:
            torch.Tensor: Cropped and resized image.
        """
        # PIL image -> numpy array
        if isinstance(img, Image.Image):
            img = np.array(img)

        # Tensor -> numpy array
        # (C, H, W) -> (H, W, C)
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()

        if isinstance(img, np.ndarray) and img.ndim == 3:
            # Ensures 3 color channels
            if img.shape[2] != 3:
                raise ValueError("Image must have 3 channels (RGB).")

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # Threshold and remove noise
            thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if not cnts:
                raise ValueError("No contours found in the image.")

            # Get the largest contour
            c = max(cnts, key=cv2.contourArea)

            # Find extreme points
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            # Apply cropping
            ADD_PIXELS = self.add_pixels_value
            cropped_img = img[
                max(0, extTop[1] - ADD_PIXELS) : min(img.shape[0], extBot[1] + ADD_PIXELS),
                max(0, extLeft[0] - ADD_PIXELS) : min(img.shape[1], extRight[0] + ADD_PIXELS),
            ]

            # Resize cropped image to the target size (224x224)
            cropped_img_resized = cv2.resize(cropped_img, self.target_size)

            # Convert cropped and resized image back to tensor
            # (H,W,C) -> (C, H, W)
            cropped_img_resized = torch.from_numpy(cropped_img_resized).permute(2, 0, 1)

            return cropped_img_resized
        else:
            raise ValueError("Input image must be a 3D ndarray or a Tensor with 3 channels.")
