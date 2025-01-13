import os
import pytest
from PIL import Image

PROCESSED_DATA_PATH = "./data/processed"

@pytest.mark.parametrize("category", ["yes", "no"])
def test_processed_images_format(category):
    """Test that processed images are in the correct format and dimensions."""
    category_path = os.path.join(PROCESSED_DATA_PATH, category)
    image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]

    for img_file in image_files:
        img_path = os.path.join(category_path, img_file)
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify the image is valid

                # Check the image mode
                assert img.mode == "RGB", f"Image {img_file} has unexpected mode {img.mode}."

                # Reopen the image to access its size (verify closes the file)
                img = Image.open(img_path)
                assert img.size == (224, 224), f"Image {img_file} has unexpected size {img.size}."
        except Exception as e:
            pytest.fail(f"Image {img_file} could not be opened or is invalid: {e}")
