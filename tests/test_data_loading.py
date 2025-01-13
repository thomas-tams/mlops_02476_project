import os
import pytest

# Define the paths to the raw and processed data
RAW_DATA_PATH = "./data/raw"
PROCESSED_DATA_PATH = "./data/processed"

@pytest.mark.parametrize("category", ["yes", "no"])
def test_raw_data_loading(category):
    """Test that raw data is loaded correctly."""
    category_path = os.path.join(RAW_DATA_PATH, category)
    assert os.path.exists(category_path), f"Raw data path for {category} does not exist."
    image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    assert len(image_files) > 0, f"No images found in raw data for {category}."

@pytest.mark.parametrize("category", ["yes", "no"])
def test_processed_data_loading(category):
    """Test that processed data is loaded correctly."""
    category_path = os.path.join(PROCESSED_DATA_PATH, category)
    assert os.path.exists(category_path), f"Processed data path for {category} does not exist."
    image_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    assert len(image_files) > 0, f"No images found in processed data for {category}."
