import pytest
from pathlib import Path

from tests import _PATH_DATA

# Define the paths to the raw and processed data
RAW_DATA_PATH = _PATH_DATA / "raw"
PROCESSED_DATA_PATH = _PATH_DATA / "processed"

@pytest.mark.parametrize("category", ["yes", "no"])
def test_raw_data_loading(category):
    """Test that raw data is loaded correctly."""
    category_path = RAW_DATA_PATH / category
    assert category_path.exists() and category_path.is_dir(), f"Raw data path for {category} does not exist."
    image_files = [f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.JPG'}]
    assert len(image_files) > 0, f"No images found in raw data for {category}."

@pytest.mark.parametrize("category", ["yes", "no"])
def test_processed_data_loading(category):
    """Test that raw data is loaded correctly."""
    category_path = PROCESSED_DATA_PATH / category
    assert category_path.exists() and category_path.is_dir(), f"Raw data path for {category} does not exist."
    image_files = [f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.JPG'}]
    assert len(image_files) > 0, f"No images found in raw data for {category}."
