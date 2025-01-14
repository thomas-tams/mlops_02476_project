import pytest
from pathlib import Path
import shutil
from src.mlops_project_tcs.data import download

from tests import _PATH_DATA

# Define the paths to the raw and processed data
RAW_DATA_PATH = _PATH_DATA / "raw"


@pytest.fixture
def setup_cleaning_fixture():
    # Create raw data directories
    for category in ["yes", "no"]:
        category_path = RAW_DATA_PATH / category
        category_path.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup after tests
    for category in ["yes", "no"]:
        category_path = RAW_DATA_PATH / category
        shutil.rmtree(category_path)


@pytest.mark.parametrize("category", ["yes", "no"])
def test_raw_data_loading(setup_cleaning_fixture, category):
    """Test that raw data is downloaded."""
    download()
    category_path = RAW_DATA_PATH / category
    assert category_path.exists() and category_path.is_dir(), f"Raw data path for {category} does not exist."
    image_files = [
        f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG"}
    ]
    assert len(image_files) > 0, f"No images found in raw data for {category}."
