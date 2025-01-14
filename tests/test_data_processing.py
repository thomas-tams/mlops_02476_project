import pytest
from PIL import Image
from pathlib import Path
import shutil
from src.mlops_project_tcs.data import preprocess, MyDataset

RAW_DATA_PATH = Path("./data/raw")
PROCESSED_DATA_PATH = Path("./data/processed")

@pytest.fixture
def setup_dummy_data():
    # Create raw data directories
    for category in ["yes", "no"]:
        category_path = RAW_DATA_PATH / category
        category_path.mkdir(parents=True, exist_ok=True)
        # Create dummy image files
        dummy_image_path = category_path / "dummy_image.jpg"
        with Image.new('RGB', (100, 100)) as img:
            img.save(dummy_image_path)
    yield
    # Cleanup after tests
    shutil.rmtree(RAW_DATA_PATH)
    shutil.rmtree(PROCESSED_DATA_PATH)

@pytest.mark.parametrize("category", ["yes", "no"])
def test_processed_data_loading(setup_dummy_data, category):
    """Test that raw data is loaded correctly."""
    dataset = MyDataset(RAW_DATA_PATH)
    dataset.preprocess(PROCESSED_DATA_PATH)
    category_path = PROCESSED_DATA_PATH / category
    assert category_path.exists() and category_path.is_dir(), f"Processed data path for {category} does not exist."
    image_files = [f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.JPG'}]
    assert len(image_files) > 0, f"No images found in processed data for {category}."

@pytest.mark.parametrize("category", ["yes", "no"])
def test_processed_images_format(setup_dummy_data, category):
    """Test that processed images are in the correct format and dimensions."""
    dataset = MyDataset(RAW_DATA_PATH)
    dataset.preprocess(PROCESSED_DATA_PATH)
    category_path = PROCESSED_DATA_PATH / category
    image_files = [f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.JPG'}]

    for img_file in image_files:
        img_path = category_path / img_file
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
