import pytest
from PIL import Image
import shutil
from tests import _PATH_DATA, _TEST_ROOT
from mlops_project_tcs.data import MyDataset

RAW_DATA_PATH = _PATH_DATA / "raw"
PROCESSED_DATA_PATH = _PATH_DATA / "processed"
TEST_IMAGE_PATH = _TEST_ROOT / "resources" / "dummy_image.jpg"


@pytest.fixture
def setup_dummy_data():
    # Create raw data directories
    for category in ["yes", "no"]:
        category_path = RAW_DATA_PATH / category
        category_path.mkdir(parents=True, exist_ok=True)
        # Create dummy image files
        dummy_image_path = category_path / "dummy_image.jpg"
        shutil.copy(TEST_IMAGE_PATH, dummy_image_path)

    yield
    # Cleanup after tests
    for data_dir in [RAW_DATA_PATH, PROCESSED_DATA_PATH]:
        for category in ["yes", "no"]:
            category_path = data_dir / category
            shutil.rmtree(category_path)


@pytest.mark.parametrize("category", ["yes", "no"])
def test_data_preprocessing(setup_dummy_data, category):
    """Test that raw data is loaded correctly."""
    dataset = MyDataset(RAW_DATA_PATH)
    dataset.preprocess(PROCESSED_DATA_PATH)
    category_path = PROCESSED_DATA_PATH / category
    assert category_path.exists() and category_path.is_dir(), f"Processed data path for {category} does not exist."
    image_files = [
        f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG"}
    ]
    assert len(image_files) > 0, f"No images found in processed data for {category}."


@pytest.mark.parametrize("category", ["yes", "no"])
def test_processed_images_format(setup_dummy_data, category):
    """Test that processed images are in the correct format and dimensions."""
    dataset = MyDataset(RAW_DATA_PATH)
    dataset.preprocess(PROCESSED_DATA_PATH)
    category_path = PROCESSED_DATA_PATH / category
    image_files = [
        f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG"}
    ]

    for img_file in image_files:
        img_path = category_path / img_file
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify the image is valid

                # Check the image mode
                assert img.mode == "RGB", f"Image {img_file} has unexpected mode {img.mode}."

                # Reopen the image to access its size (verify closes the file)
                img = Image.open(img_path)
                assert img.size == (224, 224), f"Image {img_file} has unexpected size {img.size}. Expected (224, 224)"
        except Exception as e:
            pytest.fail(f"Image {img_file} could not be opened or is invalid: {e}")
