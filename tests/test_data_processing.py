import pytest
from PIL import Image
import shutil
from tests import _PATH_DATA, _TEST_ROOT
from mlops_project_tcs.data import balance, split, augment, preprocess

RAW_DATA_PATH = _PATH_DATA / "raw"
PROCESSED_DATA_PATH = _PATH_DATA / "processed"
DUMMY_IMAGES_PATH = _TEST_ROOT / "resources" / "dummy_images"


@pytest.fixture
def setup_dummy_data():
    # Create raw data directories
    for category in ["yes", "no"]:
        category_path = RAW_DATA_PATH / category
        category_path.mkdir(parents=True, exist_ok=True)
        for item in (DUMMY_IMAGES_PATH / category).iterdir():
            shutil.copy(item, category_path / item.name)

    yield
    # Cleanup after tests
    shutil.rmtree(_PATH_DATA)


@pytest.mark.parametrize("category", ["yes", "no"])
def test_data_preprocessing(setup_dummy_data, category):
    """Tests the data preprocessing pipeline"""
    balance(RAW_DATA_PATH)
    split()
    augment()
    preprocess()

    for _split in ["train", "val", "test"]:
        category_path = PROCESSED_DATA_PATH / _split / category
        assert category_path.exists() and category_path.is_dir(), f"Processed data path for {category} does not exist."
        image_files = [
            f for f in category_path.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPG"}
        ]
        assert len(image_files) > 0, f"No images found in processed data for '{_split} {category}'."

        # Test that processed images are in the correct format and dimensions
        for img_file in image_files:
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Verify the image is valid

                    # Check the image mode
                    assert img.mode == "RGB", f"Image {img_file} has unexpected mode {img.mode}."

                    # Reopen the image to access its size (verify closes the file)
                    img = Image.open(img_file)
                    assert img.size == (224, 224), (
                        f"Image {img_file} has unexpected size {img.size}. Expected (224, 224)"
                    )
            except Exception as e:
                pytest.fail(f"Image {img_file} could not be opened or is invalid: {e}")
