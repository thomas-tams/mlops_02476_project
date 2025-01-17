import pytest
from unittest.mock import patch, MagicMock
import shutil

from tests import _PATH_DATA, _TEST_ROOT
from mlops_project_tcs.data import balance, split, augment, preprocess
from mlops_project_tcs.train import train_model
from hydra import initialize, compose

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


@pytest.fixture
def mock_hydra_config():
    with patch("hydra.core.hydra_config.HydraConfig.get") as mock_get:
        mock_get.return_value.runtime.output_dir = "./tests/mock_output"
        yield mock_get


@pytest.fixture
def mock_trainer():
    with patch("pytorch_lightning.Trainer") as mock_trainer:
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        yield mock_trainer_instance


@pytest.fixture
def mock_wandb_logger():
    with patch("pytorch_lightning.loggers.WandbLogger") as mock_logger:
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        yield mock_logger_instance


def test_train_model(setup_dummy_data, mock_hydra_config, mock_trainer, mock_wandb_logger):
    """Test the train_model function."""
    # Adding mocks to avoid writing / sending / receiving unnessecary data
    with patch("mlops_project_tcs.train.get_accelerator", return_value="cpu"):
        with patch("mlops_project_tcs.train.pl.seed_everything"):
            with patch("mlops_project_tcs.train.logger.add"):
                balance(RAW_DATA_PATH)
                split()
                augment()
                preprocess()

                with initialize(config_path="../src/mlops_project_tcs/config", version_base="1.3"):
                    config = compose(config_name="test_config.yaml")
                train_model(config)
