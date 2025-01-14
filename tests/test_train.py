import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

from tests import _PROJECT_ROOT
from src.mlops_project_tcs.data import download, preprocess
from src.mlops_project_tcs.train import train_model
from hydra import initialize, compose

_TEST_HYDRA_CONFIG = _PROJECT_ROOT / "src" / "mlops_project_tcs" / "config" / "test_config.yaml"


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
def mock_vgg16_classifier():
    with patch("src.mlops_project_tcs.train.VGG16Classifier") as mock_model:
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        yield mock_model_instance


@pytest.fixture
def mock_wandb_logger():
    with patch("pytorch_lightning.loggers.WandbLogger") as mock_logger:
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        yield mock_logger_instance


def test_train_model(mock_hydra_config, mock_trainer, mock_vgg16_classifier, mock_wandb_logger):
    """Test the train_model function."""
    # Adding mocks to avoid writing / sending / receiving unnessecary data
    with patch("src.mlops_project_tcs.train.get_accelerator", return_value="cpu"):
        with patch("src.mlops_project_tcs.train.pl.seed_everything"):
            with patch("src.mlops_project_tcs.train.logger.add"):
                download()
                preprocess()

                with initialize(config_path="../src/mlops_project_tcs/config", version_base="1.3"):
                    config = compose(config_name="test_config.yaml")
                train_model(config)
