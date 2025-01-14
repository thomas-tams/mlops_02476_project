import torch
from src.mlops_project_tcs.model import VGG16Classifier
from hydra import initialize, compose
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture
def mock_setup_dataloaders():
    with patch('src.mlops_project_tcs.data.transforms.Compose') as mock_transforms_compose, \
         patch('src.mlops_project_tcs.data.datasets.ImageFolder') as mock_image_folder, \
         patch('src.mlops_project_tcs.data.random_split') as mock_random_split, \
         patch('src.mlops_project_tcs.data.DataLoader') as mock_dataloader:
        
        # Mock the transformations
        mock_transforms_compose.return_value = MagicMock()

        # Mock the ImageFolder dataset
        mock_image_folder_instance = MagicMock()
        mock_image_folder_instance.__len__.return_value = 100
        mock_image_folder.return_value = mock_image_folder_instance

        # Mock the random_split function
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_random_split.return_value = (mock_train_dataset, mock_val_dataset)

        # Mock the DataLoader
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()
        mock_dataloader.side_effect = [mock_train_loader, mock_val_loader]

        yield mock_train_loader, mock_val_loader

def test_vgg16_classifier_output_shape(mock_setup_dataloaders):
    """Test that VGG16Classifier produces the correct output shape."""
    with initialize(config_path="../src/mlops_project_tcs/config", version_base="1.3"):
        config = compose(config_name="test_config.yaml")

    model = VGG16Classifier(config=config)

    # Create a dummy input tensor of shape (4, 3, 224, 224)
    input_tensor = torch.randn(4, 3, 224, 224)

    # Pass the input through the model
    output = model(input_tensor)

    # Check that the output shape is (4, num_classes)
    assert output.shape == (4, config.experiment.dataset["num_classes"]), f"Unexpected output shape: {output.shape}, expected: (4, {config.experiment.dataset['num_classes']})"
