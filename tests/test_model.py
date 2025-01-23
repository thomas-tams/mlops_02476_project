import torch
from mlops_project_tcs.model import VGG16Classifier
import hydra


def test_vgg16_classifier_output_shape():
    """Test that VGG16Classifier produces the correct output shape."""
    with hydra.initialize(config_path="../src/mlops_project_tcs/config", version_base="1.3"):
        config = hydra.compose(config_name="test_config.yaml")

    model = VGG16Classifier(
        input_size=config.experiment.model["input_size"],
        hidden_size=config.experiment.model["hidden_size"],
        num_classes=config.experiment.dataset["num_classes"],
        dropout_p=config.experiment.model["dropout_p"],
        criterion=hydra.utils.instantiate(config.experiment.model.loss_fn),
    )

    model.configure_optimizers(
        optimizer=hydra.utils.instantiate(config.experiment.hyperparameter.optimizer, params=model.parameters())
    )

    # Create a dummy input tensor of shape (4, 3, 224, 224)
    input_tensor = torch.randn(4, 3, 224, 224)

    # Pass the input through the model
    output = model(input_tensor)

    # Check that the output shape is (4, num_classes)
    assert output.shape == (
        4,
        config.experiment.dataset["num_classes"],
    ), f"Unexpected output shape: {output.shape}, expected: (4, {config.experiment.dataset['num_classes']})"
