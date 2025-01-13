import torch
from src.mlops_project_tcs.model import VGG16Classifier

def test_vgg16_classifier_output_shape():
    """Test that VGG16Classifier produces the correct output shape."""
    num_classes = 2  # Binary classification (yes/no)
    model = VGG16Classifier(num_classes=num_classes)

    # Create a dummy input tensor of shape (1, 3, 224, 224)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Pass the input through the model
    output = model(input_tensor)

    # Check that the output shape is (1, num_classes)
    assert output.shape == (1, num_classes), f"Unexpected output shape: {output.shape}, expected: (1, {num_classes})"
