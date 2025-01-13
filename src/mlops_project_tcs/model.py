import torch
import torch.nn as nn
import torchvision.models as models

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Classifier, self).__init__()

        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)

        # Freeze the feature extractor layers to prevent updates during training
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Replace the classifier head with a custom head for num_classes
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.vgg16(x)

# Example usage
def main():
    num_classes = 10  # Set this to the number of classes in your dataset
    model = VGG16Classifier(num_classes)

    # Test the model with dummy input
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 224x224 image
    output = model(input_tensor)

    print("Model output shape:", output.shape)  # Should be (1, num_classes)

if __name__ == "__main__":
    main()
