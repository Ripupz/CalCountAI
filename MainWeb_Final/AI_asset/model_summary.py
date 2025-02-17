import torch
from torchsummary import summary
from torchvision.models import densenet201

# Load your model
model = densenet201(pretrained=False)  # Use your specific model
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 512),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(512, 41)  # Adjust number of output classes to match your dataset
)
print(type(model))  # Should output <class 'torchvision.models.densenet.DenseNet'>

# Set the model to evaluation mode
model.eval()

# Print the summary (assuming input size is 3x224x224 for RGB images)
summary(model,input_size=(3,224,224))
