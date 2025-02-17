import torch
from torchvision import models
import torch.nn as nn

model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
model.classifier = nn.Sequential(
    nn.Linear(1920, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 101)
)
model.load_state_dict(torch.load("densenet201_food_classifier.pth", map_location=torch.device('cpu')))
model.eval()
print("Model loaded successfully")
