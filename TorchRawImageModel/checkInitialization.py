import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the AudioClassifier3Channel class
class AudioClassifier3Channel(nn.Module):
    def __init__(self, input_size=(1025, 517)):
        super(AudioClassifier3Channel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculate the flattened size
        with torch.no_grad():
            sample = torch.randn(1, 3, *input_size)
            sample = self.conv3(self.conv2(self.conv1(sample)))
            self.flat_features = np.prod(sample.shape[1:])

        # Fully Connected Layers
        self.fc1 = nn.Linear(1048576, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Two classes: AI-generated and human-generated

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Permute tensor dimensions
        x = x.permute(0, 3, 1, 2)
        
        # Conv blocks
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        
        # Flatten and pass through fully connected layers
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(xs)))
        x = self.fc3(x)
        
        return x

def load_model(model_path):
    model = AudioClassifier3Channel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Path to your saved model
model_path = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchModels/best_model_features.pth'

# Initialize the model
model = AudioClassifier3Channel()

# Print some weights before loading
print("Weights BEFORE loading:")
print("First conv layer:", model.conv1.weight.data[0][0])

# Load the saved weights
model = load_model(model_path)

# Print the same weights after loading
print("\nWeights AFTER loading:")
print("First conv layer:", model.conv1.weight.data[0][0])
