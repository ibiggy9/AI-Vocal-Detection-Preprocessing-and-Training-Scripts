import torch
import torch.nn as nn
import torch.optim as optim
import os
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
import joblib

# Define the DNN model
class FeedForwardDNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedForwardDNN, self).__init__()

        # Define the architecture
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)         # Second hidden layer
        self.fc3 = nn.Linear(64, 32)          # Third hidden layer
        self.fc4 = nn.Linear(32, 1)           # Output layer

        self.activation = nn.ReLU()           # Activation function

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Using sigmoid for binary classification
        return x





class AudioDataset(Dataset):
    def __init__(self, feature_store):
        self.files = list(feature_store.keys())
        self.features = [feature_store[file][0] for file in self.files]
        self.labels = [feature_store[file][1] for file in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        features = self.features[idx]
        # Normalize the features
        features = (features - np.mean(features)) / np.std(features)
        
        return torch.tensor(features).float(), torch.tensor(self.labels[idx]).float()

def extract_features_from_path(file_info):
    file, label, path = file_info
    features = extract_features(os.path.join(path, file))
    return file, features, label

def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)  # Load the entire file without specifying duration
        duration = librosa.get_duration(y=y, sr=sr)

        # Skip if the duration is not close to 3 seconds
        if not (2.95 <= duration <= 3.05):
            print(f"Skipped file {audio_file} of duration {duration:.2f} seconds")
            return None

        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)
        mfccs_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
        mfccs_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
        
        harmonic_peaks = librosa.effects.harmonic(y)
        harmonic_peaks_freq_mean = [np.mean(harmonic_peaks)]
        harmonic_peaks_mag_mean = [np.mean(np.abs(harmonic_peaks))]

        melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)

        spectral_centroid = [np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))]
        spectral_bandwidth = [np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))]
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        spectral_flatness = [np.mean(librosa.feature.spectral_flatness(y=y))]
        autocorrelation = [np.mean(np.correlate(y, y))]
        onset_strength = [np.mean(librosa.onset.onset_strength(y=y, sr=sr))]

        # Combine features into a single array
        return np.hstack((mfccs_mean, mfccs_var, mfccs_delta, mfccs_delta2, harmonic_peaks_freq_mean,
                        harmonic_peaks_mag_mean, melspectrogram, spectral_centroid, spectral_bandwidth,
                        spectral_contrast, spectral_flatness, autocorrelation, onset_strength))
    
    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        return []

# Set the device to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define paths
base_dir = "/Users/Main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/"
#ai_dir = os.path.join(base_dir, "AiTraining/3s")
#human_dir = os.path.join(base_dir, "humanTraining/3s")

ai_dir = os.path.join(base_dir, "testdirai")
human_dir = os.path.join(base_dir, "testdirhuman")

# 1. Check which files have already been processed
existing_files = set()
if os.path.exists("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch_features_only/saved_features/processed_files.txt"):
    with open("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch_features_only/saved_features/processed_files.txt", "r") as file:
        existing_files = set(file.read().splitlines())

ai_files = set(os.listdir(ai_dir))
human_files = set(os.listdir(human_dir))

new_ai_files = ai_files - existing_files
new_human_files = human_files - existing_files
all_new_files = new_ai_files.union(new_human_files)

# Extract features only for new files
feature_store = {}
for file in tqdm(all_new_files, desc="Extracting features"):
    path = ai_dir if file in new_ai_files else human_dir
    features = extract_features(os.path.join(path, file))
    if features is not None:
        feature_store[file] = features


joblib.dump(feature_store, "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch_features_only/saved_features/features_store.pkl")
    

# 2. Save extracted features using joblib
#joblib.dump(feature_store, "features_store.pkl")

# Assuming AudioDataset uses the new features (modify AudioDataset if needed)
dataset = AudioDataset(feature_store)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create the model and move to MPS
sample_features = list(feature_store.values())[0] if feature_store else []
input_dim = len(sample_features)
model = FeedForwardDNN(input_dim).to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Train the model
print("Training model...")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move inputs and labels to MPS
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

# After training, update the list of processed files
with open("processed_files.txt", "a") as file:
    for file_name in all_new_files:
        file.write(file_name + "\n")