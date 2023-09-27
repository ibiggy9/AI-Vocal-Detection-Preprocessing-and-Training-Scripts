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
import multiprocessing
from torch.utils.data import random_split

feature_store = None
EXPECTED_FEATURE_DIM = None


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
    global EXPECTED_FEATURE_DIM
    try:
        y, sr = librosa.load(audio_file, duration=3.0)

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

        features = np.hstack((mfccs_mean, mfccs_var, mfccs_delta, mfccs_delta2, harmonic_peaks_freq_mean,
                              harmonic_peaks_mag_mean, melspectrogram, spectral_centroid, spectral_bandwidth,
                              spectral_contrast, spectral_flatness, autocorrelation, onset_strength))

      

        # If it's the first time extracting, set the expected dimension
        if EXPECTED_FEATURE_DIM is None:
            EXPECTED_FEATURE_DIM = features.shape[0]

        # Ensure the feature has the expected dimension
        if features.shape[0] != EXPECTED_FEATURE_DIM:
            print(f"Error: Mismatched feature size for file {audio_file}. Expected {EXPECTED_FEATURE_DIM}, got {features.shape[0]}")
            return np.zeros((EXPECTED_FEATURE_DIM,))  # Return a zero-filled array of expected dimension

        return features

    except Exception as e:
        print(f"Error processing {audio_file}. Error: {e}")
        return np.zeros((EXPECTED_FEATURE_DIM,))


# Set the device to MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define paths
base_dir = "/Users/Main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/"
ai_dir = os.path.join(base_dir, "aitraining/3s")
human_dir = os.path.join(base_dir, "humantraining/3s")



ai_files = set(os.listdir(ai_dir))
human_files = set(os.listdir(human_dir))


all_new_files = ai_files.union(human_files)

# Extract features only for new files
def process_file(file):
    # Extract features for a given file
    path = ai_dir if file in ai_files else human_dir
    label = 1 if file in ai_files else 0
    features = extract_features(os.path.join(path, file))
    return file, (features, label)

def runner():
    global feature_store
    global EXPECTED_FEATURE_DIM
    # Use multiprocessing to speed up feature extraction
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, all_new_files), total=len(all_new_files), desc="Extracting features"))
    
    

    # Populate feature_store with the results
    feature_store = {file: features for (file, features) in results}

    # Check for inconsistent features after extraction
    inconsistent_files = [file for file, features in feature_store.items() if len(features) != EXPECTED_FEATURE_DIM]

    if inconsistent_files:
        print(f"Files with inconsistent feature dimensions: {inconsistent_files}")
    
    

    # 2. Save extracted features using joblib
    #joblib.dump(feature_store, "features_store.pkl")

    # Assuming AudioDataset uses the new features (modify AudioDataset if needed)
    dataset = AudioDataset(feature_store)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create the model and move to MPS
    

if __name__ == "__main__":
    runner()
    
    num_epochs = 10
    dataset = AudioDataset(feature_store)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    sample_features = list(feature_store.values())[0] if feature_store else []
    input_dim = 222
    print(input_dim)
    model = FeedForwardDNN(input_dim).to(device)

    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model_path = "/Users/Main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch_features_only/models/v4_model.pth"




    print("Training model...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                epoch_val_loss += loss.item()
                
                # For accuracy
                predicted_labels = (outputs > 0.5).float()  # Threshold the outputs
                correct_predictions += (predicted_labels == labels.unsqueeze(1)).sum().item()
                total_predictions += labels.size(0)
        
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = correct_predictions / total_predictions

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at: {model_path}")

