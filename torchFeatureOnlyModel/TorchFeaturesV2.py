import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import soundfile as sf
import librosa


sampling_rate = 44000
n_mfcc = 13

class AudioDataset(Dataset):
    def __init__(self, ai_dir, human_dir, transform=None):
        self.ai_files = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir) if f.endswith('.mp3')]
        self.human_files = [os.path.join(human_dir, f) for f in os.listdir(human_dir) if f.endswith('.mp3')]
        
        self.file_paths = self.ai_files + self.human_files
        self.labels = [0] * len(self.ai_files) + [1] * len(self.human_files)
        
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Extract features
        features = extract_chunk_features(file_path)
        if len(features) > 0:
            # Using only the first chunk's features for simplicity
            features = features[0]
        else:
            # If there's an issue with the file, use zero-filled features
            features = np.zeros(50)  # An arbitrary number; you might need to adjust based on actual feature length

        # Transform if applicable
        if self.transform:
            features = self.transform(features)

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, device="cpu"):
    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float().view(-1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train

        # Validate the model
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float().view(-1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Load the best model weights
    model.load_state_dict(best_model_weights)
    return model




class AudioClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

def extract_chunk_features(file_path, chunk_size=0.6, hop_size=0.6):
    try:
        data, sr = sf.read(file_path)

        if data.ndim > 1 and data.shape[1] == 2:
            data = np.mean(data, axis=1)

        data_resampled = librosa.resample(data, orig_sr=sr, target_sr=sampling_rate)

        chunks = librosa.util.frame(data_resampled,
                                    frame_length=int(chunk_size * sampling_rate),
                                     hop_length=int(hop_size * sampling_rate))

        all_features = []
        for chunk in chunks.T:
            # Extract features
            mfccs = librosa.feature.mfcc(y=chunk, sr=sampling_rate, n_mfcc=n_mfcc)
            mfccs_mean = np.mean(mfccs, axis=1)
            #print(f"mfccs_mean shape: {mfccs_mean.shape}")
            mfccs_var = np.var(mfccs, axis=1)
            #print(f"mfccs_var shape: {mfccs_var.shape}")

            
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta_mean = np.mean(mfccs_delta, axis=1)
            #print(f"mfccs_delta_mean shape: {mfccs_delta_mean.shape}")

            
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            mfccs_delta2_mean = np.mean(mfccs_delta2, axis=1)
            #print(f"mfccs_delta2_mean shape: {mfccs_delta2_mean.shape}")


            spectrum = np.abs(np.fft.fft(chunk))
            log_spectrum = np.log(spectrum + 1e-8)
            cepstrum = np.fft.ifft(log_spectrum).real
            
            harmonic, percussive = librosa.effects.hpss(chunk)
            harmonic_peaks_freqs = np.where(harmonic > np.mean(harmonic))[0]
            harmonic_peaks_magnitudes = harmonic[harmonic_peaks_freqs]


            
            melspectrogram = np.mean(librosa.feature.melspectrogram(y=chunk, sr=sampling_rate), axis=1)
            #print(f"melspectrogram shape: {melspectrogram.shape}")

            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sampling_rate))
            #print(f"spectral_centroid shape: {np.array([spectral_centroid]).shape}")

            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sampling_rate))
            #print(f"spectral_bandwidth shape: {np.array([spectral_bandwidth]).shape}")


            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=chunk, sr=sampling_rate), axis=1)
            #print(f"spectral_contrast shape: {spectral_contrast.shape}")


            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=chunk))
            #print(f"spectral_flatness shape: {np.array([spectral_flatness]).shape}")


            autocorrelation = np.mean(librosa.autocorrelate(chunk))
            #print(f"autocorrelation shape: {np.array([autocorrelation]).shape}")


            onset_strength = np.mean(librosa.onset.onset_strength(y=chunk, sr=sampling_rate))
            #print(f"onset_strength shape: {np.array([onset_strength]).shape}")




            spectrum = np.abs(np.fft.fft(chunk))
            log_spectrum = np.log(spectrum + 1e-8)
            cepstrum = np.fft.ifft(log_spectrum).real
            cepstral_peak = np.max(cepstrum)
            cpp = cepstral_peak - np.min(cepstrum)
            
            harmonic, percussive = librosa.effects.hpss(chunk)
            harmonic_peaks_freqs = np.where(harmonic > np.mean(harmonic))[0]
            harmonic_peaks_magnitudes = harmonic[harmonic_peaks_freqs]

            if len(harmonic_peaks_freqs) > 0:
                harmonic_peaks_freq_mean = np.mean(harmonic_peaks_freqs)
                harmonic_peaks_mag_mean = np.mean(harmonic_peaks_magnitudes)
            else:
                harmonic_peaks_freq_mean = 0
                harmonic_peaks_mag_mean = 0
            
            if len(harmonic_peaks_freqs) > 0:
                harmonic_peaks_freq_mean = np.mean(harmonic_peaks_freqs)
                harmonic_peaks_mag_mean = np.mean(harmonic_peaks_magnitudes)
            else:
                harmonic_peaks_freq_mean = 0
                harmonic_peaks_mag_mean = 0

            #print(f"harmonic_peaks_freq_mean shape: {np.array([harmonic_peaks_freq_mean]).shape}")
            #print(f"harmonic_peaks_mag_mean shape: {np.array([harmonic_peaks_mag_mean]).shape}")

            # Concatenate features
            features = np.concatenate([
                mfccs_mean, 
                mfccs_var, 
                mfccs_delta_mean,
                mfccs_delta2_mean,
                [harmonic_peaks_freq_mean],
                [harmonic_peaks_mag_mean],
                melspectrogram,
                [spectral_centroid],
                [spectral_bandwidth],
                spectral_contrast,
                [spectral_flatness],
                [autocorrelation],
                [onset_strength]
            ])

            
         
            

            all_features.append(features)

        return all_features

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

        # Check if the error message suggests that the file doesn't exist or is not a regular file
        if "File does not exist or is not a regular file" in str(e):
            # Deleting the problematic file
            os.remove(file_path)
            print(f"Deleted the problematic file: {file_path}")

        # Return a zero-filled feature vector of the expected length
        
        return []  


base_dir = "/Users/Main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/"
ai_dir = os.path.join(base_dir, "AiTraining/3s")
human_dir = os.path.join(base_dir, "humanTraining/3s")

audio_dataset = AudioDataset(ai_dir, human_dir)

# Dataset Creation and Splitting
audio_dataset = AudioDataset(ai_dir, human_dir)
train_size = int(0.8 * len(audio_dataset))
val_size = len(audio_dataset) - train_size ize, val_size])

# Data Loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model Initialization and Training
device = "mps"
input_dim = 50  # Modify based on actual feature length if different
model = AudioClassifier(input_dim).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, device=device)




# Feature Extraction for New Audio
new_audio_path = "/path/to/new/audio.mp3"
new_features = extract_chunk_features(new_audio_path)