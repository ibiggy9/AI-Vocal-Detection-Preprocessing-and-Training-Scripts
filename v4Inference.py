import librosa
import numpy as np
import torch
import torch.nn as nn

# Additional potential imports, based on the functions you might use:
from torch.utils.data import DataLoader  # if you're using a DataLoader for 'test_loader'
import os  # if you decide to save temporary audio clips to files
import tempfile 

EXPECTED_FEATURE_DIM=None

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


def extract_features_from_clip(clip, sr):

    global EXPECTED_FEATURE_DIM
    try:
        y = clip

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
            
            return np.zeros((EXPECTED_FEATURE_DIM,))  # Return a zero-filled array of expected dimension

        return features

    except Exception as e:
        
        return np.zeros((EXPECTED_FEATURE_DIM,))
def split_audio_into_clips(audio_path, clip_duration=3.0):
    """
    Splits audio located at `audio_path` into `clip_duration` second clips.
    Returns a list of numpy arrays each containing the audio samples of the clip.
    """
    y, sr = librosa.load(audio_path)
    num_samples_per_clip = int(clip_duration * sr)

    clips = []
    for start in range(0, len(y), num_samples_per_clip):
        end = start + num_samples_per_clip
        clip = y[start:end]
        if len(clip) == num_samples_per_clip:  # Ensure all clips are of equal length
            clips.append(clip)
    
    return clips, sr


def run_inference_on_audio(audio_path, model, clip_duration=3.0):
    # Split audio into clips
    clips, sr = split_audio_into_clips(audio_path, clip_duration)
    
    # Extract features from each clip
    clip_features = []
    for clip in clips:
        # Mimic the structure of `extract_features` function
        features = extract_features_from_clip(clip, sr)
        clip_features.append(features)

    # Convert features to a tensor
    clip_features_array = np.vstack(clip_features)
    clip_features_tensor = torch.tensor(clip_features_array).float().to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(clip_features_tensor)
        predicted_labels = (predictions > 0.5).float().squeeze()
    
    return predicted_labels.cpu().numpy()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = FeedForwardDNN(input_dim=222)
model.load_state_dict(torch.load('/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch_features_only/models/v4_model.pth'))
model.to(device)
model.eval()

predictions = run_inference_on_audio('/Users/main/downloads/douglas.mp3', model)
print(predictions)


