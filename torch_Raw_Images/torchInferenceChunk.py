import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, random_split, Dataset
import os
import librosa
import numpy as np
from skimage.transform import resize
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Define necessary functions from your training code
def compute_audio_representations_with_waveform_fixed(chunk, sr):
    """
    Compute multi-channel representations including the waveform for a given audio chunk.
    """
    TARGET_SHAPE = (1025, 52)
    
    # Compute the Spectrogram
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(chunk)))
    
    # Compute the Mel-spectrogram
    mel_spec = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=chunk, sr=sr))
    
    # Resize the Mel-spectrogram to match the shape of Spectrogram
    mel_spec_resized = resize(mel_spec, spec.shape, mode='reflect', anti_aliasing=True)
    
    # Waveform (creating a 2D representation by segmenting into overlapping windows)
    hop_length = len(chunk) // spec.shape[1]
    waveform_2d = np.array([chunk[i:i+hop_length] for i in range(0, len(chunk) - hop_length, hop_length // 2)])
    
    # Resizing waveform_2d to match spec's shape
    waveform_2d_resized = resize(waveform_2d, spec.shape, mode='reflect', anti_aliasing=True)
    
    # Reshape for stacking
    spec = resize(spec, TARGET_SHAPE, mode='reflect', anti_aliasing=True)
    mel_spec_resized = resize(mel_spec_resized, TARGET_SHAPE, mode='reflect', anti_aliasing=True)
    waveform_2d_resized = resize(waveform_2d_resized, TARGET_SHAPE, mode='reflect', anti_aliasing=True)
    
    # Stack the channels
    multi_channel_input = np.concatenate([spec[..., np.newaxis], mel_spec_resized[..., np.newaxis], waveform_2d_resized[..., np.newaxis]], axis=-1)

    # For debugging, print the shapes
    #print("spec shape:", spec.shape)
    #print("mel_spec_resized shape:", mel_spec_resized.shape)
    #print("waveform_2d_resized shape:", waveform_2d_resized.shape)
    #print("multi_channel_input shape:", multi_channel_input.shape)
    
    return multi_channel_input



def custom_audio_loader(path):
    y, sr = librosa.load(path, sr=None)
    chunks = [y[i:i + int(10*sr)] for i in range(0, len(y), int(10*sr))]
    multi_channel_data = [compute_audio_representations_with_waveform_fixed(chunk, sr) for chunk in chunks]
    return multi_channel_data

class AudioClassifier3Channel(nn.Module):
    def __init__(self):
        super(AudioClassifier3Channel, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # Changed to 3 input channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layers
        # We'll determine the input size for fc1 later, once we know the output shape from the convolutional layers.
        self.fc1 = nn.Linear(98304, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Two classes: AI-generated and human-generated
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        # First Conv Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Second Conv Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Third Conv Block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.reshape(x.size(0), -1)

        
        #print(x.shape)
        
        # Fully Connected Layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=1)
        
        
        return x

class AudioInference:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AudioClassifier3Channel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio_path):
        # Preprocess the audio
        chunks = custom_audio_loader(audio_path)
        inputs = torch.tensor(chunks, dtype=torch.float32).to(self.device)
        
        # Perform inference
        results = []
        with torch.no_grad():
            outputs = self.model(inputs)
            probabilities, predicted = torch.max(outputs, 1) # Getting probabilities alongside predictions
            for i, (pred, prob) in enumerate(zip(predicted, probabilities)):
                results.append({
                    'chunk': i + 1,
                    'prediction': pred.item(),
                    'probability': prob.item()
                })

        return results
if __name__ == "__main__":
    inferencer = AudioInference('/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchModels/neuralNet.pth')
    audio_path = '/Users/main/Downloads/Tom Cruise human 1 Ai-SPY.mp3'
    results = inferencer.predict(audio_path)
    for res in results:
        print(f"Chunk {res['chunk']} is predicted to be {'AI-generated' if res['prediction'] == 1 else 'Human-generated'} with a probability of {res['probability']:.2f}")

