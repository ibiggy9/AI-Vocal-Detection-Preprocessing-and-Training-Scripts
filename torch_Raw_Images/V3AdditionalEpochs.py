import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from skimage.transform import resize

# Assuming the same AudioClassifier3Channel and compute_audio_representations_with_waveform_fixed functions are available here

class AudioClassifier3Channel(nn.Module):
    def __init__(self, input_size=(1025, 517)):
        super(AudioClassifier3Channel, self).__init__()

        # First, define your layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # Changed to 3 input channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Then, calculate the flattened size
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
        
        # Flatten the tensor
        
        x = x.reshape(x.size(0), -1)
        
        
        # Fully Connected Layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


def load_model(model_path):
    model = AudioClassifier3Channel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model




def compute_audio_representation(y):
    """Compute a maximum-resolution multi-channel representation for a given audio chunk."""
    # Using a very small hop_length for maximum resolution
    hop_length_max_res = 256  # This is a commonly used small value for hop_length in audio processing
    
    sr = 44000  # Assuming a sampling rate of 44000, adjust if needed
    
    # Compute the Spectrogram with reduced hop_length
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length_max_res)))
    
    # Compute the Mel-spectrogram with reduced hop_length
    mel_spec = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length_max_res))
    
    # Explicitly resize the Spectrogram to (1025, 517)
    spec_resized = resize(spec, (1025, 517), mode='reflect', anti_aliasing=True)
    mel_spec_resized = resize(mel_spec, (1025, 517), mode='reflect', anti_aliasing=True)
    
    # Waveform (creating a 2D representation by segmenting into overlapping windows)
    waveform_2d = np.array([y[i:i+hop_length_max_res] for i in range(0, len(y) - hop_length_max_res, hop_length_max_res // 2)])
    
    # Resizing waveform_2d to match spec's shape
    waveform_2d_resized = resize(waveform_2d, (1025, 517), mode='reflect', anti_aliasing=True)
    
    # Stack the channels without resizing further to retain the maximum resolution
    multi_channel_input = np.concatenate([spec_resized[..., np.newaxis], mel_spec_resized[..., np.newaxis], waveform_2d_resized[..., np.newaxis]], axis=-1)
    
    return multi_channel_input

def run_inference(model, audio_file_path):
    # Load the entire audio file
    y, sr = librosa.load(audio_file_path, sr=44000)  # Assuming sr=22050
    if not isinstance(y, np.ndarray):
        raise ValueError(f"Loaded audio data from {audio_file_path} is not a numpy array. Instead, it's {type(y)}.")

    # Define the chunk size (3 seconds) SET HERE
    chunk_size = 3 * sr

    # Split the audio into 3-second chunks
    chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]

    # Initialize a list to store predictions
    predictions = []

    # Process each chunk
    for idx, chunk in enumerate(chunks):
        # Ensure the chunk is of expected size (last chunk might be smaller)
        if len(chunk) == chunk_size:
            # Process the chunk using the function
            processed_chunk = compute_audio_representation(chunk)  # Passing the chunk, not the file path
            processed_chunk = torch.tensor(processed_chunk).unsqueeze(0).permute(0, 2, 1, 3)
                

 # corrected tensor reshaping

            # Run inference
            with torch.no_grad():
                outputs = model(processed_chunk)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_idx = torch.max(outputs, 1)

                #print(probabilities)
                #print(predicted_idx.item())

                # Create a prediction dictionary for the chunk
                
                

                prediction = {
                    "prediction": "human" if predicted_idx.item() == 0 else "AI",
                    "probability": probabilities[0][predicted_idx.item()].item(),
                    "chunk": idx + 1
                }
                predictions.append(prediction)

    return predictions

if __name__ == "__main__":
    model_path = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchModels/neuralNet_3s.pth'
    audio_data =  "/Users/main/Downloads/New Recording 11.mp3"

    model = load_model(model_path)
    prediction = run_inference(model, audio_data)
    print(f"The audio is predicted to be generated by: {prediction}")


    try:
        ai_count = sum(1 for item in prediction if item['prediction'] == 'AI')
        human_count = sum(1 for item in prediction if item['prediction'] == 'human')
        ai_probabilities = [item['probability'] for item in prediction if item['prediction'] == 'AI']
        avg_ai_probability = sum(ai_probabilities) / len(ai_probabilities) if ai_probabilities else 0

        # For 'human' predictions:
        human_probabilities = [item['probability'] for item in prediction if item['prediction'] == 'human']
        avg_human_probability = sum(human_probabilities) / len(human_probabilities)


        

        if(ai_count > human_count):
            print(f"Overall Predicion: AI")
        else:
            print(f"Overall Prediction: Human")
        print(f"Average AI Probability: {avg_ai_probability:.4f}")
        print(f"Average Human Probability: {avg_human_probability:.4f}")
    
    except:
        pass