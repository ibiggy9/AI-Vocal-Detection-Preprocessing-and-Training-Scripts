import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
# Assuming the same AudioClassifier3Channel and compute_audio_representations_with_waveform_fixed functions are available here

class AudioClassifier3Channel(nn.Module):
    def __init__(self, input_size=(1025, 517)):
        super(AudioClassifier3Channel, self).__init__()

        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)  # Changed input channels to 5
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=3)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculating the flattened size for convolutional outputs
        with torch.no_grad():
            sample = torch.randn(1, 3, *input_size)
            for _ in range(2):  # Adjusted for 3 layers
                sample = F.max_pool2d(sample, 2)
            self.conv_flat_features = np.prod(sample.shape[1:])

        # Fully Connected Layers
        self.fc1 = nn.Linear(1048576, 64)
        self.fc2 = nn.Linear(64, 2)  # Two classes: AI-generated and human-generated

        # Dropout
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.25)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)

        # First Conv Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Second Conv Block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        # Third Conv Block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def load_model(model_path):
    model = AudioClassifier3Channel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model




def compute_audio_representation(y, sr):
    hop_length_max_res = 256  
    #y, sr = librosa.load(file_path, sr=None, duration=3.0)
    
    # Compute the Spectrogram
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length_max_res)))
    
    # Compute the Mel-spectrogram
    mel_spec = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length_max_res))

    # Compute Chromagram
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length_max_res)

    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length_max_res, n_mfcc=13)
    
    # Compute RMS Energy
    rms_energy = librosa.feature.rms(y=y, hop_length=hop_length_max_res)
    
    # Zero-Crossing Rate
    zero_crossings = librosa.feature.zero_crossing_rate(y, hop_length=hop_length_max_res)
    
    # Explicitly resize the Spectrogram to (1025, 517)
    spec_resized = resize(spec, (1025, 517), mode='reflect', anti_aliasing=True)
    mel_spec_resized = resize(mel_spec, (1025, 517), mode='reflect', anti_aliasing=True)
    chroma_resized = resize(chroma, (1025, 517), mode='reflect', anti_aliasing=True)
    mfcc_resized = resize(mfcc, (1025, 517), mode='reflect', anti_aliasing=True)
    rms_energy_resized = resize(rms_energy, (1025, 517), mode='reflect', anti_aliasing=True)
    
    # Stack the channels
    multi_channel_input = np.concatenate([
        spec_resized[..., np.newaxis], 
        mel_spec_resized[..., np.newaxis],
        chroma_resized[..., np.newaxis],
        mfcc_resized[..., np.newaxis],
        rms_energy_resized[..., np.newaxis]
    ], axis=-1)
    
    return multi_channel_input

def run_inference(model, audio_file_path):
    # Load the entire audio file
    y, sr = librosa.load(audio_file_path, sr=44000)  # Assuming sr=22050

    
    plt.figure(figsize=(10, 4))
    plt.plot(y)
    plt.title('Audio Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.show()
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
            
            processed_chunk = compute_audio_representation(chunk, sr)  # Passing the chunk, not the file path
            #print(processed_chunk)
            processed_chunk = torch.tensor(processed_chunk).unsqueeze(0).permute(0, 2, 1, 3)
            #print(processed_chunk)
                

 # corrected tensor reshaping

            # Run inference
            with torch.no_grad():
                outputs = model(processed_chunk)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted_idx = torch.max(outputs, 1)
                print(outputs)
                print(probabilities)
                print(predicted_idx.item())

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
    audio_data =  "/Users/main/Downloads/new recording 11.mp3"

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