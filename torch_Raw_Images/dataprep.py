import os
import numpy as np
import librosa
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import multiprocessing
from tqdm import tqdm

def compute_audio_representation(file_path):
    """Compute a maximum-resolution multi-channel representation for a given audio file path."""
    # Using a very small hop_length for maximum resolution
    hop_length_max_res = 256  # This is a commonly used small value for hop_length in audio processing
    
    y, sr = librosa.load(file_path, sr=None, duration=3.0)
    
    # Compute the Spectrogram with reduced hop_length
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length_max_res)))
    
    # Compute the Mel-spectrogram with reduced hop_length
    mel_spec = librosa.amplitude_to_db(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length_max_res))
    
    # Resizing the Mel-spectrogram to match the shape of Spectrogram
    mel_spec_resized = resize(mel_spec, spec.shape, mode='reflect', anti_aliasing=True)
    
    # Waveform (creating a 2D representation by segmenting into overlapping windows)
    waveform_2d = np.array([y[i:i+hop_length_max_res] for i in range(0, len(y) - hop_length_max_res, hop_length_max_res // 2)])
    
    # Resizing waveform_2d to match spec's shape
    waveform_2d_resized = resize(waveform_2d, spec.shape, mode='reflect', anti_aliasing=True)
    
    # Stack the channels without resizing further to retain the maximum resolution
    multi_channel_input = np.concatenate([spec[..., np.newaxis], mel_spec_resized[..., np.newaxis], waveform_2d_resized[..., np.newaxis]], axis=-1)
    
    return multi_channel_input


def compute_audio_representation(file_path):
    """Compute a maximum-resolution multi-channel representation for a given audio file path."""
    # Using a very small hop_length for maximum resolution
    hop_length_max_res = 256  # This is a commonly used small value for hop_length in audio processing
    
    y, sr = librosa.load(file_path, sr=None, duration=3.0)
    
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

def process_file(args):
    """
    Function to process an individual file. Designed for parallel processing.
    """
    file_path, save_dir, label = args  # Unpack arguments
    try:
        images = compute_audio_representation(file_path)

        tensor_representation = torch.tensor(images, dtype=torch.float32)
        tensor_output_path = os.path.join(save_dir, f"{label}_tensor_{os.path.basename(file_path).split('.')[0]}.pt")
        torch.save(tensor_representation, tensor_output_path)
        
        return tensor_output_path  # Returning the saved tensor path for logging purposes

    except Exception as e:
        print(f"Error processing file {file_path}. Reason: {e}")
        return None

def compute_class_representations(directory, save_dir, label):
    files = [os.path.join(directory, file_name) for file_name in os.listdir(directory) if file_name.endswith('.mp3')]  # Assumption: Only processing .mp3 files

    # Use multiprocessing.Pool
    pool = multiprocessing.Pool()
    results = list(tqdm(pool.imap(process_file, [(file, save_dir, label) for file in files]), total=len(files)))

    pool.close()
    pool.join()

    # Return the paths to the saved tensors for logging purposes
    return results


if __name__ == "__main__":
    human_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/3s/'
    ai_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aiTraining/3s'
    save_dir_ai = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch/data_prepped/ai/3s'
    save_dir_human = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch/data_prepped/human/3s'

    # Ensure save directory exists
   

    human_tensor_paths = compute_class_representations(human_dir, save_dir_human, 'human')
    ai_tensor_paths = compute_class_representations(ai_dir, save_dir_ai, 'ai')

    # Log the tensor paths if required (optional)
    print(f"Saved human audio tensors at: {human_tensor_paths}")
    print(f"Saved AI audio tensors at: {ai_tensor_paths}")