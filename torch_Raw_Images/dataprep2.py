import os
import numpy as np
import librosa
from skimage.transform import resize
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path
import uuid

# Constants for feature extraction and preprocessing
HOP_LENGTH_MAX_RES = 256
N_MELS = 40
NEW_SR_MULTIPLIER = 2
MFCC_COUNT = 13
DURATION = 2.0
RESIZE_SHAPE = (1025, 517)


def process_file(args):
    """
    Function to process an individual file. Designed for parallel processing.
    """
    file_path, save_dir, label = args  # Unpack arguments
    try:
        images = compute_audio_representation(file_path)
        
        tensor_representation = torch.tensor(images, dtype=torch.float32)
        
        # Generate a random UUID
        uid = uuid.uuid4()
        
        # Construct the tensor filename with the UID
        tensor_output_path = os.path.join(save_dir, f"{label}_tensor_{uid}.pt")
        
        torch.save(tensor_representation, tensor_output_path)
        
        return tensor_output_path  # Returning the saved tensor path for logging purposes
    
    except Exception as e:
        print(f"Error processing file {file_path}. Reason: {e}")
        return None

def compute_audio_representation(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None, duration=3.0)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta_mean = np.mean(mfccs_delta, axis=1)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs_delta2_mean = np.mean(mfccs_delta2, axis=1)
    
    # Harmonic peaks
    harmonic, _ = librosa.effects.hpss(y)
    harmonic_peaks_freqs = np.where(harmonic > np.mean(harmonic))[0]
    harmonic_peaks_magnitudes = harmonic[harmonic_peaks_freqs]
    harmonic_peaks_freq_mean = np.mean(harmonic_peaks_freqs) if len(harmonic_peaks_freqs) > 0 else 0
    harmonic_peaks_mag_mean = np.mean(harmonic_peaks_magnitudes) if len(harmonic_peaks_magnitudes) > 0 else 0
    
    # Mel spectrogram
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # Autocorrelation
    autocorrelation = np.mean(librosa.autocorrelate(y))
    
    # Onset strength
    onset_strength = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
    
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
    
    return features

def compute_class_representations(directory, save_dir, label):

    # Get all .mp3 files in the directory
    files = list(directory.glob("*.mp3"))
    
    # Ensure the save directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process files in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_file, [(file, save_dir, label) for file in files]), total=len(files)))

    return results

if __name__ == "__main__":
    # Define directory paths
    human_dir = Path('/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/5s')
    ai_dir = Path('/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aiTraining/5s')
    save_dir_ai = Path('/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch/dataFeatures/ai/5s')
    save_dir_human = Path('/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch/dataFeatures/human/5s')

    # Compute and save representations
    ai_tensor_paths = compute_class_representations(ai_dir, save_dir_ai, 'ai')
    human_tensor_paths = compute_class_representations(human_dir, save_dir_human, 'human')

    # Log any errors that occurred during processing
    '''
    failed_files = [x for x in human_tensor_paths + ai_tensor_paths if isinstance(x, tuple)]
    if failed_files:
        print("Files with errors:")
        for file, error in failed_files:
            print(f"{file}: {error}")
    '''