import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import soundfile as sf
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import pickle
from functools import partial
from multiprocessing import Manager
# Your extract_chunk_features function should be defined here or imported from another module.

# Define the sampling rate for the audio files (in Hz)
SAVED_FEATURES_PATH = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/torch_features_only/saved_features/dataprep//extracted_features.pkl"
sampling_rate = 44000
n_mfcc = 13
n_mels = 128
X_data = []
extracted_files = []


# Neural Network Model


def save_features_callback(result, X_data, extracted_files):
    """
    Callback function to save extracted features after each file is processed.
    """
    X_data.append(result[0])
    extracted_files.append(result[1])
    data_to_save = {
        'features': X_data,
        'audio_files': extracted_files
    }
    with open(SAVED_FEATURES_PATH, 'wb') as f:
        pickle.dump(data_to_save, f)

def pad_or_truncate(features, target_shape=(83, 194)):
    """
    Pad or truncate the features to match the target shape.
    Handles both dimensions of the features and ensures input is at least 2D.
    """
    # Ensure the features are at least 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Handle the first dimension
    diff_rows = target_shape[0] - features.shape[0]
    if diff_rows > 0:
        padding_rows = np.zeros((diff_rows, features.shape[1]))
        features = np.vstack([features, padding_rows])
    elif diff_rows < 0:
        features = features[:target_shape[0], :]

    # Handle the second dimension
    diff_cols = target_shape[1] - features.shape[1]
    if diff_cols > 0:
        padding_cols = np.zeros((features.shape[0], diff_cols))
        features = np.hstack([features, padding_cols])
    elif diff_cols < 0:
        features = features[:, :target_shape[1]]

    return features



def parallel_feature_extractor(file_path):
    """
    Function to extract features from a single file and return them along with the file path.
    """
    features = extract_chunk_features(file_path)
    return features, file_path

def train_model(audio_files, labels, X_data, extracted_files):
    # Attempt to load extracted features
    try:
        X_data, extracted_files = load_extracted_features()
    except FileNotFoundError:
        print("No saved features found.")
        X_data = []
        extracted_files = []

    new_files = [f for f in audio_files if f not in extracted_files]

    if new_files:
        '''
        print("Extracting features from new audio files using multiprocessing...")
        
        with Manager() as manager:
            X_data = manager.list(X_data)  # Convert to Manager list for sharing across processes
            extracted_files = manager.list(extracted_files)  # Convert to Manager list for sharing across processes
            completed_tasks = manager.list()  # Create Manager list to keep track of completed tasks
            
            with Pool(os.cpu_count()) as pool, tqdm(total=len(new_files), desc="Processing audio files") as pbar:
                def update(result):
                    pbar.update()
                    completed_tasks.append(1)  # Append to completed_tasks list every time a task is completed
                    X_data.append(result[0])
                    extracted_files.append(result[1])
                    data_to_save = {
                        'features': list(X_data),
                        'audio_files': list(extracted_files)
                    }
                    with open(SAVED_FEATURES_PATH, 'wb') as f:
                        pickle.dump(data_to_save, f)
                
                for file_path in new_files:
                    pool.apply_async(parallel_feature_extractor, args=(file_path,), callback=update)  # Use update as callback
                pool.close()
                pool.join()
            
            # Convert Manager list back to regular list
            X_data = list(X_data)
            extracted_files = list(extracted_files)
        '''

    print("\nPreprocessing data...")
    X_data_flat = [np.array(features).flatten() for features in X_data]
    
    X_train, X_val, y_train, y_val = train_test_split(X_data_flat, labels, test_size=0.2, random_state=42)
    for i, x in enumerate(X_train[:10]):
        print(f"Element {i} shape: {np.array(x).shape}")

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    X_val = scaler.transform(X_val)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Model Initialization
    print("\nInitializing model...")
    model = AudioClassifier(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("\nStarting training...")
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # You can further evaluate the model on the validation set and adjust hyperparameters accordingly.


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



def load_extracted_features():
    with open(SAVED_FEATURES_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['audio_files']

def get_audio_files_from_directory(directory_path):
    """Returns a list of audio file paths from the specified directory."""
    return [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.mp3')]

if __name__ == "__main__":
    base_dir = "/Users/Main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/"
    ai_dir = os.path.join(base_dir, "AiTraining/3s")
    human_dir = os.path.join(base_dir, "humanTraining/3s")
    
    # Get audio files from the directories
    ai_files = get_audio_files_from_directory(ai_dir)
    human_files = get_audio_files_from_directory(human_dir)
    
    # Combine the lists of audio files
    audio_files = ai_files + human_files
    # Create labels: 1 for AI-generated and 0 for human-generated
    #labels = [1] * len(ai_files) + [0] * len(human_files)

    labels= [1] * 40 + [0] * 40
    
    print("Extracting features from audio files using multiprocessing...")
    
    train_model(audio_files, labels, X_data, extracted_files)
  # Call train_model instead of extract_features_parallel
