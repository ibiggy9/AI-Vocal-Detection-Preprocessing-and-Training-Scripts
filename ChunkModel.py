import xgboost as xgb
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from multiprocessing import Pool
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
import joblib
from imblearn.over_sampling import SMOTE
from multiprocessing import Lock
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import random
import soundfile as sf
from sklearn.model_selection import KFold



'''
HOW TO USE THIS PROGRAM:
1) Go to collab which uses gpu to train and get the right hyper params then plug them into the final calcu function below. 
2) This file is used to extract feature as it is much faster at this than collab.
'''


print_lock = Lock()
# Use appropriate paths from your Google Drive
base_dir = "/Users/Main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/"
ai_dir = os.path.join(base_dir, "AiTraining/")
human_dir = os.path.join(base_dir, "humanTraining/splitSongs/")

feature_file_path_human = os.path.join(base_dir, "processedFeatureHuman/human_features_mfcc.joblib")
processed_files_path_human = os.path.join(base_dir, "processedFileHuman/processedFileHuman_mfcc.joblib")

feature_file_path_ai = os.path.join(base_dir, "processedFeatureAI/ai_features_mfcc.joblib")
processed_files_path_ai = os.path.join(base_dir, "processedFileAI/processedFileAI_mfcc.joblib")

# When saving or loading models
model_file_path = os.path.join(base_dir, "chunkModel/my_chunk_model.joblib")

#Test Dir:
#ai_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/testAIDir"
#human_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/testHumanDir"

# Define the sampling rate for the audio files (in Hz)
sampling_rate = 44000
n_mfcc = 13
n_mels = 128  # or any other value you want to use


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
            mfccs_var = np.var(mfccs, axis=1)
            
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta_mean = np.mean(mfccs_delta, axis=1)
            
            mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
            mfccs_delta2_mean = np.mean(mfccs_delta2, axis=1)

            spectrum = np.abs(np.fft.fft(chunk))
            log_spectrum = np.log(spectrum + 1e-8)
            cepstrum = np.fft.ifft(log_spectrum).real
            
            harmonic, percussive = librosa.effects.hpss(chunk)
            harmonic_peaks_freqs = np.where(harmonic > np.mean(harmonic))[0]
            harmonic_peaks_magnitudes = harmonic[harmonic_peaks_freqs]
            
            melspectrogram = np.mean(librosa.feature.melspectrogram(y=chunk, sr=sampling_rate), axis=1)
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sampling_rate))
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=chunk, sr=sampling_rate))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=chunk, sr=sampling_rate), axis=1)
            spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=chunk))
            autocorrelation = np.mean(librosa.autocorrelate(chunk))
            onset_strength = np.mean(librosa.onset.onset_strength(y=chunk, sr=sampling_rate))

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

        return []



def process_chunk_file(file_name, directory, label):
    
    file_path = os.path.join(directory, file_name)
    
    
    if file_path.endswith(".mp3"):
        
        features = extract_chunk_features(file_path)
        
        data = [[file_name, label] + feature.tolist() for feature in features if feature is not None]
        
   
        
        return data
    return []

def create_dataframe(directory, label, processed_files=[]):
    files = os.listdir(directory)

    new_files = [file for file in files if file not in processed_files]


    if not new_files:
        print(f"No new files found in {directory}.")
        return pd.DataFrame()  # Return an empty DataFrame

    all_data = []
    args = [(file, directory, label) for file in new_files]
    

    # Add tqdm() around the iterable to show progress
    
    with Pool(os.cpu_count()) as p:
        all_features = list(tqdm(p.imap_unordered(process_file_wrapper, args), total=len(args)))
        p.close()  # Close the pool
        p.join()   # Join the worker processes
    

    #all_features = [process_chunk_file(*arg) for arg in tqdm(args)]
    # Accumulate processed file names
    processed_file_names = [arg[0] for arg in args]

    for features in all_features:
        all_data.extend(features)


    if not all_data:
      print(f"No features found for files in {directory}.")
      return pd.DataFrame()  # Return an empty DataFrame


    columns = ["file_name", "label"] + [f"feature_{i}" for i in range(len(all_data[0]) - 2)]
    df = pd.DataFrame(all_data, columns=columns)
    #print(f"Processed {len(processed_file_names)} files from {directory}.")
    return df

def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10,8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(model_type + ' - Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.show()


def process_file_wrapper(args):
    #print("Processing")
    return process_chunk_file(*args)

def finalCalculation(ai_df, human_df, hyperparameters=None):
    #print("Starting model training...")

    df = pd.concat([ai_df, human_df], ignore_index=True)

    X = df.iloc[:, 2:]
    label_map = {'ai': 0, 'human': 1}
    y = df["label"].map(label_map)
    nan_rows = X.isnull().any(axis=1)
    if nan_rows.sum() > 0:
        print(f"Found {nan_rows.sum()} rows with NaN values in X. Removing these rows...")
        X = X[~nan_rows]
        y = y[~nan_rows]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

    #print("Scaling the data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # If hyperparameters are not provided, use some default values
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': 481,
            'learning_rate': 0.15225214872619755,
            'max_depth': 10,
            'subsample': 0.9034564885253897
        }

    print(f"Training with specified hyperparameters: {hyperparameters}")

    gbc_model = xgb.XGBClassifier(**hyperparameters, nthread=-1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    gbc_model.fit(X_train, y_train)

    y_pred = gbc_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=0)
    recall = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred, pos_label=0)
    auc_roc = roc_auc_score(y_test, gbc_model.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    print(f"AUC-ROC: {auc_roc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Feature importance
    importance = gbc_model.feature_importances_
    for i, v in enumerate(importance):
        print(f'Feature: {i}, Score: {v:.5f}')

    scaler_file_path = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/chunkModel/my_chunk_scaler.joblib"

    joblib.dump(scaler, scaler_file_path)
    print("Scaler for chunk model saved")

    joblib.dump(gbc_model, model_file_path)
    print(f"Model saved with accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    # Load list of processed files for human voices
    if os.path.exists(processed_files_path_human):
        print("Loading previously processed human files")
        processed_files_human = joblib.load(processed_files_path_human)
    else:
        processed_files_human = []

    human_df = pd.DataFrame()
    if os.path.exists(feature_file_path_human):
        print("Loading previously processed human features")
        human_df = joblib.load(feature_file_path_human)

    print("Starting feature extraction for human voices...")
    new_human_data = create_dataframe(human_dir, "human", processed_files_human)

    if not new_human_data.empty:
        human_df = pd.concat([human_df, new_human_data], ignore_index=True)
        joblib.dump(human_df, feature_file_path_human)
        joblib.dump(processed_files_human + new_human_data['file_name'].tolist(), processed_files_path_human)

    # Load list of processed files for AI voicesf
    if os.path.exists(processed_files_path_ai):
        print("Loading previously processed ai files")
        processed_files_ai = joblib.load(processed_files_path_ai)
    else:
        processed_files_ai = []

    ai_df = pd.DataFrame()
    if os.path.exists(feature_file_path_ai):
        print("Loading previously processed ai features")
        ai_df = joblib.load(feature_file_path_ai)

    print("Starting feature extraction for AI voices...")
    new_ai_data = create_dataframe(ai_dir, "ai", processed_files_ai)

    if not new_ai_data.empty:
        ai_df = pd.concat([ai_df, new_ai_data], ignore_index=True)
        joblib.dump(ai_df, feature_file_path_ai)
        joblib.dump(processed_files_ai + new_ai_data['file_name'].tolist(), processed_files_path_ai)

    print("Feature extraction complete. Proceeding to model training.")
    finalCalculation(ai_df, human_df)

