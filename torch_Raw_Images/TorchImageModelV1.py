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


human_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/splitsongs/'
ai_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aiTraining/'
#human_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/testdirhuman/'
#human_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/testhuman2/'
#ai_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/testdirai/'




def collate_fn(batch):
    # Sort batch by tensor size
    batch.sort(key=lambda x: x[0].shape[1], reverse=True)
    
    # Separate sequences and labels
    sequences, labels = zip(*batch)

    # Find max length
    max_len = max([s.shape[1] for s in sequences])

    # Pad all sequences to max length
    sequences_padded = [torch.nn.functional.pad(s, (0, max_len - s.shape[1], 0, 0)) for s in sequences]
    tensor_shapes = [s.shape for s in sequences_padded]
    if len(set(tensor_shapes)) > 1:  # if there are multiple unique sizes
        print("Found tensors with inconsistent sizes:", tensor_shapes)



    return torch.stack(sequences_padded), torch.tensor(labels)


class AudioClassifier3Channel(nn.Module):
    def __init__(self, input_size=(1025, 52)):
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
        self.fc1 = nn.Linear(98304, 128)
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
    chunks = [y[i:i + int(3*sr)] for i in range(0, len(y) - int(3*sr) + 1, int(3*sr))]
    multi_channel_data = [compute_audio_representations_with_waveform_fixed(chunk, sr) for chunk in chunks]
    return multi_channel_data

def process_file(args):
    dir, file_name, label = args
    file_path = os.path.join(dir, file_name)
    
    try:
        n_chunks = len(custom_audio_loader(file_path))
        return (file_path, n_chunks, label)
    except Exception as e:
        print(f"Error processing file {file_path}. Reason: {e}")
        return None

class AudioDataset(Dataset):
    def __init__(self, human_dir, ai_dir, save_dir):
        self.file_data = []  # Instead of just paths, now we're going to store (path, number of chunks)
        self.save_dir= save_dir
        self._process_and_save(human_dir, ai_dir)

     

    def _process_and_save(self, human_dir, ai_dir):
        
        
        human_files = [(human_dir, file_name, 0) for file_name in os.listdir(human_dir)]
        ai_files = [(ai_dir, file_name, 1) for file_name in os.listdir(ai_dir)]
        all_files = human_files + ai_files

        with Pool(cpu_count()) as pool:
            for res in tqdm(pool.imap(process_file, all_files), total=len(all_files), desc="Processing Audio Files"):
                self.file_data.append(res)
        
        '''
        for file_name in tqdm(os.listdir(human_dir), desc="Loading Human Data", unit="file"):
            try:
                file_path = os.path.join(human_dir, file_name)
                n_chunks = len(custom_audio_loader(file_path))  # Load the file to determine the number of chunks
                self.file_data.append((file_path, n_chunks, 0))  # 0 for human
            except:
                pass
        
        for file_name in tqdm(os.listdir(ai_dir), desc="Loading AI Data", unit="file"):
            try:
                file_path = os.path.join(ai_dir, file_name)
                n_chunks = len(custom_audio_loader(file_path))
                self.file_data.append((file_path, n_chunks, 1))  # 1 for AI

            except:
                pass
        '''
        

        # Load AI data
        
        

    def __len__(self):
        # Sum of all chunks across all files
        return sum(item[1] for item in self.file_data if item is not None)

    def __getitem__(self, idx):
        # Find the file and chunk corresponding to the index
        
        # We'll iterate through our file_data to find the file and chunk index corresponding to idx
        file_idx = 0
        while self.file_data[file_idx] is None or idx >= self.file_data[file_idx][1]:
            if self.file_data[file_idx] is not None:
                idx -= self.file_data[file_idx][1]
            file_idx += 1
        
        # Now, file_idx is the index of the file, and idx is the index of the chunk within that file
        file_path, _, label = self.file_data[file_idx]
        chunks = custom_audio_loader(file_path)
        
        return torch.tensor(chunks[idx], dtype=torch.float32), label






# Paths to directories



if __name__ == '__main__':

    print("Loading and processing audio dataset...")
    save_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchProcessedData"
    audio_dataset = AudioDataset(human_dir, ai_dir, save_dir)
    #print(f"Total audio chunks in the dataset: {len(audio_dataset)}")

    # Splitting the dataset
    print("Splitting dataset into training, validation, and test sets...")
    train_size = int(0.8 * len(audio_dataset))
    val_size = int(0.1 * len(audio_dataset))
    test_size = len(audio_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(audio_dataset, [train_size, val_size, test_size])

    # DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=11)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn, num_workers=11)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, num_workers=11)


    # Model, Loss, and Optimizer
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    print(f"Using device: {device}")
    model = AudioClassifier3Channel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 7
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            #print(inputs.shape)  # Add this line
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(inputs.to(device)), labels.to(device)) for inputs, labels in val_loader) / len(val_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Time taken: {time.time() - start_time:.2f} seconds")

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    print("Evaluating on the test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total}%")
    torch.save(model.state_dict(), '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchModels/neuralNet.pth')