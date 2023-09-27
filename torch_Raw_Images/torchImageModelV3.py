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
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score

human_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch/data_prepped/human/3s'
ai_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torch/data_prepped/ai/3s'
#human_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/testdirhuman/'
#human_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/testhuman2/'
#ai_dir = '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/testdirai/'

def collate_fn(batch):  
    # Filter out None values from batch
    batch = [(sequence, label) for sequence, label in batch if sequence is not None and label is not None]
    
    # If after filtering, batch is empty, return None (handle this case in your training loop)
    if not batch:
        return None, None

    # Just separate sequences and labels since we're not dealing with variable-length chunks
    sequences, labels = zip(*batch)
    return torch.stack(sequences), torch.tensor(labels)

class AudioClassifier3Channel(nn.Module):
    def __init__(self, input_size=(1025, 517)):
        super(AudioClassifier3Channel, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # This line was modified
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

class AudioDataset(Dataset):
    def __init__(self, tensor_file):
        self.data, self.labels = torch.load(tensor_file)  # Assuming the saved tensor file contains (data, labels) tuple

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LazyLoadingAudioDataset(Dataset):
    def __init__(self, tensor_paths, labels):
        self.tensor_paths = tensor_paths
        self.labels = labels

    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.tensor_paths[idx])
            label = self.labels[idx]
            return data, label
        except Exception as e:  # Catch all types of exceptions
            print(f"Error loading file: {self.tensor_paths[idx]}, Error: {e}")
            return None, None

def split_data(tensor_paths, labels, train_ratio=0.8, val_ratio=0.1):
    total_size = len(tensor_paths)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    indices = torch.randperm(total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_tensor_paths = [tensor_paths[i] for i in train_indices]
    val_tensor_paths = [tensor_paths[i] for i in val_indices]
    test_tensor_paths = [tensor_paths[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    return train_tensor_paths, val_tensor_paths, test_tensor_paths, train_labels, val_labels, test_labels


def evaluate(model, test_loader, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return precision, recall, f1


if __name__ == '__main__':
    #Global Variables for early stopping and checkpointing:
    patience = 5  
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    # Getting all tensor paths for human voices
    human_tensor_paths = [os.path.join(human_dir, filename) for filename in os.listdir(human_dir) if filename.endswith('.pt')]
    

    # Getting all tensor paths for AI voices
    ai_tensor_paths = [os.path.join(ai_dir, filename) for filename in os.listdir(ai_dir) if filename.endswith('.pt')]

    human_labels = [0] * len(human_tensor_paths)  # 0 for 'human'
    ai_labels = [1] * len(ai_tensor_paths)   

    human_train_paths, human_val_paths, human_test_paths, human_train_labels, human_val_labels, human_test_labels = split_data(human_tensor_paths, human_labels)

    # Splitting AI data
    ai_train_paths, ai_val_paths, ai_test_paths, ai_train_labels, ai_val_labels, ai_test_labels = split_data(ai_tensor_paths, ai_labels)

    # Combining them
    combined_train_paths = human_train_paths + ai_train_paths
    combined_train_labels = human_train_labels + ai_train_labels

    combined_val_paths = human_val_paths + ai_val_paths
    combined_val_labels = human_val_labels + ai_val_labels

    combined_test_paths = human_test_paths + ai_test_paths
    combined_test_labels = human_test_labels + ai_test_labels

    train_dataset = LazyLoadingAudioDataset(combined_train_paths, combined_train_labels)
    val_dataset = LazyLoadingAudioDataset(combined_val_paths, combined_val_labels)
    test_dataset = LazyLoadingAudioDataset(combined_test_paths, combined_test_labels)

    # DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True, collate_fn=collate_fn, num_workers=0)
    num_batches_to_inspect = 5

    for i, (inputs, labels) in enumerate(train_loader):
        if i >= num_batches_to_inspect:
            break
        
    val_loader = DataLoader(val_dataset, batch_size=25, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=25, collate_fn=collate_fn, num_workers=0)

    # Model, Loss, and Optimizer
    device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
    print(f"Using device: {device}")
    model = AudioClassifier3Channel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Training Loop
    num_epochs = 100
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            try:
                if inputs is None or labels is None:
                    print("Skipping an empty batch...")
                    continue
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
            
            except Exception as e:
                print(f"Encountered error during training: {e}. Skipping this batch.")
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(inputs.to(device)), labels.to(device)) for inputs, labels in val_loader) / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchModels/best_model.pth')  # Save the best model
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print("Stopping early due to no improvement!")
            break

        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Time taken: {time.time() - start_time:.2f} seconds")
        print("Saving ModeL")
        torch.save(model.state_dict(), '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchModels/neuralNet_3s.pth')
        print("Model Saved")

    # Evaluation
    model.eval()
    precision, recall, f1 = evaluate(model, test_loader, device)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
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
    torch.save(model.state_dict(), '/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/torchModels/neuralNet_3s.pth')
