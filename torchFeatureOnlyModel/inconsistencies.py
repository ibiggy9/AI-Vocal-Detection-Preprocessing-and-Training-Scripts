import pickle
import numpy as np

SAVED_FEATURES_PATH = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/torch_features_only/saved_features/extracted_features.pkl"


with open(SAVED_FEATURES_PATH, 'rb') as f:
    data = pickle.load(f)
    features = data['features']

# Calculate the shapes of all feature sets
shapes = [np.array(feature_set).shape for feature_set in features]

# Find the most common shape
most_common_shape = max(set(shapes), key=shapes.count)
print(shapes)
print("Most common shape:", most_common_shape)

