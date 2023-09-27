'''
import xgboost 
import joblib
import pickle

model = joblib.load('./chunkModel/my_chunk_model.joblib')
scaler = joblib.load('./chunkModel/my_chunk_scaler.joblib')

model.save_model('model.xgb')

# Loading the model
loaded_model = xgboost.Booster()
loaded_model.load_model('model.xgb')

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

'''

import shap
import xgboost

# Load your model
loaded_model = xgboost.Booster()
loaded_model.load_model('./model.xgb')

# Create a SHAP explainer
explainer = shap.TreeExplainer(loaded_model)

sample_data = "/Users/Main/Downloads/vocals.mp3"

# Assume 'sample_data' is the data point you want to explain (after scaling)
shap_values = explainer.shap_values(sample_data)

# Plot the SHAP values
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, sample_data)