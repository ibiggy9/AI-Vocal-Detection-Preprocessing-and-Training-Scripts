# AI-Vocal-Detection-Preprocessing-Training-Programs
This repo contains the ongoing training experiments for ai voice detection.

## Image Only Model
These models ingest 3 second audio clips, extract images like spectrograms from those audios, save them in tensors and get passed along for training. As a quick note on architecture, from testing
any more than 3 layers results in overfitting. The reality is that audio data is inherently complex. You can have background noise, different accents, languages, vocal tones etc such that 
the features that distinguish ai-vocals from human are not found in the super-structures of the audio, they are found in the microstructures (the edges). So your most important layer is the first 
convolutional one. 

## Features Only
Rather than extracting those images which can be computationally expensive, this approach uses metrics from the audio files instead to try to train the model. 

## Gradient Boosting Classification
In the search for the best accuracy, I also used an off the shelf algorithm to try to help boost accuracy. In general this type of model performs very well on smaller datasets. From my experience, 
anything under a sample size of 5 million audio samples (2.5M per class) should use this approach.
