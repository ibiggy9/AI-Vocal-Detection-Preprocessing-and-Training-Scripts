from pydub import AudioSegment
import os

def convert_directory_to_mp3(input_directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".wav"):
            wav_file_path = os.path.join(input_directory, filename)
            mp3_file_path = os.path.join(output_directory, filename.replace(".wav", ".mp3"))

            audio = AudioSegment.from_wav(wav_file_path)
            audio.export(mp3_file_path, format='mp3')

# Paths to the directories containing the voice files
ai_dir = "/users/main/downloads/archive (1)/kaggle/audio/real"
#human_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/HumanWav"

# Paths to the output directories
ai_output_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/humantraining"
#human_output_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/humanTraining/splitsongs"

# Convert the directories
convert_directory_to_mp3(ai_dir, ai_output_dir)
