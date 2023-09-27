from pydub import AudioSegment
import os

def convert_directory_to_wav(input_directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".mp3"):
            mp3_file_path = os.path.join(input_directory, filename)
            wav_file_path = os.path.join(output_directory, filename.replace(".mp3", ".wav"))

            audio = AudioSegment.from_mp3(mp3_file_path)
            audio.export(wav_file_path, format='wav')

# Paths to the directories containing the voice files
ai_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/AiTraining"
human_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/humanTraining/splitsongs"

# Paths to the output directories
ai_output_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/AiWav"
human_output_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/VoiceOnly/HumanWav"

# Convert the directories
convert_directory_to_wav(ai_dir, ai_output_dir)
convert_directory_to_wav(human_dir, human_output_dir)