from pydub import AudioSegment
import math
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def split_song(args):
    filename, split_length, output_dir = args
    song = AudioSegment.from_mp3(filename)

    # Calculate the number of chunks
    song_length_in_sec = len(song) // 1000  # pydub works in milliseconds
    num_chunks = math.ceil(song_length_in_sec / split_length)

    # Split the song and save the chunks
    base_name = os.path.splitext(os.path.basename(filename))[0]
    for i in range(num_chunks):
        start_time = i * split_length * 1000  # pydub works in milliseconds
        end_time = (i + 1) * split_length * 1000  # pydub works in milliseconds

        chunk = song[start_time:end_time]
        chunk.export(os.path.join(output_dir, f"{base_name}_chunk_{i}.mp3"), format="mp3")

def process_directory(input_dir, output_dir, split_length):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print("Making Directory: splitSongs")
        os.mkdir(output_dir)

    # List all the MP3 files in the input directory
    tasks = []
    for filename in os.listdir(input_dir):
        print(filename)
        if filename.endswith(".mp3"):
            tasks.append((os.path.join(input_dir, filename), split_length, output_dir))
            
    # Use multiprocessing to process the tasks with progress bar
    print("Running tasks")
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(split_song, tasks), total=len(tasks)))

if __name__ == "__main__":
    #process_directory("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining", "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining/1s", 1)
    #process_directory("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining", "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining/2s", 2)
    #process_directory("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/splitsongs", "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/2s", 2)
    process_directory("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining", "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining/5s", 5)
    process_directory("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/splitsongs", "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/5s", 5)
    #process_directory("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/splitsongs", "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/2s", 2)
    #process_directory("/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/splitsongs", "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/humantraining/1s", 1)

