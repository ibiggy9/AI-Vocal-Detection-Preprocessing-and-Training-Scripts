from pydub import AudioSegment
import math
import os
from multiprocessing import Pool, cpu_count, current_process
from tqdm import tqdm

def split_song(args):
    filename, split_lengths, output_dirs = args
    song = AudioSegment.from_mp3(filename)
    song_length_in_sec = len(song) // 1000

    for split_length, output_dir in zip(split_lengths, output_dirs):
        num_chunks = math.ceil(song_length_in_sec / split_length)
        for i in range(num_chunks):
            start_time = i * split_length * 1000
            end_time = (i + 1) * split_length * 1000

            chunk = song[start_time:end_time]
            chunk.export(os.path.join(output_dir, f"{os.path.basename(filename)}_chunk_{i}.mp3"), format="mp3")

def process_directory(input_dir, output_dirs, split_lengths):
    # Create the output directories if they don't exist
    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            print(f"Making Directory: {output_dir}")
            os.mkdir(output_dir)

    tasks = [(os.path.join(input_dir, filename), split_lengths, output_dirs) 
             for filename in os.listdir(input_dir) if filename.endswith(".mp3")]

    print("Running tasks")
    with Pool(cpu_count()) as p:
        list(tqdm(p.imap(split_song, tasks), total=len(tasks)))

if __name__ == "__main__":
    input_dir = "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining"
    output_dirs = [
        "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining/1s",
        "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining/2s",
        "/Users/main/Desktop/projects/businesses/AI-SPY/trainer/voiceonly/aitraining/3s"
    ]
    split_lengths = [1, 2, 3]
    process_directory(input_dir, output_dirs, split_lengths)
