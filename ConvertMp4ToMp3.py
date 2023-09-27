import os
import subprocess

def convert_mp4_to_mp3(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename.replace(".mp4", ".mp3"))
            
            # Command to convert MP4 to MP3 using ffmpeg
            cmd = ["ffmpeg", "-i", input_filepath, "-q:a", "0", "-map", "a", output_filepath]
            
            subprocess.run(cmd)
            print(f"Converted {filename} to MP3")

if __name__ == "__main__":
    input_directory = "/Users/main/downloads/steven" # Replace with your directory path
    output_directory = "/Users/main/downloads/steven/mp3"      # Replace with your desired output directory path
    
    convert_mp4_to_mp3(input_directory, output_directory)
