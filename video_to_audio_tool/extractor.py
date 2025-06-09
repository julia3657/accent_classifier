# extract.py
from moviepy import VideoFileClip

def extract_audio(video_path, output_audio_path="output_audio.wav"):
    print(f"Extracting audio from {video_path}...")
    video = VideoFileClip(video_path)
    if video.audio is None:
        video.close()  # close before raising
        raise ValueError("No audio stream found in the video.")
    video.audio.write_audiofile(output_audio_path)
    video.close()  # important to release file lock on Windows
    print(f"Audio saved to {output_audio_path}")
    return True