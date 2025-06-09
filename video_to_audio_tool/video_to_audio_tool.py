import os
from video_to_audio_tool.downloader import download_video
from video_to_audio_tool.extractor import extract_audio

def video_to_audio_tool_main_flow(video_url):
    video_path = None
    try:
        video_path = download_video(video_url)
        extracted = extract_audio(video_path)
        if extracted:
            return {
                    'success': True,
                    'audio_path': 'output_audio.wav',  
                    'message': 'Audio extracted successfully'
            }
        else:
            return {
                'success': False,
                'error': 'Audio extraction failed - no audio stream found'
            }
    except Exception as e:
            return {
                'success': False,
                'error': 'Audio extraction failed - invalid audio path'
            }
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except PermissionError:
                print(f"Could not delete {video_path}, it is still in use.")

if __name__ == "__main__":
    video_to_audio_tool_main_flow()
