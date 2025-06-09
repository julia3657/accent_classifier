# download.py

import requests

def download_video(url, filename="temp_video.mp4"):
    print(f"Downloading video from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception("Failed to download video.")

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    print("Download complete.")
    return filename

