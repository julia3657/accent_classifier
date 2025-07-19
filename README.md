# Accent Identification Web App

This repository contains a web-based application for identifying English speaker accents from video files using a pretrained ECAPA-TDNN model.

## Features

* **Accent Detection** from uploaded video files
* **Pretrained ECAPA-TDNN Model** for high accuracy
* **Web Interface** with simple drag-and-drop support
* **Automatic Video to Audio Conversion**
* Easy to run locally with `Flask`

## Project Structure

```
├── pretrained_models/
│   └── accent-id-commonaccent_ecapa/      # Pretrained ECAPA-TDNN model
├── templates/
│   └── index.html                          # Web UI template
├── video_to_audio_tool/
│   └── (conversion scripts and utilities)
├── app.py                                 # Flask app to serve the UI
├── classifier_main.py                     # Inference logic using ECAPA
├── requirements.txt                       # Python dependencies
└── README.md                              # Project documentation
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/accent-id-app.git
cd accent-id-app
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

### 3. Download/Verify Pretrained Model

Ensure the ECAPA-TDNN model is present under `pretrained_models/accent-id-commonaccent_ecapa/`. If not, download it from the [CommonAccent project](https://github.com/speechbrain/accents).

### 4. Run the app

```bash
python app.py
```

Then open your browser and go to:
`http://127.0.0.1:5000`

## Model Information

* **Model**: ECAPA-TDNN
* **Source**: [SpeechBrain](https://speechbrain.readthedocs.io/)
* **Task**: Accent Classification (CommonAccent dataset)

## Requirements
All dependencies are listed in `requirements.txt`.

## Tools

* **video\_to\_audio\_tool/** contains scripts to extract audio from video input.
* **classifier\_main.py** handles preprocessing and inference using the pretrained model.

## Acknowledgements

* [SpeechBrain](https://speechbrain.readthedocs.io/) for ECAPA-TDNN model and training framework.
* CommonAccent Dataset authors.

