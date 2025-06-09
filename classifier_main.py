import torch
import torchaudio
import librosa
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from speechbrain.pretrained.interfaces import foreign_class
import speech_recognition as sr
from pydub import AudioSegment
import io
import warnings
import os
warnings.filterwarnings('ignore')

class EnglishAccentAnalyzer:
    """
    English Accent Analyzer using pretrained SpeechBrain models from Hugging Face
    
    Available Models:
    1. ECAPA-TDNN model: 87% accuracy, 16 English accents
    2. XLSR model: 95% accuracy, 16 English accents
    """
    
    def __init__(self):
        """
        Initialize the analyzer with a pretrained model
        """
        self.classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Accent labels (16 English accents)
        self.accent_labels = [
            'african', 'australia', 'bermuda', 'canada', 'england', 
            'hongkong', 'indian', 'ireland', 'malaysia', 'newzealand', 
            'philippines', 'scotland', 'singapore', 'southatlantic', 
            'us', 'wales'
        ]
        
        # More readable accent names
        self.accent_display_names = {
            'african': 'South African',
            'australia': 'Australian',
            'bermuda': 'Bermudian',
            'canada': 'Canadian',
            'england': 'British (English)',
            'hongkong': 'Hong Kong',
            'indian': 'Indian',
            'ireland': 'Irish',
            'malaysia': 'Malaysian',
            'newzealand': 'New Zealand',
            'philippines': 'Filipino',
            'scotland': 'Scottish',
            'singapore': 'Singaporean',
            'southatlantic': 'South Atlantic',
            'us': 'American',
            'wales': 'Welsh'
        }
        
        # Initialize speech recognizer for English detection
        self.recognizer = sr.Recognizer()
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained model from Hugging Face"""
        try:
            print("Loading ECAPA-TDNN model (87% accuracy)...")
            self.classifier = EncoderClassifier.from_hparams(
                    source="Jzuluaga/accent-id-commonaccent_ecapa",
                    savedir="pretrained_models/accent-id-commonaccent_ecapa",
                    run_opts={"device": str(self.device)}
            )

            print(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("Installing SpeechBrain: pip install speechbrain")
            raise e
    
    def detect_english_speech(self, audio_path):
        """
        Detect if the audio contains English speech using speech recognition
        """
        try:
            # Load and convert audio for speech recognition
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Convert to temporary wav for speech recognition
            temp_wav = io.BytesIO()
            audio.export(temp_wav, format="wav")
            temp_wav.seek(0)
            
            # Perform speech recognition
            with sr.AudioFile(temp_wav) as source:
                audio_data = self.recognizer.record(source)
            
            try:
                # Try to recognize as English
                text = self.recognizer.recognize_google(audio_data, language='en-US')
                confidence = min(100, max(50, len(text.split()) * 10))  # Rough confidence estimate
                return True, confidence, text
            except sr.UnknownValueError:
                return False, 0, "Could not understand speech"
            except sr.RequestError as e:
                print(f"Speech recognition service error: {e}")
                return None, 0, "Speech recognition unavailable"
                
        except Exception as e:
            print(f"Error in English detection: {e}")
            return None, 0, f"Error: {str(e)}"
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file for the model (ensure 16kHz, mono)
        """
        try:
            # Load audio with librosa (automatically converts to mono and resamples)
            signal, sr = librosa.load(audio_path, sr=16000, mono=True)
            return signal, sr
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            return None, None
    
    def classify_accent(self, audio_path):
        """
        Classify the accent in the audio file
        
        Returns:
            tuple: (probabilities, score, predicted_index, predicted_label)
        """
        try:
            # Use the classifier to predict accent
            out_prob, score, index, text_lab = self.classifier.classify_file(audio_path)
            
            return out_prob, score, index, text_lab
            
        except Exception as e:
            print(f"Error in accent classification: {e}")
            return None, None, None, None
    
    def get_top_predictions(self, probabilities, top_k=3):
        """
        Get top-k accent predictions with confidence scores
        """
        if probabilities is None:
            return []
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(probabilities):
            probs = probabilities.cpu().numpy().flatten()
        else:
            probs = np.array(probabilities).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            accent_key = self.accent_labels[idx] if idx < len(self.accent_labels) else f"unknown_{idx}"
            accent_name = self.accent_display_names.get(accent_key, accent_key.title())
            confidence = float(probs[idx]) * 100
            
            predictions.append({
                'accent': accent_name,
                'confidence': round(confidence, 1),
                'accent_code': accent_key
            })
        
        return predictions
    
    def analyze_accent(self, audio_path):
        """
        Complete accent analysis pipeline
        
        Returns:
            dict: Analysis results with classification, confidence, and summary
        """
        print(f"\n🎤 Analyzing audio: {audio_path}")
        
        results = {
            'classification': 'Error',
            'confidence': 0,
            'english_confidence': 0,
            'summary': '',
            'top_predictions': [],
            'model_used': 'ECAPA-TDNN (87% accuracy)',
            'transcript_sample': None
        }
        
        # Step 1: Check if audio contains English speech
        print("🔍 Detecting English speech...")
        is_english, speech_confidence, transcript = self.detect_english_speech(audio_path)
        
        if is_english is None:
            results['summary'] = f"Speech recognition error: {transcript}"
            return results
        
        if not is_english:
            results['classification'] = 'Non-English or No Speech Detected'
            results['summary'] = 'No clear English speech detected in the audio file'
            return results
        
        print(f"✅ English speech detected (confidence: {speech_confidence}%)")
        results['english_confidence'] = speech_confidence
        results['transcript_sample'] = transcript[:100] if transcript else None
        
        # Step 2: Classify accent using pretrained model
        print("🌍 Classifying accent...")
        try:
            out_prob, score, index, predicted_label = self.classify_accent(audio_path)
            
            if out_prob is None:
                results['summary'] = 'Failed to classify accent'
                return results
            
            # Get top predictions
            top_preds = self.get_top_predictions(out_prob, top_k=3)
            results['top_predictions'] = top_preds
            
            if top_preds:
                # Main prediction
                main_prediction = top_preds[0]
                results['classification'] = main_prediction['accent']
                results['confidence'] = main_prediction['confidence']
                
                # Create summary
                model_accuracy = "87%"
                model = "ECAPA-TDNN"
                results['summary'] = f"Detected {main_prediction['accent']} accent with {main_prediction['confidence']}% confidence using {model} model ({model_accuracy} accuracy). "
                
                if len(top_preds) > 1:
                    results['summary'] += f"Alternative predictions: {top_preds[1]['accent']} ({top_preds[1]['confidence']}%)"
                    if len(top_preds) > 2:
                        results['summary'] += f", {top_preds[2]['accent']} ({top_preds[2]['confidence']}%)"
                    results['summary'] += ". "
                
                if transcript:
                    results['summary'] += f"Speech sample: '{transcript[:50]}...'"
                
                print(f"🎯 Prediction: {main_prediction['accent']} ({main_prediction['confidence']}%)")
            
        except Exception as e:
            results['summary'] = f"Classification error: {str(e)}"
            print(f"❌ Classification error: {e}")
        
        return results

def analyze_wav_accent(audio_path):
    """
    Convenience function to analyze accent in a WAV file using pretrained models
    
    Args:
        audio_path (str): Path to the audio file (.wav, .mp3, etc.)
        
    Returns:
        dict: Analysis results with classification, confidence, and summary
    """
    analyzer = EnglishAccentAnalyzer()
    return analyzer.analyze_accent(audio_path)

def run_analysis_with_choice(audio_path = 'output_audio.wav'):
    """
    Run the accent analysis with user-selected model and audio file
    """
    try:
        
        print(f"\n📁 Analyzing: {audio_path}")
        print("-" * 50)
        
        # Run analysis
        result = analyze_wav_accent(audio_path)
        return result
                
    except FileNotFoundError:
        print(f"❌ Audio file not found.")
        print("\n📋 Supported audio formats: .wav, .mp3, .flac, .m4a, .ogg")
        return None

# Main execution
if __name__ == "__main__":
    run_analysis_with_choice()  