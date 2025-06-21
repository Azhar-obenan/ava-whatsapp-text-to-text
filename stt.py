import os
import requests
import tempfile
import logging
from pathlib import Path
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def download_audio(url, headers):
    """
    Download audio file from the provided URL
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Create a temporary file with .ogg extension
        temp_dir = tempfile.gettempdir()
        temp_path = Path(temp_dir) / "temp_audio.ogg"
        
        # Write the audio content to the temporary file
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        logger.info(f"Audio file downloaded successfully to {temp_path}")
        return temp_path
    except Exception as e:
        logger.error(f"Error downloading audio file: {e}")
        raise

def transcribe_audio(audio_path):
    """
    Transcribe an audio file using OpenAI Whisper
    """
    try:
        # Use OpenAI's API for transcription
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        with open(audio_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe(
                model="whisper-1", 
                file=audio_file
            )
        
        transcript = transcription.get("text", "").strip()
        logger.info(f"Transcription successful: {transcript[:50]}...")
        return transcript
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return "Sorry, I couldn't transcribe your audio message."

def get_media_url(media_id, whatsapp_token):
    """
    Get the URL of a media file from WhatsApp API
    """
    try:
        url = f"https://graph.facebook.com/v17.0/{media_id}"
        headers = {
            "Authorization": f"Bearer {whatsapp_token}"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        media_data = response.json()
        media_url = media_data.get("url")
        
        if not media_url:
            raise ValueError("No URL found in media data")
        
        return media_url
    except Exception as e:
        logger.error(f"Error getting media URL: {e}")
        raise

def process_voice_message(media_id, whatsapp_token):
    """
    Process a voice message:
    1. Get media URL from WhatsApp API
    2. Download the audio file
    3. Transcribe the audio to text
    """
    try:
        # Get media URL
        media_url = get_media_url(media_id, whatsapp_token)
        logger.info(f"Got media URL for ID {media_id}")
        
        # Download audio file
        headers = {
            "Authorization": f"Bearer {whatsapp_token}"
        }
        audio_path = download_audio(media_url, headers)
        
        # Transcribe audio
        transcript = transcribe_audio(audio_path)
        
        # Clean up temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return transcript
    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        return "Sorry, I couldn't process your voice message."
