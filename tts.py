"""
Text-to-Speech module using ElevenLabs API.

This module provides functionality to convert text to speech using ElevenLabs' API
and save the resulting audio as an MP3 file.
"""

import os
import time
import requests
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get ElevenLabs API key
ELEVENLABS_API_KEY = os.getenv("ELEVENLAB_API_KEY")
if not ELEVENLABS_API_KEY:
    logger.warning("ELEVENLAB_API_KEY is not set in environment variables")

def text_to_speech(text, voice_id="21m00Tcm4TlvDq8ikWAM", output_path=None):
    """
    Convert text to speech using ElevenLabs API and save the audio file.
    
    Args:
        text (str): The text to convert to speech
        voice_id (str): ElevenLabs voice ID to use (default: "21m00Tcm4TlvDq8ikWAM" - Rachel voice)
        output_path (str, optional): Path to save the audio file. If None, a temporary file is created.
    
    Returns:
        str: Path to the saved audio file, or None if conversion failed
    """
    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs API key not found")
        return None

    try:
        # Create output directory if it doesn't exist
        if output_path is None:
            # Create a temp directory for audio files if it doesn't exist
            audio_dir = Path("temp_audio")
            audio_dir.mkdir(exist_ok=True)
            
            # Generate a unique filename based on timestamp
            timestamp = int(time.time())
            output_path = f"temp_audio/tts_{timestamp}.mp3"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # ElevenLabs API URL
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        # Set up headers with API key
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        # Prepare the data payload
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        logger.info(f"Converting text to speech: {text[:50]}...")
        
        # Make the API request
        response = requests.post(url, json=data, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            # Save the audio to the specified path
            with open(output_path, 'wb') as audio_file:
                audio_file.write(response.content)
            
            logger.info(f"Audio saved to {output_path}")
            return output_path
        else:
            logger.error(f"Error generating audio: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        return None

def list_available_voices():
    """
    List all available voices from ElevenLabs API.
    
    Returns:
        list: List of dictionaries containing voice information, or empty list if request failed
    """
    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs API key not found")
        return []
        
    try:
        url = "https://api.elevenlabs.io/v1/voices"
        
        headers = {
            "Accept": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            voices_data = response.json()
            return voices_data.get("voices", [])
        else:
            logger.error(f"Error getting voices: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        logger.error(f"Error in list_available_voices: {e}")
        return []
