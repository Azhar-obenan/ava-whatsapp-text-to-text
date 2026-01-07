import base64
import logging
import os
import tempfile
from typing import Optional, Union
import requests
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

class ImageToText:
    """A class to handle image-to-text conversion using OpenAI's vision capabilities."""

    def __init__(self):
        """Initialize the ImageToText class."""
        self.logger = logging.getLogger(__name__)
        
        # Validate API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Missing required environment variable: OPENAI_API_KEY")
            
        # Set the API key for OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def analyze_image(self, image_data: Union[str, bytes], prompt: str = "") -> str:
        """Analyze an image using OpenAI's vision capabilities.

        Args:
            image_data: Either a file path (str) or binary image data (bytes)
            prompt: Optional prompt to guide the image analysis

        Returns:
            str: Description or analysis of the image

        Raises:
            ValueError: If the image data is empty or invalid
            Exception: If the image analysis fails
        """
        try:
            # Handle file path
            if isinstance(image_data, str):
                if not os.path.exists(image_data):
                    raise ValueError(f"Image file not found: {image_data}")
                with open(image_data, "rb") as f:
                    image_bytes = f.read()
            else:
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data cannot be empty")

            # Convert image to base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            # Default prompt if none provided
            if not prompt:
                prompt = "Please describe what you see in this image in detail."

            # For OpenAI 0.28.0, we need to use a different format for vision analysis
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            if not response.choices or len(response.choices) == 0:
                raise Exception("No response received from the vision model")

            description = response.choices[0].message['content']
            self.logger.info(f"Generated image description: {description}")

            return description

        except Exception as e:
            self.logger.error(f"Failed to analyze image: {str(e)}")
            raise Exception(f"Failed to analyze image: {str(e)}") from e

def download_whatsapp_media(media_id: str, whatsapp_token: str) -> str:
    """Download media from WhatsApp using the media ID.
    
    Args:
        media_id: The ID of the media to download
        whatsapp_token: The WhatsApp token for authentication
        
    Returns:
        str: Path to the downloaded media file
    """
    try:
        # First, get the media URL
        media_url_endpoint = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {"Authorization": f"Bearer {whatsapp_token}"}
        
        response = requests.get(media_url_endpoint, headers=headers)
        response.raise_for_status()
        
        media_data = response.json()
        if "url" not in media_data:
            raise ValueError(f"Media URL not found in response: {media_data}")
        
        # Download the media file
        download_response = requests.get(media_data["url"], headers=headers)
        download_response.raise_for_status()
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(download_response.content)
        temp_file_path = temp_file.name
        temp_file.close()
        
        logging.info(f"Downloaded media file to {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        logging.error(f"Error downloading WhatsApp media: {e}")
        raise Exception(f"Failed to download media: {str(e)}")

def process_image_file(media_id: str, whatsapp_token: str, prompt: str = "") -> str:
    """Process an image file from WhatsApp and return a description.
    
    Args:
        media_id: The WhatsApp media ID
        whatsapp_token: The WhatsApp token for authentication
        prompt: Optional prompt to guide the image analysis
        
    Returns:
        str: Description of the image
    """
    try:
        # Download the image
        image_path = download_whatsapp_media(media_id, whatsapp_token)
        
        # Analyze the image
        itt = ImageToText()
        description = itt.analyze_image(image_path, prompt)
        
        # Clean up the temporary file
        try:
            os.remove(image_path)
            logging.info(f"Removed temporary image file: {image_path}")
        except Exception as e:
            logging.warning(f"Failed to remove temporary image file: {e}")
        
        return description
    
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return f"I couldn't analyze that image: {str(e)}"
