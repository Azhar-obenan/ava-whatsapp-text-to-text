import base64
import logging
import os
import tempfile
from typing import Tuple
import openai
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TextToImage:
    """A class to handle text-to-image generation using Together AI."""

    def __init__(self):
        """
        Initialize the TextToImage class to generate images from text prompts
        using OpenAI's DALL-E.
        """
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        openai.api_key = self.openai_api_key
        self.logger = logging.getLogger(__name__)

    def generate_image(self, prompt: str) -> tuple[bytes, str]:
        """
        Generate an image based on the text prompt using OpenAI's DALL-E.
        
        Args:
            prompt: The text prompt to guide the image generation.
            
        Returns:
            Tuple of (image_data, temp_file_path)
        """
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            self.logger.info(f"Generating image for prompt: '{prompt}'")

            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json"
            )

            # Decode the base64 image
            image_data = base64.b64decode(response['data'][0]['b64_json'])
            
            # Save to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file.write(image_data)
            temp_file_path = temp_file.name
            temp_file.close()
            
            self.logger.info(f"Image saved to {temp_file_path}")
            
            return image_data, temp_file_path

        except Exception as e:
            self.logger.error(f"Failed to generate image: {str(e)}")
            raise ValueError(f"Failed to generate image: {str(e)}")

# Simple function for use in the app
def generate_image_from_prompt(prompt: str) -> tuple[bytes, str]:
    """Generate an image from a text prompt"""
    tti = TextToImage()
    return tti.generate_image(prompt)
