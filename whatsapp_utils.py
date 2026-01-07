import os
import json
import logging
import requests
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WhatsApp credentials
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")


def mark_message_as_read(message_id, max_retries=3, retry_delay=2):
    """Mark a message as read (blue tick).
    
    This uses the WhatsApp Cloud API to mark a message as read,
    which will display blue ticks to the sender.
    
    Args:
        message_id: The ID of the message to mark as read
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not message_id:
        logger.warning("Cannot mark message as read: No message ID provided")
        return False
        
    if not all([WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_TOKEN]):
        logger.error("WhatsApp credentials are not set")
        return False
    
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WHATSAPP_TOKEN}"
    }
    
    # The payload for marking a message as read
    data = {
        "messaging_product": "whatsapp",
        "status": "read", 
        "message_id": message_id
    }
    
    retry_count = 0
    last_exception = None
    
    while retry_count <= max_retries:
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            logger.info(f"Message {message_id} marked as read")
            return True
        except requests.exceptions.RequestException as e:
            last_exception = e
            retry_count += 1
            
            # Try to extract detailed error information from the response
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    error_details = f", Details: {json.dumps(error_data)}"
                except:
                    pass
                    
            if retry_count <= max_retries:
                logger.warning(f"Mark as read attempt {retry_count} failed: {str(e)}{error_details}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay exponentially for next retry
                retry_delay *= 2
            else:
                break
    
    logger.error(f"Failed to mark message as read after {max_retries} retries: {str(last_exception)}")
    return False
