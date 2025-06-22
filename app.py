import os
import json
import logging
import time
from typing import Dict, Any
from dotenv import load_dotenv
import requests
import uvicorn
from fastapi import FastAPI, Request, Response, BackgroundTasks, Form, HTTPException
from fastapi.responses import JSONResponse

# LLM integration
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Import custom modules
import stt
import text_to_image
import image_to_text
import whatsapp_utils

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="WhatsApp Text-to-Text Agent")

# Get environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM client
def get_llm_client():
    """Initialize and return the LLM client"""
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set in environment variables")
        raise ValueError("GROQ_API_KEY is not set")
    
    return ChatGroq(
        model="llama3-70b-8192",  # You can change model as needed
        temperature=0.7,
        groq_api_key=GROQ_API_KEY
    )

def upload_whatsapp_media(file_path: str, mime_type: str = "image/jpeg"):
    """Upload a media file to WhatsApp servers and return the media ID"""
    if not all([WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_TOKEN]):
        logger.error("WhatsApp credentials not set")
        raise ValueError("WhatsApp credentials not set")
    
    try:
        # API endpoint for uploading media
        url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
        
        # Open the file and prepare the request
        with open(file_path, "rb") as file_data:
            file_content = file_data.read()
        
        # Prepare the files parameter with the correct format
        files = {
            "messaging_product": (None, "whatsapp"),
            "file": (os.path.basename(file_path), file_content, mime_type)
        }
        
        # Send the file upload request
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}"
        }
        
        response = requests.post(url, files=files, headers=headers)
        response.raise_for_status()
        
        # Return the media ID
        data = response.json()
        logger.info(f"Media upload response: {data}")
        return data.get("id")
    
    except Exception as e:
        logger.error(f"Error uploading media to WhatsApp: {e}")
        raise ValueError(f"Error uploading media: {str(e)}")


def send_whatsapp_image(to: str, image_id: str, caption: str = ""):
    """Send an image message to a WhatsApp user"""
    if not all([WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_TOKEN]):
        logger.error("WhatsApp credentials not set")
        raise ValueError("WhatsApp credentials not set")
    
    try:
        # API endpoint for sending messages
        url = f"https://graph.facebook.com/v19.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
        
        # Prepare the request payload
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "image",
            "image": {
                "id": image_id,
                "caption": caption
            }
        }
        
        # Send the request
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        logger.info(f"Image sent successfully to {to}")
        return response.json()
    
    except Exception as e:
        logger.error(f"Error sending image via WhatsApp: {e}")
        raise ValueError(f"Error sending image: {str(e)}")


def send_whatsapp_message(to: str, message: str, max_retries: int = 3, retry_delay: int = 2):
    """Send a message to WhatsApp user with retry mechanism
    
    Args:
        to: Recipient's phone number
        message: Message content
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
    """
    if not all([WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_TOKEN]):
        logger.error("WhatsApp credentials are not set")
        return
    
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {WHATSAPP_TOKEN}"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    
    retry_count = 0
    last_exception = None
    
    while retry_count <= max_retries:
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            logger.info(f"Message sent successfully to {to}")
            return True
        except requests.exceptions.RequestException as e:
            last_exception = e
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(f"WhatsApp message send attempt {retry_count} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay exponentially for next retry
                retry_delay *= 2
            else:
                break
    
    logger.error(f"Error sending WhatsApp message after {max_retries} retries: {str(last_exception)}")
    return False


def detect_image_generation_intent(message_text: str) -> tuple[bool, str]:
    """Detect if a message is asking for image generation and extract the prompt.
    
    Returns:
        tuple: (is_image_request, image_prompt)
    """
    # First check for the explicit /image command
    if message_text.startswith("/image"):
        # Extract the prompt by removing '/image ' from the start
        prompt = message_text[7:].strip()
        return True, prompt
    
    # For natural language requests, use the LLM to detect intent
    try:
        # Initialize LLM
        llm = get_llm_client()
        
        # Create system prompt for classification
        system_prompt = """
        You are an AI that determines if a message is asking for image generation.
        If the message is asking you to create, generate, draw, or show an image or picture, 
        respond with 'YES' followed by a comma and then the image description.
        If the message is not asking for an image, respond with 'NO'.
        
        Examples:
        Input: "Generate an image of a cat"
        Output: YES, a cat
        
        Input: "Show me a picture of mountains at sunset"
        Output: YES, mountains at sunset
        
        Input: "How does photosynthesis work?"
        Output: NO
        
        Input: "Can you draw a forest?"
        Output: YES, a forest
        
        Keep your response format strictly as:  
        YES, [image description]   
        or just:  
        NO
        """
        
        # Generate response using LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_text)
        ]
        
        response = llm.invoke(messages)
        result = response.content.strip()
        
        # Parse the response
        if result.upper().startswith("YES"):
            parts = result.split(",", 1)
            if len(parts) > 1:
                return True, parts[1].strip()
            else:
                # If the format is not as expected, use the original message as prompt
                return True, message_text
        else:
            return False, ""
            
    except Exception as e:
        logger.error(f"Error detecting image intent: {e}")
        # Default to not an image request on error
        return False, ""

def get_groq_response(message_text: str) -> str:
    """Generate a response using the Groq LLM"""
    try:
        # Initialize LLM
        llm = get_llm_client()
        
        # Create system prompt
        system_prompt = """You are Ava, a helpful WhatsApp AI assistant. 
        Provide clear, concise, and friendly responses.
        Keep your answers brief and to the point, ideal for mobile reading.
        Be helpful, accurate, and respectful in all communications."""
        
        # Generate response using LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message_text)
        ]
        
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm sorry, I'm having trouble processing your request right now."

def process_text_message(sender_id: str, message_text: str):
    """Process incoming text message and generate a response"""
    try:
        # Generate response using LLM
        response_text = get_groq_response(message_text)
        
        # Send the response back to the user
        send_whatsapp_message(sender_id, response_text)
    except Exception as e:
        logger.error(f"Error processing text message: {e}")
        # Send an error message to the user
        send_whatsapp_message(sender_id, "I'm sorry, I'm having trouble processing your text message right now.")

def process_audio_message(sender_id: str, media_id: str):
    """Process incoming audio/voice message and generate a response"""
    try:
        # First, transcribe the audio
        logger.info(f"Processing voice message from {sender_id} with media ID: {media_id}")
        
        # Use STT module to transcribe audio
        transcript = stt.process_voice_message(media_id, WHATSAPP_TOKEN)
        logger.info(f"Transcription: {transcript}")
        
        if transcript:
            # Generate response to the transcribed text
            response_text = get_groq_response(transcript)
            
            # Send the response back to the user
            send_whatsapp_message(sender_id, response_text)
        else:
            send_whatsapp_message(sender_id, "I couldn't understand your voice message. Could you please try again?")
    except Exception as e:
        logger.error(f"Error processing audio message: {e}")
        # Send an error message to the user
        send_whatsapp_message(sender_id, "I'm sorry, I'm having trouble processing your voice message right now.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "WhatsApp Text-to-Text Agent API is running"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Verify webhook for WhatsApp API integration"""
    # Parse query parameters
    query_params = dict(request.query_params)
    
    # Extract verification parameters
    mode = query_params.get("hub.mode")
    token = query_params.get("hub.verify_token")
    challenge = query_params.get("hub.challenge")
    
    # Verify the webhook
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        if not challenge:
            return HTTPException(status_code=400, detail="Missing challenge parameter")
        
        logger.info("Webhook verified successfully")
        return Response(content=challenge)
    
    # If verification fails
    logger.warning("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming webhook events from WhatsApp"""
    try:
        # Parse the request body
        body = await request.json()
        
        # Log the incoming webhook (optional for debugging)
        logger.debug(f"Received webhook: {json.dumps(body)}")
        
        # Check if this is a valid WhatsApp message
        if "entry" in body and body.get("object") == "whatsapp_business_account":
            for entry in body["entry"]:
                if "changes" in entry:
                    for change in entry["changes"]:
                        if "value" in change and "messages" in change["value"]:
                            for message in change["value"]["messages"]:
                                # Extract sender ID and message ID
                                sender_id = message["from"]
                                message_id = message["id"]
                                
                                # Mark the message as read (blue tick)
                                background_tasks.add_task(whatsapp_utils.mark_message_as_read, message_id)
                                
                                # Handle different message types
                                if message.get("type") == "text":
                                    # Get the message text
                                    message_text = message["text"]["body"]
                                    
                                    # Check if this is an image generation request (either with /image or natural language)
                                    is_image_request, image_prompt = detect_image_generation_intent(message_text)
                                    
                                    if is_image_request:
                                        logger.info(f"Image generation request from {sender_id}: {image_prompt}")
                                        
                                        if not image_prompt:
                                            send_whatsapp_message(
                                                sender_id, 
                                                "Please describe what kind of image you'd like me to generate."
                                            )
                                        else:
                                            # Process the image generation in the background
                                            background_tasks.add_task(
                                                process_image_generation,
                                                sender_id,
                                                image_prompt
                                            )
                                    else:
                                        # Process regular text message
                                        logger.info(f"Message from {sender_id}: {message_text}")
                                        
                                        # Process the message in the background
                                        background_tasks.add_task(
                                            process_text_message, 
                                            sender_id, 
                                            message_text
                                        )
                                
                                elif message.get("type") == "audio" or message.get("type") == "voice":
                                    # Process audio/voice message
                                    media_id = message[message["type"]]["id"]
                                    logger.info(f"Voice message from {sender_id}, media ID: {media_id}")
                                    
                                    # Process the audio in the background
                                    background_tasks.add_task(
                                        process_audio_message,
                                        sender_id,
                                        media_id
                                    )
                                
                                elif message.get("type") == "image":
                                    # Process image message
                                    media_id = message["image"]["id"]
                                    logger.info(f"Image message from {sender_id}, media ID: {media_id}")
                                    
                                    # Process the image in the background
                                    background_tasks.add_task(
                                        process_image_message,
                                        sender_id,
                                        media_id
                                    )
        
        # Always return a 200 OK response to acknowledge receipt
        return JSONResponse(content={"status": "received"})
    
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error processing webhook: {str(e)}"}
        )

def process_image_generation(sender_id: str, prompt: str):
    """Process image generation request and send image back to user"""
    try:
        # Send a message indicating that image generation is in progress
        send_whatsapp_message(sender_id, "Generating your image... Please wait.")
        
        # Generate the image using the Text-to-Image module
        _, image_path = text_to_image.generate_image_from_prompt(prompt)
        logger.info(f"Generated image at path: {image_path}")
        
        # Upload the image to WhatsApp's servers
        media_id = upload_whatsapp_media(image_path)
        logger.info(f"Uploaded image to WhatsApp, media ID: {media_id}")
        
        # Send the image to the user without a caption
        send_whatsapp_image(sender_id, media_id, "")
        logger.info(f"Image sent to {sender_id}")
        
        # Clean up the temporary file
        try:
            os.remove(image_path)
            logger.info(f"Removed temporary image file: {image_path}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary image file: {e}")
    
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        send_whatsapp_message(sender_id, f"I'm sorry, I couldn't generate that image: {str(e)}")


def process_image_message(sender_id: str, media_id: str):
    """Process an image received from the user and send back a description."""
    try:
        # Send a message indicating that image analysis is in progress
        send_whatsapp_message(sender_id, "Analyzing your image... Please wait.")
        
        # Get WhatsApp token from environment
        whatsapp_token = os.getenv("WHATSAPP_TOKEN")
        if not whatsapp_token:
            raise ValueError("WHATSAPP_TOKEN environment variable is not set")
            
        # Process the image using image_to_text module
        description = image_to_text.process_image_file(media_id, whatsapp_token)
        
        # Send the description back to the user
        send_whatsapp_message(sender_id, description)
        logger.info(f"Image analysis sent to {sender_id}")
        
    except Exception as e:
        logger.error(f"Error processing image message: {e}")
        send_whatsapp_message(sender_id, f"I'm sorry, I couldn't analyze that image: {str(e)}")


if __name__ == "__main__":
    # Validate environment variables before starting
    missing_vars = []
    if not GROQ_API_KEY:
        missing_vars.append("GROQ_API_KEY")
    if not WHATSAPP_PHONE_NUMBER_ID:
        missing_vars.append("WHATSAPP_PHONE_NUMBER_ID")
    if not WHATSAPP_TOKEN:
        missing_vars.append("WHATSAPP_TOKEN")
    if not WHATSAPP_VERIFY_TOKEN:
        missing_vars.append("WHATSAPP_VERIFY_TOKEN")
    if not OPENAI_API_KEY:
        missing_vars.append("OPENAI_API_KEY")
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        print(f"Warning: The following environment variables are not set: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file before running in production.")
    
    # Start the server
    print("Starting WhatsApp Text-to-Text Agent server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
