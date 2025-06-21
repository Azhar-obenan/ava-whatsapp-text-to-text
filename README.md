# WhatsApp Text-to-Text Agent

This is a simple WhatsApp Text-to-Text agent that uses Groq's LLM to process and respond to WhatsApp messages.

## Prerequisites

- Python 3.8 or higher
- Groq API key (signup at [groq.com](https://console.groq.com))
- WhatsApp Business API access via Meta

## Setup Instructions

1. **Clone/Navigate to the project directory**

```bash
cd /Users/prom1/Documents/Ava-whatsapp/whatsapp-text-to-text
```

2. **Create a virtual environment and activate it**

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
# venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**

Edit the `.env` file in the project root and add your API keys:

```env
# API Keys configuration
GROQ_API_KEY="your_groq_api_key"

# WhatsApp Cloud API configuration
WHATSAPP_PHONE_NUMBER_ID="your_phone_number_id"
WHATSAPP_TOKEN="your_whatsapp_token"
WHATSAPP_VERIFY_TOKEN="your_verification_token"
```

- `GROQ_API_KEY`: Your API key from Groq
- `WHATSAPP_PHONE_NUMBER_ID`: Your WhatsApp phone number ID from Meta Business dashboard
- `WHATSAPP_TOKEN`: Your WhatsApp access token
- `WHATSAPP_VERIFY_TOKEN`: A custom verification token you choose for webhook setup

5. **Run the application**

```bash
python app.py
```

The server will start on `http://0.0.0.0:8080`.

## Setting up WhatsApp Business API

1. Go to [Meta Developer Portal](https://developers.facebook.com/)
2. Create an app or use an existing one
3. Add WhatsApp to your app
4. Set up Webhooks with:
   - Callback URL: `https://your-domain.com/webhook` (you'll need to expose your local server using ngrok or similar)
   - Verify Token: The same one you set in your `.env` file
5. Subscribe to the messages webhook

## Using ngrok for development

To expose your local server for webhook testing:

```bash
# Install ngrok if you haven't already
# Run ngrok on port 8080
ngrok http 8080
```

Use the ngrok URL as your webhook callback URL in the Meta Developer Portal.

## How It Works

1. WhatsApp sends incoming messages to your webhook
2. The application processes the message using Groq's LLM
3. A response is generated and sent back to the user via WhatsApp

## Customization

- Modify the system prompt in `app.py` to change the assistant's personality
- Change the LLM model in the `get_llm_client()` function 
- Adjust temperature and other parameters as needed

## Troubleshooting

- Check the logs for detailed error messages
- Verify your API keys are correctly set
- Ensure your webhook is properly configured and accessible
