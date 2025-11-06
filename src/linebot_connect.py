# src/linebot_connect.py
import logging
import os
import threading
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from .services.line_service import process_message_in_background

logger = logging.getLogger(__name__)

channel_secret = os.getenv("LINE_CHANNEL_SECRET")
if not channel_secret:
    raise ValueError("LINE_CHANNEL_SECRET not set")

handler = WebhookHandler(channel_secret)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    Handles incoming text messages from LINE.
    Delegates the processing to a background thread to avoid webhook timeouts.
    """
    thread = threading.Thread(target=process_message_in_background, args=(event,))
    thread.start()

def get_handler():
    """Returns the configured WebhookHandler instance."""
    return handler
