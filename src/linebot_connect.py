# src/linebot_connect.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from .services.line_service import process_message_in_background

logger = logging.getLogger(__name__)

channel_secret = os.getenv("LINE_CHANNEL_SECRET")
if not channel_secret:
    raise ValueError("LINE_CHANNEL_SECRET not set")

handler = WebhookHandler(channel_secret)

# Initialize a thread pool with a limited number of workers
# Adjust max_workers based on your server's capacity
executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="LineBotWorker")


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    Handles incoming text messages from LINE.
    Delegates the processing to a background thread pool to avoid webhook timeouts
    and prevent resource exhaustion.
    """
    try:
        executor.submit(process_message_in_background, event)
    except Exception as e:
        logger.error(f"Failed to submit task to thread pool: {e}")


def get_handler():
    """Returns the configured WebhookHandler instance."""
    return handler
