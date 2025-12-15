import logging
from linebot.v3.messaging import TextMessage

logger = logging.getLogger(__name__)

def dispatch_command(text, db, user_id):
    """
    Dispatches commands based on the text input.
    Returns a ReplyMessage object (e.g., TextMessage) if a command is matched.
    Returns None if no command is matched, allowing further processing (e.g., AI reply).
    """
    if text == 'help':
        return TextMessage(text="您好！我可以協助您查詢設備狀態、異常紀錄等。您可以直接輸入問題，例如：\n'查詢設備 A 的狀態'\n'列出最近的警報'")

    # Add more commands here as needed, e.g., subscribe/unsubscribe

    return None
