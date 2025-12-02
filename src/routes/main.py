# src/routes/main.py

from flask import request, abort, render_template
from . import main_bp
from linebot.v3.exceptions import InvalidSignatureError
import logging

# This will be initialized in app.py
handler = None
logger = logging.getLogger(__name__)


def set_handler(webhook_handler):
    global handler
    handler = webhook_handler


@main_bp.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    if not signature:
        logger.error("缺少 X-Line-Signature 標頭。")
        abort(400)
    try:
        if handler:
            handler.handle(body, signature)
        else:
            logger.error("Webhook handler not initialized.")
            abort(500)
    except InvalidSignatureError:
        logger.error("無效的簽名")
        abort(400)
    return "OK"


@main_bp.route("/")
def index():
    return render_template("index.html")
