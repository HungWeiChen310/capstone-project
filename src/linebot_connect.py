import json
import os
import logging
from flask import Flask, request, abort, render_template
from linebot.v3.messaging import MessagingApi
from linebot.v3 import WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from src.powerbi_integration import get_powerbi_embed_config
from src.main import reply_message

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 從環境變數取得 LINE 金鑰
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
channel_secret = os.getenv("LINE_CHANNEL_SECRET")

if not channel_access_token or not channel_secret:
    raise ValueError("LINE 金鑰未正確設置。請確定環境變數 LINE_CHANNEL_ACCESS_TOKEN、LINE_CHANNEL_SECRET 已設定。")

app = Flask(__name__)

line_bot_api = MessagingApi(channel_access_token)
handler = WebhookHandler(channel_secret)

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    if not signature:
        logger.error("缺少 X-Line-Signature 標頭。")
        abort(400)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError as e:
        logger.error(f"驗證失敗：{e}")
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    text = event.message.text.strip().lower()
    # 當使用者輸入 "powerbi" 或 "報表" 時，回覆 PowerBI 報表連結
    if text in ["powerbi", "報表", "powerbi報表"]:
        try:
            config = get_powerbi_embed_config()
            embed_url = config["embedUrl"]
            reply_text = f"請點選下方連結查看 PowerBI 報表：{embed_url}"
        except Exception as e:
            logger.error(f"取得 PowerBI 資訊失敗：{e}")
            reply_text = f"取得 PowerBI 報表資訊失敗：{str(e)}"
        reply = TextSendMessage(text=reply_text)
    else:
        # 其他情況仍由 ChatGPT 處理
        response_text = reply_message(event)
        reply = TextSendMessage(text=response_text)
    line_bot_api.reply_message(event.reply_token, reply)

@app.route("/powerbi")
def powerbi():
    try:
        config = get_powerbi_embed_config()
    except Exception as e:
        logger.error(f"PowerBI 整合錯誤：{e}")
        return f"Error: {str(e)}", 500
    return render_template("powerbi.html", config=config)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
