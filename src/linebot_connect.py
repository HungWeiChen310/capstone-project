import os
import logging
import time
import datetime
import functools
from flask import Flask, request, abort, render_template, session, redirect, url_for, flash
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, Source
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
    TemplateMessage,
    ButtonsTemplate,
    CarouselTemplate,
    CarouselColumn,
    QuickReply,
    QuickReplyItem,
    MessageAction,
    URIAction
)
from src.powerbi_integration import get_powerbi_embed_config
from src.database import db
from flask_talisman import Talisman
from werkzeug.middleware.proxy_fix import ProxyFix
from collections import defaultdict

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 從環境變數取得 LINE 金鑰
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
channel_secret = os.getenv("LINE_CHANNEL_SECRET")

if not channel_access_token or not channel_secret:
    raise ValueError("LINE 金鑰未正確設置。請確定環境變數 LINE_CHANNEL_ACCESS_TOKEN、LINE_CHANNEL_SECRET 已設定。")

# 判斷是否在測試環境 - Moved earlier to ensure it's set before app initialization
is_testing = os.environ.get('TESTING', 'False').lower() == 'true'

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())  # 為 session 管理設定密鑰

csp = {
    'default-src': "'self'",
    'script-src': [
        "'self'",
        'https://cdn.powerbi.com',
        "'unsafe-inline'",  # Only needed for inline PowerBI embed script
    ],
    'style-src': [
        "'self'",
        "'unsafe-inline'",  # Only needed for inline styles
    ],
    'img-src': "'self'",
    'frame-src': [
        'https://app.powerbi.com',
        'https://cdn.powerbi.com',
    ],
    'connect-src': [
        "'self'",
        'https://api.powerbi.com',
        'https://login.microsoftonline.com',
    ]
}

# Only apply Talisman in non-testing environments to avoid redirects during tests
if not is_testing:
    Talisman(app, 
        content_security_policy=csp,
        content_security_policy_nonce_in=['script-src'],
        force_https=True,
        session_cookie_secure=True,
        session_cookie_http_only=True,
        feature_policy="geolocation 'none'; microphone 'none'; camera 'none'"
    )
else:
    logger.info("Running in test mode - Talisman security features disabled")

# Handle proxy headers (if behind a proxy)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# Setup with the appropriate API client configuration
configuration = Configuration(access_token=channel_access_token)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(channel_secret)

request_counts = defaultdict(list)

def rate_limit_check(ip, max_requests=30, window_seconds=60):
    """
    簡單的 IP 請求限制，防止暴力攻擊
    """
    current_time = time.time()
    
    # 清理舊的請求記錄
    request_counts[ip] = [timestamp for timestamp in request_counts[ip] 
                         if current_time - timestamp < window_seconds]
    
    # 檢查請求數量
    if len(request_counts[ip]) >= max_requests:
        return False
    
    # 記錄新請求
    request_counts[ip].append(current_time)
    return True

# 簡單的管理員認證設定
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")

def admin_required(f):
    """簡單的管理員認證裝飾器"""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

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

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    text = event.message.text.strip()
    text_lower = text.lower()
    
    # 當使用者輸入 "powerbi" 或 "報表" 時，回覆 PowerBI 報表連結
    if text_lower in ["powerbi", "報表", "powerbi報表", "report"]:
        try:
            config = get_powerbi_embed_config()
            embed_url = config["embedUrl"]
            
            # 創建一個按鈕模板，附帶 PowerBI 報表連結
            buttons_template = ButtonsTemplate(
                title="PowerBI 報表",
                text="點擊下方按鈕查看我們的數據報表",
                actions=[
                    URIAction(
                        label="查看報表",
                        uri=embed_url
                    )
                ]
            )
            
            template_message = TemplateMessage(
                alt_text="PowerBI 報表連結",
                template=buttons_template
            )
            
            # 創建回覆請求
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[template_message]
            )
            
            line_bot_api.reply_message_with_http_info(reply_request)
            
        except Exception as e:
            logger.error(f"取得 PowerBI 資訊失敗：{e}")
            
            # 若失敗則使用文字訊息回覆
            message = TextMessage(text=f"取得 PowerBI 報表資訊失敗，請稍後再試。")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)
    
    # 幫助命令
    elif text_lower in ["help", "幫助", "選單", "menu"]:
        # 創建快速回覆按鈕
        quick_reply = QuickReply(items=[
            QuickReplyItem(
                action=MessageAction(label="查看報表", text="powerbi")
            ),
            QuickReplyItem(
                action=MessageAction(label="使用說明", text="使用說明")
            ),
            QuickReplyItem(
                action=MessageAction(label="關於", text="關於")
            )
        ])
        
        message = TextMessage(
            text="您可以選擇以下選項或直接輸入您的問題：",
            quick_reply=quick_reply
        )
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)
    
    # 使用說明
    elif text_lower in ["使用說明", "說明", "教學", "指南", "guide"]:
        carousel_template = CarouselTemplate(
            columns=[
                CarouselColumn(
                    title="如何使用聊天機器人",
                    text="直接輸入您的問題，AI 將為您提供解答。",
                    actions=[
                        MessageAction(
                            label="試試問問題",
                            text="如何建立一個簡單的網頁？"
                        )
                    ]
                ),
                CarouselColumn(
                    title="查看 PowerBI 報表",
                    text="輸入 'powerbi' 查看數據報表。",
                    actions=[
                        MessageAction(
                            label="查看報表",
                            text="powerbi"
                        )
                    ]
                ),
                CarouselColumn(
                    title="語言設定",
                    text="輸入 'language:語言代碼' 更改語言。",
                    actions=[
                        MessageAction(
                            label="查看語言選項",
                            text="language"
                        )
                    ]
                )
            ]
        )
        
        template_message = TemplateMessage(
            alt_text="使用說明",
            template=carousel_template
        )
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[template_message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)
    
    # 關於命令
    elif text_lower in ["關於", "about"]:
        message = TextMessage(
            text="這是一個整合 LINE Bot、OpenAI 與 PowerBI 的智能助理，可以回答您的技術問題並展示 PowerBI 報表。您可以輸入 'help' 查看更多功能。"
        )
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)
    
    # 語言選項
    elif text_lower == "language":
        message = TextMessage(
            text="您可以通過輸入以下命令設置語言：\n\n"
                 "language:zh-Hant - 繁體中文\n"
                 "language:zh-Hans - 简体中文\n"
                 "language:en - English\n"
                 "language:ja - 日本語\n"
                 "language:ko - 한국어"
        )
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)
    
    # 語言設定
    elif text_lower.startswith("language:") or text.startswith("語言:"):
        # 提取語言代碼
        lang_code = text.split(":", 1)[1].strip().lower()
        
        # 驗證語言代碼
        valid_langs = {
            "zh": "zh-Hant",
            "zh-hant": "zh-Hant",
            "zh-hans": "zh-Hans",
            "en": "en",
            "ja": "ja",
            "ko": "ko"
        }
        
        if lang_code in valid_langs:
            # 保存使用者偏好
            lang = valid_langs[lang_code]
            db.set_user_preference(event.source.user_id, language=lang)
            
            # 確認語言變更
            lang_names = {
                "zh-Hant": "繁體中文",
                "zh-Hans": "简体中文",
                "en": "English",
                "ja": "日本語",
                "ko": "한국어"
            }
            
            message = TextMessage(
                text=f"語言已設置為 {lang_names[lang]}"
            )
        else:
            message = TextMessage(
                text="不支援的語言。支援的語言有：繁體中文 (zh-Hant)、简体中文 (zh-Hans)、English (en)、日本語 (ja)、한국어 (ko)"
            )
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)
    
    # 預設：從 ChatGPT 取得回應
    else:
        # 從 ChatGPT 取得回應
        from src.main import reply_message
        response_text = reply_message(event)
        
        # 創建訊息
        message = TextMessage(text=response_text)
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)

@app.route("/powerbi")
def powerbi():
    # 基本請求限制
    if not rate_limit_check(request.remote_addr):
        return "請求太多，請稍後再試。", 429
        
    try:
        config = get_powerbi_embed_config()
    except Exception as e:
        logger.error(f"PowerBI 整合錯誤: {e}")
        return "系統錯誤，請稍後再試。", 500
    return render_template("powerbi.html", config=config)

@app.route("/")
def index():
    """首頁，顯示簡單的服務狀態"""
    return render_template("index.html")

# 管理後台路由
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(request.args.get("next") or url_for("admin_dashboard"))
        else:
            flash("登入失敗，請確認帳號密碼是否正確", "error")
    
    return render_template("admin_login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    # 取得總對話數
    conversation_stats = db.get_conversation_stats()
    
    # 取得近期使用者與對話
    recent_conversations = db.get_recent_conversations(limit=20)
    
    # 取得系統資訊
    system_info = {
        "openai_api_key": "已設置" if os.getenv("OPENAI_API_KEY") else "未設置",
        "line_channel_secret": "已設置" if os.getenv("LINE_CHANNEL_SECRET") else "未設置", 
        "powerbi_config": "已設置" if all([os.getenv(f"POWERBI_{key}") for key in ["CLIENT_ID", "CLIENT_SECRET", "TENANT_ID", "WORKSPACE_ID", "REPORT_ID"]]) else "未設置"
    }
    
    return render_template(
        "admin_dashboard.html",
        stats=conversation_stats,
        recent=recent_conversations,
        system_info=system_info
    )

@app.route("/admin/conversation/<user_id>")
@admin_required
def admin_view_conversation(user_id):
    # 取得該使用者的對話記錄
    conversation = db.get_conversation_history(user_id, limit=50)
    
    # 取得使用者資訊
    user_info = db.get_user_preference(user_id)
    
    return render_template(
        "admin_conversation.html",
        conversation=conversation,
        user_id=user_id,
        user_info=user_info
    )

# Jinja過濾器與功能函數
@app.template_filter('nl2br')
def nl2br(value):
    if not value:
        return ""
    return value.replace('\n', '<br>')

@app.context_processor
def utility_processor():
    def now():
        return datetime.datetime.now()
    return dict(now=now)

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug_mode)