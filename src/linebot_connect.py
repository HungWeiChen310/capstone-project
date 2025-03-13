import os
import logging
import time
import datetime
import functools
import sqlite3
import threading
import secrets
import urllib.parse
from collections import defaultdict
from flask import Flask, request, abort, render_template, session, redirect, url_for, flash
from linebot.v3.webhook import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, Source
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
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
from src.equipment_scheduler import start_scheduler
from src.initial_data import initialize_equipment_data

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

# 固定的密鑰文件路徑
SECRET_KEY_FILE = "data/secret_key.txt"

def get_or_create_secret_key():
    """獲取或創建一個固定的 secret key"""
    # 首先檢查環境變數
    env_key = os.getenv('SECRET_KEY')
    if env_key:
        return env_key
        
    # 然後檢查文件
    os.makedirs(os.path.dirname(SECRET_KEY_FILE), exist_ok=True)
    try:
        if os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, 'r') as f:
                key = f.read().strip()
                if key:
                    return key
                    
        # 如果文件不存在或為空，生成新密鑰
        key = secrets.token_hex(24)
        with open(SECRET_KEY_FILE, 'w') as f:
            f.write(key)
        return key
    except Exception as e:
        logger.warning(f"無法讀取或寫入密鑰文件: {e}，使用臨時密鑰")
        return secrets.token_hex(24)

# 要放入全局作用域以在整個應用程序中使用
request_counts = defaultdict(list)
last_cleanup_time = time.time()
request_counts_lock = threading.Lock()  # 添加鎖以確保線程安全

def cleanup_request_counts():
    """清理長時間未使用的 IP 地址"""
    global last_cleanup_time
    current_time = time.time()
    
    # 每小時執行一次清理
    if current_time - last_cleanup_time < 3600:
        return
        
    with request_counts_lock:
        # 找出需要刪除的 IP
        ips_to_remove = []
        for ip, timestamps in request_counts.items():
            # 如果 IP 最近一小時沒有請求，則移除
            if not timestamps or current_time - max(timestamps) > 3600:
                ips_to_remove.append(ip)
                
        # 刪除過期的 IP
        for ip in ips_to_remove:
            del request_counts[ip]
            
        last_cleanup_time = current_time
        logger.info(f"已清理 {len(ips_to_remove)} 個過期 IP 地址")

def rate_limit_check(ip, max_requests=30, window_seconds=60):
    """
    簡單的 IP 請求限制，防止暴力攻擊
    """
    current_time = time.time()
    
    # 先清理過期的 IP 記錄
    cleanup_request_counts()
    
    with request_counts_lock:
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

def create_app():
    """創建 Flask 應用程序"""
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
    app.secret_key = get_or_create_secret_key()

    # Handle proxy headers (if behind a proxy)
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

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
        
    return app

app = create_app()

# Setup with the appropriate API client configuration
configuration = Configuration(access_token=channel_access_token)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(channel_secret)

def register_routes(app):
    """註冊所有路由"""
    
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

    @app.route("/powerbi")
    def powerbi():
        # 基本請求限制
        if not rate_limit_check(request.remote_addr):
            return "請求太多，請稍後再試。", 429
            
        try:
            # 如果有用戶ID參數，使用該用戶的訂閱過濾報表
            user_id = request.args.get('user_id')
            config = get_powerbi_embed_config(user_id)
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

# 註冊路由
register_routes(app)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    text = event.message.text.strip()
    text_lower = text.lower()
    
    # 當使用者輸入 "powerbi" 或 "報表" 時，回覆 PowerBI 報表連結
    if text_lower in ["powerbi", "報表", "powerbi報表", "report"]:
        try:
            # 傳遞用戶 ID 以獲取過濾後的報表配置
            user_id = event.source.user_id
            config = get_powerbi_embed_config(user_id)
            embed_url = config["embedUrl"]
            
            # 添加過濾器參數（如果有）
            equipment_filter = config.get("equipmentFilter")
            if equipment_filter and len(equipment_filter) > 0:
                # 將設備清單轉換為 PowerBI URL 過濾參數格式
                quoted_items = [f"'{eq}'" for eq in equipment_filter]
                equipment_list = f"[{','.join(quoted_items)}]"
                filter_param = f"$filter=Equipment/EquipmentID in {equipment_list}"
                # 編碼過濾參數
                encoded_filter = urllib.parse.quote(filter_param)
                # 添加到 URL
                embed_url = f"{embed_url}&{encoded_filter}"
                
                # 還需要添加用戶ID參數，以便在網頁中顯示用戶訂閱設備
                embed_url = f"{embed_url}&user_id={user_id}"
            
            # 創建一個按鈕模板，附帶 PowerBI 報表連結
            buttons_template = ButtonsTemplate(
                title="PowerBI 報表",
                text="點擊下方按鈕查看您訂閱的設備報表",
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
                action=MessageAction(label="我的訂閱", text="我的訂閱")
            ),
            QuickReplyItem(
                action=MessageAction(label="訂閱設備", text="訂閱設備")
            ),
            QuickReplyItem(
                action=MessageAction(label="設備狀態", text="設備狀態")
            ),
            QuickReplyItem(
                action=MessageAction(label="使用說明", text="使用說明")
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
                    title="設備訂閱功能",
                    text="訂閱您需要監控的設備，接收警報並查看報表。",
                    actions=[
                        MessageAction(
                            label="我的訂閱",
                            text="我的訂閱"
                        )
                    ]
                ),
                CarouselColumn(
                    title="查看 PowerBI 報表",
                    text="輸入 'powerbi' 查看已訂閱設備的數據報表。",
                    actions=[
                        MessageAction(
                            label="查看報表",
                            text="powerbi"
                        )
                    ]
                ),
                CarouselColumn(
                    title="設備監控功能",
                    text="查看半導體設備的狀態和異常警告。",
                    actions=[
                        MessageAction(
                            label="查看設備狀態",
                            text="設備狀態"
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
            text="這是一個整合 LINE Bot、OpenAI 與 PowerBI 的智能助理，可以回答您的技術問題、監控半導體設備狀態並展示 PowerBI 報表。您可以輸入 'help' 查看更多功能。"
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
    
    # 設備狀態查詢指令
    elif text_lower in ["設備狀態", "機台狀態", "equipment status"]:
        try:
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT e.type, COUNT(*) as total, 
                           SUM(CASE WHEN e.status = 'normal' THEN 1 ELSE 0 END) as normal,
                           SUM(CASE WHEN e.status = 'warning' THEN 1 ELSE 0 END) as warning,
                           SUM(CASE WHEN e.status = 'critical' THEN 1 ELSE 0 END) as critical,
                           SUM(CASE WHEN e.status = 'emergency' THEN 1 ELSE 0 END) as emergency,
                           SUM(CASE WHEN e.status = 'offline' THEN 1 ELSE 0 END) as offline
                    FROM equipment e
                    GROUP BY e.type
                """)
                
                stats = cursor.fetchall()
                
                if not stats:
                    message = TextMessage(text="目前尚未設定任何設備。")
                else:
                    response_text = "📊 設備狀態摘要：\n\n"
                    
                    for equipment_type, total, normal, warning, critical, emergency, offline in stats:
                        type_name = {"die_bonder": "黏晶機", "wire_bonder": "打線機", "dicer": "切割機"}.get(equipment_type, equipment_type)
                        response_text += f"【{type_name}】共 {total} 台\n"
                        response_text += f"• 正常: {normal} 台\n"
                        
                        if warning > 0:
                            response_text += f"• ⚠️ 警告: {warning} 台\n"
                        
                        if critical > 0:
                            response_text += f"• 🔴 嚴重: {critical} 台\n"
                        
                        if emergency > 0:
                            response_text += f"• 🚨 緊急: {emergency} 台\n"
                        
                        if offline > 0:
                            response_text += f"• ⚫ 離線: {offline} 台\n"
                        
                        response_text += "\n"
                    
                    # 加入異常設備詳細資訊
                    cursor.execute("""
                        SELECT e.name, e.type, e.status, e.equipment_id
                        FROM equipment e
                        WHERE e.status NOT IN ('normal', 'offline')
                        ORDER BY CASE e.status
                            WHEN 'emergency' THEN 1
                            WHEN 'critical' THEN 2
                            WHEN 'warning' THEN 3
                            ELSE 4
                        END
                        LIMIT 5
                    """)
                    
                    abnormal_equipments = cursor.fetchall()
                    
                    if abnormal_equipments:
                        response_text += "⚠️ 異常設備：\n\n"
                        
                        for name, eq_type, status, eq_id in abnormal_equipments:
                            type_name = {"die_bonder": "黏晶機", "wire_bonder": "打線機", "dicer": "切割機"}.get(eq_type, eq_type)
                            status_emoji = {"warning": "⚠️", "critical": "🔴", "emergency": "🚨"}.get(status, "⚠️")
                            
                            response_text += f"{status_emoji} {type_name} {name}\n"
                            
                            # 加入最新警告資訊
                            cursor.execute("""
                                SELECT alert_type, created_at
                                FROM alert_history
                                WHERE equipment_id = ? AND is_resolved = 0
                                ORDER BY created_at DESC
                                LIMIT 1
                            """, (eq_id,))
                            
                            latest_alert = cursor.fetchone()
                            if latest_alert:
                                alert_type, alert_time = latest_alert
                                alert_desc = alert_type.replace("metric_", "").replace("_", " ")
                                response_text += f"  - {alert_desc} ({alert_time})\n"
                        
                        response_text += "\n輸入「設備詳情 [設備名稱]」可查看更多資訊"
                    
                    message = TextMessage(text=response_text)
            
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[message]
            )
            
            line_bot_api.reply_message_with_http_info(reply_request)
            
        except Exception as e:
            logger.error(f"取得設備狀態失敗：{e}")
            message = TextMessage(text="取得設備狀態失敗，請稍後再試。")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)
    
    # 處理「設備詳情」指令
    elif text_lower.startswith("設備詳情") or text_lower.startswith("機台詳情"):
        equipment_name = text[4:].strip() if text_lower.startswith("設備詳情") else text[4:].strip()
        
        if not equipment_name:
            message = TextMessage(text="請指定設備名稱，例如「設備詳情 黏晶機A1」")
        else:
            try:
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 尋找指定名稱的設備
                    cursor.execute("""
                        SELECT e.equipment_id, e.name, e.type, e.status, e.location, e.last_updated
                        FROM equipment e
                        WHERE e.name LIKE ?
                        LIMIT 1
                    """, (f"%{equipment_name}%",))
                    
                    equipment = cursor.fetchone()
                    
                    if not equipment:
                        message = TextMessage(text=f"找不到名稱含「{equipment_name}」的設備，請確認設備名稱。")
                    else:
                        eq_id, name, eq_type, status, location, last_updated = equipment
                        
                        type_name = {"die_bonder": "黏晶機", "wire_bonder": "打線機", "dicer": "切割機"}.get(eq_type, eq_type)
                        status_emoji = {
                            "normal": "✅",
                            "warning": "⚠️",
                            "critical": "🔴",
                            "emergency": "🚨",
                            "offline": "⚫"
                        }.get(status, "❓")
                        
                        response_text = f"📋 {type_name} {name} 詳細資訊\n\n"
                        response_text += f"狀態：{status_emoji} {status}\n"
                        response_text += f"位置：{location}\n"
                        response_text += f"最後更新：{last_updated}\n\n"
                        
                        # 取得最新的監測指標
                        cursor.execute("""
                            SELECT em.metric_type, em.value, em.unit, em.timestamp
                            FROM equipment_metrics em
                            WHERE em.equipment_id = ?
                            GROUP BY em.metric_type
                            HAVING em.timestamp = MAX(em.timestamp)
                            ORDER BY em.metric_type
                        """, (eq_id,))
                        
                        metrics = cursor.fetchall()
                        
                        if metrics:
                            response_text += "📊 最新監測值：\n"
                            
                            for metric_type, value, unit, timestamp in metrics:
                                unit_str = f" {unit}" if unit else ""
                                response_text += f"• {metric_type}：{value}{unit_str}\n"
                            
                            response_text += "\n"
                        
                        # 取得未解決的警告
                        cursor.execute("""
                            SELECT alert_type, severity, created_at
                            FROM alert_history
                            WHERE equipment_id = ? AND is_resolved = 0
                            ORDER BY created_at DESC
                            LIMIT 3
                        """, (eq_id,))
                        
                        alerts = cursor.fetchall()
                        
                        if alerts:
                            response_text += "⚠️ 未解決的警告：\n"
                            
                            for alert_type, severity, alert_time in alerts:
                                severity_emoji = {
                                    "warning": "⚠️",
                                    "critical": "🔴",
                                    "emergency": "🚨"
                                }.get(severity, "⚠️")
                                
                                alert_desc = alert_type.replace("metric_", "").replace("_", " ")
                                response_text += f"• {severity_emoji} {alert_desc} ({alert_time})\n"
                            
                            response_text += "\n"
                        
                        # 取得目前運行的作業
                        cursor.execute("""
                            SELECT operation_type, start_time, lot_id, product_id
                            FROM equipment_operation_logs
                            WHERE equipment_id = ? AND end_time IS NULL
                            ORDER BY start_time DESC
                            LIMIT 1
                        """, (eq_id,))
                        
                        operation = cursor.fetchone()
                        
                        if operation:
                            op_type, start_time, lot_id, product_id = operation
                            response_text += "🔄 目前運行中的作業：\n"
                            response_text += f"• 類型：{op_type}\n"
                            response_text += f"• 開始時間：{start_time}\n"
                            
                            if lot_id:
                                response_text += f"• 批次號：{lot_id}\n"
                            if product_id:
                                response_text += f"• 產品ID：{product_id}\n"
                        
                        message = TextMessage(text=response_text)
            except Exception as e:
                logger.error(f"取得設備詳情失敗：{e}")
                message = TextMessage(text="取得設備詳情失敗，請稍後再試。")
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)
    
    # 設備訂閱相關指令處理
    elif text_lower.startswith("訂閱設備") or text_lower.startswith("subscribe equipment"):
        # 從命令中提取設備ID
        parts = text.split(" ", 1)
        if len(parts) < 2:
            # 如果沒有提供設備ID，列出可用設備
            try:
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 查詢所有設備
                    cursor.execute("""
                        SELECT equipment_id, name, type, location 
                        FROM equipment
                        ORDER BY type, name
                    """)
                    
                    equipments = cursor.fetchall()
                    
                    if not equipments:
                        message = TextMessage(text="目前沒有可用的設備。")
                    else:
                        # 按類型分組顯示設備
                        equipment_types = {}
                        for equipment_id, name, equipment_type, location in equipments:
                            if equipment_type not in equipment_types:
                                equipment_types[equipment_type] = []
                            equipment_types[equipment_type].append((equipment_id, name, location))
                        
                        response_text = "可訂閱的設備清單：\n\n"
                        
                        for equipment_type, equipment_list in equipment_types.items():
                            type_name = {
                                "die_bonder": "黏晶機",
                                "wire_bonder": "打線機",
                                "dicer": "切割機"
                            }.get(equipment_type, equipment_type)
                            
                            response_text += f"【{type_name}】\n"
                            
                            for equipment_id, name, location in equipment_list:
                                response_text += f"• {name} (ID: {equipment_id})\n"
                                response_text += f"  位置: {location}\n"
                            
                            response_text += "\n"
                        
                        response_text += "使用方式: 訂閱設備 [設備ID]\n"
                        response_text += "例如: 訂閱設備 DB001"
                        
                        message = TextMessage(text=response_text)
            except Exception as e:
                logger.error(f"獲取設備清單失敗: {e}")
                message = TextMessage(text="獲取設備清單失敗，請稍後再試。")
        else:
            # 提供了設備ID，進行訂閱
            equipment_id = parts[1].strip()
            user_id = event.source.user_id
            
            try:
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 檢查設備是否存在
                    cursor.execute("SELECT name FROM equipment WHERE equipment_id = ?", (equipment_id,))
                    equipment = cursor.fetchone()
                    
                    if not equipment:
                        message = TextMessage(text=f"找不到ID為 {equipment_id} 的設備，請檢查ID是否正確。")
                    else:
                        equipment_name = equipment[0]
                        
                        # 檢查是否已訂閱
                        cursor.execute("""
                            SELECT id FROM user_equipment_subscriptions
                            WHERE user_id = ? AND equipment_id = ?
                        """, (user_id, equipment_id))
                        
                        existing = cursor.fetchone()
                        
                        if existing:
                            message = TextMessage(text=f"您已經訂閱了設備 {equipment_name} ({equipment_id})。")
                        else:
                            # 添加訂閱
                            cursor.execute("""
                                INSERT INTO user_equipment_subscriptions
                                (user_id, equipment_id, notification_level)
                                VALUES (?, ?, 'all')
                            """, (user_id, equipment_id))
                            
                            conn.commit()
                            
                            message = TextMessage(text=f"成功訂閱設備 {equipment_name} ({equipment_id})。\n\n您現在可以查看此設備的 PowerBI 報表並接收其警報通知。")
            except Exception as e:
                logger.error(f"訂閱設備失敗: {e}")
                message = TextMessage(text="訂閱設備失敗，請稍後再試。")
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)

    elif text_lower.startswith("取消訂閱") or text_lower.startswith("unsubscribe"):
        # 從命令中提取設備ID
        parts = text.split(" ", 1)
        if len(parts) < 2:
            # 如果沒有提供設備ID，列出用戶已訂閱的設備
            try:
                user_id = event.source.user_id
                
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 查詢用戶已訂閱的設備
                    cursor.execute("""
                        SELECT s.equipment_id, e.name, e.type, e.location
                        FROM user_equipment_subscriptions s
                        JOIN equipment e ON s.equipment_id = e.equipment_id
                        WHERE s.user_id = ?
                        ORDER BY e.type, e.name
                    """, (user_id,))
                    
                    subscriptions = cursor.fetchall()
                    
                    if not subscriptions:
                        message = TextMessage(text="您目前沒有訂閱任何設備。")
                    else:
                        response_text = "您已訂閱的設備：\n\n"
                        
                        for equipment_id, name, equipment_type, location in subscriptions:
                            type_name = {
                                "die_bonder": "黏晶機",
                                "wire_bonder": "打線機",
                                "dicer": "切割機"
                            }.get(equipment_type, equipment_type)
                            
                            response_text += f"• {name} ({type_name})\n"
                            response_text += f"  ID: {equipment_id}\n"
                            response_text += f"  位置: {location}\n\n"
                        
                        response_text += "使用方式: 取消訂閱 [設備ID]\n"
                        response_text += "例如: 取消訂閱 DB001"
                        
                        message = TextMessage(text=response_text)
            except Exception as e:
                logger.error(f"獲取訂閱清單失敗: {e}")
                message = TextMessage(text="獲取訂閱清單失敗，請稍後再試。")
        else:
            # 提供了設備ID，取消訂閱
            equipment_id = parts[1].strip()
            user_id = event.source.user_id
            
            try:
                with sqlite3.connect(db.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 檢查設備是否存在
                    cursor.execute("SELECT name FROM equipment WHERE equipment_id = ?", (equipment_id,))
                    equipment = cursor.fetchone()
                    
                    if not equipment:
                        message = TextMessage(text=f"找不到ID為 {equipment_id} 的設備，請檢查ID是否正確。")
                    else:
                        equipment_name = equipment[0]
                        
                        # 檢查是否已訂閱
                        cursor.execute("""
                            DELETE FROM user_equipment_subscriptions
                            WHERE user_id = ? AND equipment_id = ?
                        """, (user_id, equipment_id))
                        
                        if cursor.rowcount > 0:
                            conn.commit()
                            message = TextMessage(text=f"已取消訂閱設備 {equipment_name} ({equipment_id})。")
                        else:
                            message = TextMessage(text=f"您未訂閱設備 {equipment_name} ({equipment_id})。")
            except Exception as e:
                logger.error(f"取消訂閱設備失敗: {e}")
                message = TextMessage(text="取消訂閱設備失敗，請稍後再試。")
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)

    elif text_lower == "我的訂閱" or text_lower == "my subscriptions":
        # 顯示用戶已訂閱的設備
        try:
            user_id = event.source.user_id
            
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                
                # 查詢用戶已訂閱的設備
                cursor.execute("""
                    SELECT s.equipment_id, e.name, e.type, e.location, e.status
                    FROM user_equipment_subscriptions s
                    JOIN equipment e ON s.equipment_id = e.equipment_id
                    WHERE s.user_id = ?
                    ORDER BY e.type, e.name
                """, (user_id,))
                
                subscriptions = cursor.fetchall()
                
                if not subscriptions:
                    response_text = "您目前沒有訂閱任何設備。\n\n請使用「訂閱設備」指令查看可訂閱的設備列表。"
                else:
                    response_text = "您已訂閱的設備：\n\n"
                    
                    for equipment_id, name, equipment_type, location, status in subscriptions:
                        type_name = {
                            "die_bonder": "黏晶機",
                            "wire_bonder": "打線機",
                            "dicer": "切割機"
                        }.get(equipment_type, equipment_type)
                        
                        status_emoji = {
                            "normal": "✅",
                            "warning": "⚠️",
                            "critical": "🔴",
                            "emergency": "🚨",
                            "offline": "⚫"
                        }.get(status, "❓")
                        
                        response_text += f"{status_emoji} {name} ({type_name})\n"
                        response_text += f"  ID: {equipment_id}\n"
                        response_text += f"  位置: {location}\n\n"
                    
                    response_text += "管理訂閱:\n"
                    response_text += "• 訂閱設備 [設備ID] - 新增訂閱\n"
                    response_text += "• 取消訂閱 [設備ID] - 取消訂閱\n"
                    response_text += "• 輸入「報表」查看訂閱設備的 PowerBI 報表"
                    
                message = TextMessage(text=response_text)
        except Exception as e:
            logger.error(f"獲取訂閱清單失敗: {e}")
            message = TextMessage(text="獲取訂閱清單失敗，請稍後再試。")
        
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[message]
        )
        
        line_bot_api.reply_message_with_http_info(reply_request)
    
    # 預設：從 ChatGPT 取得回應
    else:
        try:
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
        except Exception as e:
            logger.error(f"取得 AI 回應失敗：{e}")
            
            # 若失敗則使用文字訊息回覆
            message = TextMessage(text=f"處理訊息時發生錯誤，請稍後再試。")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)

def send_notification(user_id, message):
    """發送 LINE 訊息給特定使用者"""
    try:
        message_obj = TextMessage(text=message)
        
        # 使用推送訊息 API 而非回覆
        push_request = PushMessageRequest(
            to=user_id,
            messages=[message_obj]
        )
        
        line_bot_api.push_message_with_http_info(push_request)
        return True
    except Exception as e:
        logger.error(f"發送通知失敗: {e}")
        return False

# 若此檔案被直接執行
# This is the updated main section for your linebot_connect.py file
# Keep all your existing code above this point unchanged

# 若此檔案被直接執行
if __name__ == "__main__":
    # 初始化設備資料
    initialize_equipment_data()
    
    # 啟動設備監控排程器
    start_scheduler()
    
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    port = int(os.environ.get("PORT", 5000))
    
# Use Flask's built-in adhoc SSL certificates (requires pyOpenSSL)
    # ssl_context = 'adhoc'
    ssl_cert = os.environ.get("SSL_CERT_PATH")
    ssl_key = os.environ.get("SSL_KEY_PATH")
    if ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
    else:
        print("SSL_CERT_PATH and SSL_KEY_PATH not set. Running without SSL.")
        ssl_context = None
    
    print(f"Starting Flask app with SSL on port {port}")
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=debug_mode,
        ssl_context=ssl_context  # This creates temporary certificates automatically
    )