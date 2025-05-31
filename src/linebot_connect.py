import datetime
import functools
import logging
import os
import secrets
import threading
import time
from collections import defaultdict

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_talisman import Talisman
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    CarouselColumn,
    CarouselTemplate,
    Configuration,
    MessageAction,
    MessagingApi,
    PushMessageRequest,
    QuickReply,
    QuickReplyItem,
    ReplyMessageRequest,
    TemplateMessage,
    TextMessage,
)
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from werkzeug.middleware.proxy_fix import ProxyFix

from database import db
from equipment_scheduler import start_scheduler
from initial_data import initialize_equipment_data

# 設定 logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 從環境變數取得 LINE 金鑰
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
channel_secret = os.getenv("LINE_CHANNEL_SECRET")

if not channel_access_token or not channel_secret:
    raise ValueError(
        "LINE 金鑰未正確設置。請確定環境變數 LINE_CHANNEL_ACCESS_TOKEN、LINE_CHANNEL_SECRET 已設定。"
    )

# 判斷是否在測試環境 - 避免在測試期間啟用 Talisman 重定向
is_testing = os.environ.get("TESTING", "False").lower() == "true"

# 固定的密鑰文件路徑
SECRET_KEY_FILE = "data/secret_key.txt"


def get_or_create_secret_key():
    """獲取或創建一個固定的 secret key"""
    # 先檢查環境變數
    env_key = os.getenv("SECRET_KEY")
    if env_key:
        return env_key

    os.makedirs(os.path.dirname(SECRET_KEY_FILE), exist_ok=True)
    try:
        if os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, "r") as f:
                key = f.read().strip()
                if key:
                    return key
        # 如果文件不存在或為空，生成新密鑰
        key = secrets.token_hex(24)
        with open(SECRET_KEY_FILE, "w") as f:
            f.write(key)
        return key
    except Exception as e:
        logger.warning(f"無法讀取或寫入密鑰文件: {e}，使用臨時密鑰")
        return secrets.token_hex(24)


# 全局請求計數器與鎖 (線程安全)
request_counts = defaultdict(list)
last_cleanup_time = time.time()
request_counts_lock = threading.Lock()


def cleanup_request_counts():
    """清理長時間未使用的 IP 地址"""
    global last_cleanup_time
    current_time = time.time()

    # 每小時執行一次清理
    if current_time - last_cleanup_time < 3600:
        return

    with request_counts_lock:
        ips_to_remove = []
        for ip, timestamps in request_counts.items():
            if not timestamps or current_time - max(timestamps) > 3600:
                ips_to_remove.append(ip)
        for ip in ips_to_remove:
            del request_counts[ip]
        last_cleanup_time = current_time
        logger.info("已清理過期請求記錄")


def rate_limit_check(ip, max_requests=30, window_seconds=60):
    """
    簡單的 IP 請求限制，防止暴力攻擊
    """
    current_time = time.time()
    cleanup_request_counts()
    with request_counts_lock:
        request_counts[ip] = [
            timestamp for timestamp in request_counts[ip]
            if current_time - timestamp < window_seconds
        ]
        if len(request_counts[ip]) >= max_requests:
            return False
        request_counts[ip].append(current_time)
        return True


# 簡單的管理員認證設定
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")


def admin_required(f):
    """簡單的管理員認證裝飾器"""

    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login", next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def create_app():
    """創建 Flask 應用程序"""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"),
    )
    app.secret_key = get_or_create_secret_key()

    # 處理代理標頭
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    csp = {
        "default-src": "'self'",
        "script-src": ["'self'", "'unsafe-inline'"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": "'self'",
        "frame-src": [],
        "connect-src": ["'self'"],
    }

    if not is_testing:
        Talisman(
            app,
            content_security_policy=csp,
            content_security_policy_nonce_in=["script-src"],
            force_https=True,
            session_cookie_secure=True,
            session_cookie_http_only=True,
            feature_policy="geolocation 'none'; microphone 'none'; camera 'none'",
        )
    else:
        logger.info("Running in test mode - Talisman security features disabled")

    return app


app = create_app()

# 設定 API 客戶端
configuration = Configuration(access_token=channel_access_token)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(channel_secret)


def register_routes(app):
    """註冊所有路由"""

    @app.route("/callback", methods=["POST"])
    def callback():
        signature = request.headers.get("X-Line-Signature")
        body = request.get_data(as_text=True)
        if not signature:
            logger.error("缺少 X-Line-Signature 標頭。")
            abort(400)
        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            logger.error("無效的簽名")
            abort(400)
        return "OK"

    @app.route("/")
    def index():
        """首頁，顯示簡單的服務狀態"""
        return render_template("index.html")

    # 管理後台路由
    @app.route("/admin/login", methods=["GET", "POST"])
    def admin_login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "") # No strip for password

            # Input validation
            if not username or len(username) < 3 or len(username) > 50:
                flash("用戶名長度必須在 3 到 50 個字符之間。", "error")
            elif not password or len(password) < 8 or len(password) > 100:
                flash("密碼長度必須在 8 到 100 個字符之間。", "error")
            elif username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                session["admin_logged_in"] = True
                logger.info(f"管理員 {username} 成功登入。")
                return redirect(request.args.get("next") or url_for("admin_dashboard"))
            else:
                logger.warning(f"管理員登入失敗，嘗試的用戶名: {username}")
                flash("登入失敗，請確認帳號密碼是否正確", "error")
        return render_template("admin_login.html")

    @app.route("/admin/logout")
    def admin_logout():
        session.pop("admin_logged_in", None)
        return redirect(url_for("admin_login"))

    @app.route("/admin/dashboard")
    @admin_required
    def admin_dashboard():
        conversation_stats = db.get_conversation_stats()
        recent_conversations = db.get_recent_conversations(limit=20)
        system_info = {
            "openai_api_key": "已設置" if os.getenv("OPENAI_API_KEY") else "未設置",
            "line_channel_secret": "已設置" if os.getenv("LINE_CHANNEL_SECRET") else "未設置",
        }
        return render_template(
            "admin_dashboard.html",
            stats=conversation_stats,
            recent=recent_conversations,
            system_info=system_info,
        )

    @app.route("/admin/conversation/<user_id>")
    @admin_required
    def admin_view_conversation(user_id):
        # Validate user_id format (LINE user IDs usually start with 'U' and are alphanumeric)
        # This is a basic check, a more robust regex might be needed for strict validation.
        if not user_id or not isinstance(user_id, str) or not user_id.isalnum() or len(user_id) < 20 or len(user_id) > 50 :
            logger.error(f"無效的 user_id 格式: {user_id}")
            flash("無效的使用者 ID 格式。", "error")
            return redirect(url_for("admin_dashboard")) # Or show an error page

        try:
            conversation = db.get_conversation_history(user_id, limit=50)
            user_info = db.get_user_preference(user_id) # This also creates a default if not found
            if not user_info: # Should not happen if get_user_preference creates default
                 logger.warning(f"在 admin_view_conversation 中找不到 user_id 的使用者資訊: {user_id}")
                 # Fallback, though get_user_preference should handle this
                 user_info = {"language": "zh-Hant", "role": "user"}

            return render_template(
                "admin_conversation.html",
                conversation=conversation,
                user_id=user_id,
                user_info=user_info,
            )
        except pyodbc.Error as db_err:
            logger.error(f"查看對話記錄時發生資料庫錯誤 (user_id: {user_id}): {db_err}")
            flash("查詢對話記錄時發生資料庫錯誤。", "error")
            return redirect(url_for("admin_dashboard"))
        except Exception as e:
            logger.error(f"查看對話記錄時發生未知錯誤 (user_id: {user_id}): {e}")
            flash("查看對話記錄時發生未知錯誤。", "error")
            return redirect(url_for("admin_dashboard"))


    @app.template_filter("nl2br")
    def nl2br(value):
        if not value:
            return ""
        return value.replace("\n", "<br>")

    @app.context_processor
    def utility_processor():
        def now_func():
            return datetime.datetime.now()
        return dict(now=now_func)


register_routes(app)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    text = event.message.text.strip()
    text_lower = text.lower()

    # 幫助命令
    if text_lower in ["help", "幫助", "選單", "menu"]:
        quick_reply = QuickReply(
            items=[
                QuickReplyItem(action=MessageAction(label="查看報表", text="powerbi")),
                QuickReplyItem(action=MessageAction(label="我的訂閱", text="我的訂閱")),
                QuickReplyItem(action=MessageAction(label="訂閱設備", text="訂閱設備")),
                QuickReplyItem(action=MessageAction(label="設備狀態", text="設備狀態")),
                QuickReplyItem(action=MessageAction(label="使用說明", text="使用說明")),
            ]
        )
        message = TextMessage(
            text="您可以選擇以下選項或直接輸入您的問題：", quick_reply=quick_reply
        )
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
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
                        MessageAction(label="試試問問題", text="如何建立一個簡單的網頁？")
                    ],
                ),
                CarouselColumn(
                    title="設備訂閱功能",
                    text="訂閱您需要監控的設備，接收警報並查看報表。",
                    actions=[MessageAction(label="我的訂閱", text="我的訂閱")],
                ),
                CarouselColumn(
                    title="設備監控功能",
                    text="查看半導體設備的狀態和異常警告。",
                    actions=[MessageAction(label="查看設備狀態", text="設備狀態")],
                ),
                CarouselColumn(
                    title="語言設定",
                    text="輸入 'language:語言代碼' 更改語言。",
                    actions=[MessageAction(label="查看語言選項", text="language")],
                ),
            ]
        )
        template_message = TemplateMessage(
            alt_text="使用說明", template=carousel_template
        )
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[template_message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    # 關於命令
    elif text_lower in ["關於", "about"]:
        message = TextMessage(
            text=(
                "這是一個整合 LINE Bot 與 OpenAI 的智能助理，"
                "可以回答您的技術問題、監控半導體設備狀態並展示。"
                "您可以輸入 'help' 查看更多功能。"
            )
        )
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    # 語言選項
    elif text_lower == "language":
        message = TextMessage(
            text=(
                "您可以通過輸入以下命令設置語言：\n\n"
                "language:zh-Hant - 繁體中文"
            )
        )
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    # 語言設定
    elif text_lower.startswith("language:") or text.startswith("語言:"):
        try:
            parts = text.split(":", 1)
            if len(parts) < 2 or not parts[1].strip():
                message = TextMessage(text="請提供語言代碼，例如 'language:zh-Hant'")
            else:
                lang_code = parts[1].strip().lower()
                # Add length validation for lang_code
                if len(lang_code) > 10:
                     message = TextMessage(text="語言代碼過長。")
                else:
                    valid_langs = {"zh": "zh-Hant", "zh-hant": "zh-Hant"}
                    if lang_code in valid_langs:
                        lang = valid_langs[lang_code]
                        # Ensure db.set_user_preference has error handling
                        if db.set_user_preference(event.source.user_id, language=lang):
                            confirmation_map = {"zh-Hant": "繁體中文"}
                            message = TextMessage(text="語言已切換至 " + confirmation_map.get(lang, lang))
                        else:
                            message = TextMessage(text="語言設定失敗，請稍後再試。")
                    else:
                        message = TextMessage(
                            text="不支援的語言。支援的語言有：繁體中文 (zh-Hant)"
                        )
        except Exception as e:
            logger.error(f"處理語言設定時發生錯誤: {e}")
            message = TextMessage(text="處理語言設定時發生內部錯誤。")
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    # 設備狀態查詢指令
    elif text_lower in ["設備狀態", "機台狀態", "equipment status"]:
        try:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                # SQL Server uses TOP N instead of LIMIT N
                # Also, ensure correct syntax for conditional aggregation
                cursor.execute(
                    """
                    SELECT e.type, COUNT(*) as total,
                           SUM(CASE WHEN e.status = 'normal' THEN 1 ELSE 0 END) as normal,
                           SUM(CASE WHEN e.status = 'warning' THEN 1 ELSE 0 END) as warning,
                           SUM(CASE WHEN e.status = 'critical' THEN 1 ELSE 0 END) as critical,
                           SUM(CASE WHEN e.status = 'emergency' THEN 1 ELSE 0 END) as emergency,
                           SUM(CASE WHEN e.status = 'offline' THEN 1 ELSE 0 END) as offline
                    FROM equipment e
                    GROUP BY e.type
                    """
                )
                stats = cursor.fetchall()
                if not stats:
                    message = TextMessage(text="目前尚未設定任何設備。")
                else:
                    response_text = "📊 設備狀態摘要：\n\n"
                    for row in stats:
                        equipment_type, total, normal, warning, critical, emergency, offline = row
                        type_name = {
                            "die_bonder": "黏晶機",
                            "wire_bonder": "打線機",
                            "dicer": "切割機",
                        }.get(equipment_type, equipment_type)
                        response_text += (
                            f"{type_name}：總數 {total or 0}, 正常 {normal or 0}"
                        )
                        if warning and warning > 0:
                            response_text += f", 警告 {warning}"
                        if critical and critical > 0:
                            response_text += f", 嚴重 {critical}"
                        if emergency and emergency > 0:
                            response_text += f", 緊急 {emergency}"
                        if offline and offline > 0:
                            response_text += f", 離線 {offline}"
                        response_text += "\n"

                    # 加入異常設備詳細資訊
                    # SQL Server uses TOP N
                    cursor.execute(
                        """
                        SELECT TOP 5 e.name, e.type, e.status, e.equipment_id
                        FROM equipment e
                        WHERE e.status NOT IN ('normal', 'offline')
                        ORDER BY CASE e.status
                            WHEN 'emergency' THEN 1
                            WHEN 'critical' THEN 2
                            WHEN 'warning' THEN 3
                            ELSE 4
                        END
                        """
                    )
                    abnormal_equipments = cursor.fetchall()
                    if abnormal_equipments:
                        response_text += "\n⚠️ 異常設備：\n\n"
                        for ab_row in abnormal_equipments:
                            name, eq_type, status, eq_id = ab_row
                            type_name = {
                                "die_bonder": "黏晶機",
                                "wire_bonder": "打線機",
                                "dicer": "切割機",
                            }.get(eq_type, eq_type)
                            status_emoji = {
                                "warning": "⚠️",
                                "critical": "🔴",
                                "emergency": "🚨",
                            }.get(status, "⚠️")
                            response_text += (
                                f"{name} ({type_name}) 狀態: {status_emoji} "
                            )
                            # SQL Server uses TOP N
                            cursor.execute(
                                """
                                SELECT TOP 1 alert_type, created_at
                                FROM alert_history
                                WHERE equipment_id = ? AND is_resolved = 0
                                ORDER BY created_at DESC
                                """,
                                (eq_id,),
                            )
                            latest_alert = cursor.fetchone()
                            if latest_alert:
                                alert_type, alert_time = latest_alert
                                response_text += (
                                    f"最新警告: {alert_type} 於 {alert_time}\n"
                                )
                            else:
                                response_text += "\n"
                        response_text += "\n輸入「設備詳情 [設備名稱]」可查看更多資訊"
                    message = TextMessage(text=response_text)
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token, messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)
        except pyodbc.Error as db_err:
            logger.error(f"取得設備狀態時發生資料庫錯誤: {db_err}")
            message = TextMessage(text="資料庫查詢失敗，無法取得設備狀態。")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token, messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)
        except Exception as e:
            logger.error(f"取得設備狀態失敗: {e}")
            message = TextMessage(text="取得設備狀態失敗，請稍後再試。")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token, messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)

    # 處理「設備詳情」指令
    elif text_lower.startswith("設備詳情") or text_lower.startswith("機台詳情"):
        equipment_name_input = text[4:].strip()
        if not equipment_name_input: # Basic validation
            message = TextMessage(text="請指定設備名稱，例如「設備詳情 黏晶機A1」")
        elif len(equipment_name_input) > 100: # Length validation
            message = TextMessage(text="設備名稱過長，請輸入少於100個字元。")
        else:
            try:
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    # SQL Server uses TOP 1 and parameterization for LIKE
                    cursor.execute(
                        """
                        SELECT TOP 1 e.equipment_id, e.name, e.type, e.status, e.location, e.last_updated
                        FROM equipment e
                        WHERE e.name LIKE ?
                        """,
                        (f"%{equipment_name_input}%",),
                    )
                    equipment = cursor.fetchone()
                    if not equipment:
                        message = TextMessage(text="查無設備資料。")
                    else:
                        eq_id, name, eq_type, status, location, last_updated = equipment
                        type_name = {
                            "die_bonder": "黏晶機",
                            "wire_bonder": "打線機",
                            "dicer": "切割機",
                        }.get(eq_type, eq_type)
                        status_emoji = {
                            "normal": "✅",
                            "warning": "⚠️",
                            "critical": "🔴",
                            "emergency": "🚨",
                            "offline": "⚫",
                        }.get(status, "❓")
                        response_text = (
                            f"設備詳情：\n名稱: {name}\n類型: {type_name}\n"
                            f"狀態: {status_emoji}\n地點: {location or '未提供'}\n" # Handle None location
                            f"最後更新: {last_updated or '未知'}\n\n" # Handle None last_updated
                        )
                        # SQL Server requires a different approach for getting latest metric per type
                        # This might be complex for a single query, consider stored procedure or multiple queries
                        # For now, let's get all recent metrics for simplicity, this could be refined
                        cursor.execute(
                            """
                            SELECT em.metric_type, em.value, em.unit, em.timestamp
                            FROM equipment_metrics em
                            WHERE em.equipment_id = ?
                            ORDER BY em.metric_type, em.timestamp DESC
                            """,
                            (eq_id,),
                        )
                        metrics_raw = cursor.fetchall()
                        # Process metrics to get the latest for each type
                        latest_metrics = {}
                        for metric_type, value, unit, timestamp in metrics_raw:
                            if metric_type not in latest_metrics:
                                latest_metrics[metric_type] = (value, unit, timestamp)

                        if latest_metrics:
                            response_text += "📊 最新監測值：\n"
                            for metric_type, (value, unit, timestamp) in latest_metrics.items():
                                response_text += (
                                    f"{metric_type}: {value} {unit or ''} ({timestamp})\n" # Handle None unit
                                )

                        # SQL Server uses TOP N
                        cursor.execute(
                            """
                            SELECT TOP 3 alert_type, severity, created_at
                            FROM alert_history
                            WHERE equipment_id = ? AND is_resolved = 0
                            ORDER BY created_at DESC
                            """,
                            (eq_id,),
                        )
                        alerts = cursor.fetchall()
                        if alerts:
                            response_text += "\n⚠️ 未解決的警告：\n"
                            for alert_row in alerts: # Iterate over rows
                                alert_type, severity, alert_time = alert_row
                                status_map = {
                                    "warning": "⚠️",
                                    "critical": "🔴",
                                    "emergency": "🚨",
                                }
                                emoji = status_map.get(severity, "⚠️")
                                response_text += (
                                    f"{emoji} {alert_type} 於 {alert_time}\n"
                                )

                        # SQL Server uses TOP N
                        cursor.execute(
                            """
                            SELECT TOP 1 operation_type, start_time, lot_id, product_id
                            FROM equipment_operation_logs
                            WHERE equipment_id = ? AND end_time IS NULL
                            ORDER BY start_time DESC
                            """,
                            (eq_id,),
                        )
                        operation = cursor.fetchone()
                        if operation:
                            op_type, start_time, lot_id, product_id = operation
                            response_text += "\n🔄 目前運行中的作業：\n"
                            response_text += f"作業類型: {op_type}\n開始時間: {start_time}\n"
                            if lot_id:
                                response_text += f"批次: {lot_id}\n"
                            if product_id:
                                response_text += f"產品: {product_id}\n"
                        message = TextMessage(text=response_text)
            except pyodbc.Error as db_err:
                logger.error(f"取得設備詳情時發生資料庫錯誤: {db_err}")
                message = TextMessage(text="資料庫查詢失敗，無法取得設備詳情。")
            except Exception as e:
                logger.error(f"取得設備詳情失敗: {e}")
                message = TextMessage(text="取得設備詳情失敗，請稍後再試。")
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    # 設備訂閱相關指令處理
    elif text_lower.startswith("訂閱設備") or text_lower.startswith("subscribe equipment"):
        parts = text.split(" ", 1) # Max split 1
        if len(parts) < 2 or not parts[1].strip(): # Check if equipment_id is provided
            try:
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT equipment_id, name, type, location
                        FROM equipment
                        ORDER BY type, name
                        """
                    )
                    equipments = cursor.fetchall()
                    if not equipments:
                        message = TextMessage(text="目前沒有可用的設備。")
                    else:
                        equipment_types = {}
                        for eq_row in equipments: # Iterate over rows
                            equipment_id_db, name, equipment_type, location = eq_row
                            equipment_types.setdefault(equipment_type, []).append(
                                (equipment_id_db, name, location)
                            )
                        response_text = "可訂閱的設備清單：\n\n"
                        for equipment_type, equipment_list in equipment_types.items():
                            type_name = {
                                "die_bonder": "黏晶機",
                                "wire_bonder": "打線機",
                                "dicer": "切割機",
                            }.get(equipment_type, equipment_type)
                            response_text += f"{type_name}：\n"
                            for equipment_id_db, name, location in equipment_list:
                                response_text += f"  {equipment_id_db} - {name} ({location or '未提供'})\n" # Handle None location
                            response_text += "\n"
                        response_text += "使用方式: 訂閱設備 [設備ID]\n例如: 訂閱設備 DB001"
                        message = TextMessage(text=response_text)
            except pyodbc.Error as db_err:
                logger.error(f"獲取可訂閱設備清單時發生資料庫錯誤: {db_err}")
                message = TextMessage(text="資料庫查詢失敗，無法獲取設備清單。")
            except Exception as e:
                logger.error(f"獲取設備清單失敗: {e}")
                message = TextMessage(text="獲取設備清單失敗，請稍後再試。")
        else:
            equipment_id_input = parts[1].strip()
            user_id = event.source.user_id

            # Input validation for equipment_id_input
            if not equipment_id_input or len(equipment_id_input) > 50:
                 message = TextMessage(text="請提供有效且長度不超過50字元的設備ID。")
            else:
                try:
                    with db._get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT name FROM equipment WHERE equipment_id = ?",
                            (equipment_id_input,),
                        )
                        equipment = cursor.fetchone()
                        if not equipment:
                            message = TextMessage(text="查無此設備ID，請確認後再試。")
                        else:
                            cursor.execute(
                                """
                                SELECT id FROM user_equipment_subscriptions
                                WHERE user_id = ? AND equipment_id = ?
                                """,
                                (user_id, equipment_id_input),
                            )
                            existing = cursor.fetchone()
                            if existing:
                                message = TextMessage(text="您已訂閱該設備。")
                            else:
                                cursor.execute(
                                    """
                                    INSERT INTO user_equipment_subscriptions
                                    (user_id, equipment_id, notification_level)
                                    VALUES (?, ?, 'all')
                                    """,
                                    (user_id, equipment_id_input),
                                )
                                conn.commit()
                                message = TextMessage(text="訂閱成功！")
                except pyodbc.Error as db_err:
                    logger.error(f"訂閱設備時發生資料庫錯誤: {db_err}")
                    # Check for unique constraint violation (though pyodbc might raise a generic DataError or IntegrityError)
                    if "UNIQUE constraint failed" in str(db_err) or "Violation of UNIQUE KEY constraint" in str(db_err): # Adapt based on actual SQL Server error
                        message = TextMessage(text="您似乎已訂閱此設備。")
                    else:
                        message = TextMessage(text="訂閱設備時資料庫操作失敗，請稍後再試。")
                except Exception as e:
                    logger.error(f"訂閱設備失敗: {e}")
                    message = TextMessage(text="訂閱設備失敗，請稍後再試。")
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    elif text_lower.startswith("取消訂閱") or text_lower.startswith("unsubscribe"):
        parts = text.split(" ", 1) # Max split 1
        if len(parts) < 2 or not parts[1].strip(): # Check if equipment_id is provided
            try:
                user_id = event.source.user_id
                with db._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT s.equipment_id, e.name, e.type, e.location
                        FROM user_equipment_subscriptions s
                        JOIN equipment e ON s.equipment_id = e.equipment_id
                        WHERE s.user_id = ?
                        ORDER BY e.type, e.name
                        """,
                        (user_id,),
                    )
                    subscriptions = cursor.fetchall()
                    if not subscriptions:
                        message = TextMessage(text="您目前沒有訂閱任何設備。")
                    else:
                        response_text = "您已訂閱的設備：\n\n"
                        for sub_row in subscriptions: # Iterate over rows
                            equipment_id_db, name, equipment_type, location = sub_row
                            type_name = {
                                "die_bonder": "黏晶機",
                                "wire_bonder": "打線機",
                                "dicer": "切割機",
                            }.get(equipment_type, equipment_type)
                            response_text += f"{equipment_id_db} - {name} ({type_name}, {location or '未提供'})\n" # Handle None location
                        response_text += "\n使用方式: 取消訂閱 [設備ID]\n例如: 取消訂閱 DB001"
                        message = TextMessage(text=response_text)
            except pyodbc.Error as db_err:
                logger.error(f"獲取已訂閱設備清單時發生資料庫錯誤: {db_err}")
                message = TextMessage(text="資料庫查詢失敗，無法獲取您的訂閱清單。")
            except Exception as e:
                logger.error(f"獲取訂閱清單失敗: {e}")
                message = TextMessage(text="獲取訂閱清單失敗，請稍後再試。")
        else:
            equipment_id_input = parts[1].strip()
            user_id = event.source.user_id

            if not equipment_id_input or len(equipment_id_input) > 50: # Input validation
                message = TextMessage(text="請提供有效且長度不超過50字元的設備ID。")
            else:
                try:
                    with db._get_connection() as conn:
                        cursor = conn.cursor()
                        # First check if the equipment exists to give a more specific message
                        cursor.execute(
                            "SELECT name FROM equipment WHERE equipment_id = ?",
                            (equipment_id_input,),
                        )
                        equipment = cursor.fetchone()
                        if not equipment:
                            message = TextMessage(text="查無此設備ID，無法取消訂閱。")
                        else:
                            cursor.execute(
                                """
                                DELETE FROM user_equipment_subscriptions
                                WHERE user_id = ? AND equipment_id = ?
                                """,
                                (user_id, equipment_id_input),
                            )
                            if cursor.rowcount > 0:
                                conn.commit()
                                message = TextMessage(text="取消訂閱成功！")
                            else:
                                message = TextMessage(text="您並未訂閱該設備。")
                except pyodbc.Error as db_err:
                    logger.error(f"取消訂閱時發生資料庫錯誤: {db_err}")
                    message = TextMessage(text="取消訂閱時資料庫操作失敗，請稍後再試。")
                except Exception as e:
                    logger.error(f"取消訂閱失敗: {e}")
                    message = TextMessage(text="取消訂閱設備失敗，請稍後再試。")
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    elif text_lower in ["我的訂閱", "my subscriptions"]:
        try:
            user_id = event.source.user_id
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT s.equipment_id, e.name, e.type, e.location, e.status
                    FROM user_equipment_subscriptions s
                    JOIN equipment e ON s.equipment_id = e.equipment_id
                    WHERE s.user_id = ?
                    ORDER BY e.type, e.name
                    """,
                    (user_id,),
                )
                subscriptions = cursor.fetchall()
                if not subscriptions:
                    response_text = (
                        "您目前沒有訂閱任何設備。\n\n"
                        "請使用「訂閱設備」指令查看可訂閱的設備列表。"
                    )
                else:
                    response_text = "您已訂閱的設備：\n\n"
                    for sub_row in subscriptions: # Iterate over rows
                        equipment_id_db, name, equipment_type, location, status = sub_row
                        type_name = {
                            "die_bonder": "黏晶機",
                            "wire_bonder": "打線機",
                            "dicer": "切割機",
                        }.get(equipment_type, equipment_type)
                        status_emoji = {
                            "normal": "✅",
                            "warning": "⚠️",
                            "critical": "🔴",
                            "emergency": "🚨",
                            "offline": "⚫",
                        }.get(status, "❓")
                        response_text += f"{equipment_id_db} - {name} ({type_name}, {location or '未提供'}) 狀態: {status_emoji}\n" # Handle None location
                    response_text += (
                        "\n管理訂閱:\n"
                        "• 訂閱設備 [設備ID] - 新增訂閱\n"
                        "• 取消訂閱 [設備ID] - 取消訂閱\n"
                    )
                message = TextMessage(text=response_text)
        except pyodbc.Error as db_err:
            logger.error(f"獲取我的訂閱清單時發生資料庫錯誤: {db_err}")
            message = TextMessage(text="資料庫查詢失敗，無法獲取您的訂閱清單。")
        except Exception as e:
            logger.error(f"獲取訂閱清單失敗: {e}")
            message = TextMessage(text="獲取訂閱清單失敗，請稍後再試。")
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token, messages=[message]
        )
        line_bot_api.reply_message_with_http_info(reply_request)

    # 預設：從 ChatGPT 取得回應
    else:
        try:
            # Ensure user_id is passed for database operations within reply_message if needed
            from src.main import reply_message

            # Validate text input length before sending to reply_message (OpenAI)
            if len(text) > 1000: # Example length limit
                message = TextMessage(text="您的訊息過長，請縮短後再試。")
            else:
                response_text = reply_message(event) # Assuming event contains user_id
                message = TextMessage(text=response_text)

            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token, messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)
        except InvalidSignatureError: # Already handled in callback, but good practice if called elsewhere
            logger.error("無效的簽名於 handle_message")
            abort(400) # Should not happen here if callback is entry point
        except pyodbc.Error as db_err: # Catch potential DB errors from reply_message
            logger.error(f"處理訊息時發生資料庫錯誤: {db_err}")
            message = TextMessage(text="處理您的訊息時發生資料庫問題，請稍後再試。")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token, messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)
        except Exception as e: # General exception for reply_message logic or LINE API
            logger.error(f"回覆訊息失敗: {e}")
            message = TextMessage(text="抱歉，處理您的請求時發生錯誤，請稍後再試。")
            reply_request = ReplyMessageRequest(
                reply_token=event.reply_token, messages=[message]
            )
            line_bot_api.reply_message_with_http_info(reply_request)


def send_notification(user_id, message_text): # Renamed message to message_text to avoid conflict
    """發送 LINE 訊息給特定使用者"""
    if not user_id or not message_text: # Basic validation
        logger.error("send_notification: user_id 或 message_text 為空。")
        return False
    if len(message_text) > 2000: # LINE message length limit
        logger.warning("send_notification: 訊息內容過長，可能無法完整發送。")
        message_text = message_text[:2000]

    try:
        message_obj = TextMessage(text=message_text)
        push_request = PushMessageRequest(to=user_id, messages=[message_obj])
        # Using _with_http_info for more detailed error potential, though not strictly necessary here
        response = line_bot_api.push_message_with_http_info(push_request)
        if response.status_code == 200:
            logger.info(f"成功發送通知給 {user_id}")
            return True
        else:
            logger.error(f"發送通知給 {user_id} 失敗，狀態碼: {response.status_code}, 回應: {response.data}")
            return False
    except Exception as e: # More specific exceptions can be caught from linebot SDK if needed
        logger.error(f"發送通知給 {user_id} 時發生例外: {e}")
        return False


if __name__ == "__main__":
    initialize_equipment_data()
    start_scheduler()
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    port = int(os.environ.get("PORT", os.getenv("HTTPS_PORT", 443)))
    print("啟動伺服器中……")
    app.run(ssl_context=(
        os.environ.get('SSL_CERT_PATH', 'certs/capstone-project.me-chain.pem'),  # fullchain
        os.environ.get('SSL_KEY_PATH', 'certs/capstone-project.me-key.pem')),  # key
        host="0.0.0.0", port=port, debug=debug_mode)
