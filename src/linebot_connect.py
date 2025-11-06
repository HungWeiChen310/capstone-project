import datetime
import functools
import logging
import os
import secrets
import threading  # 保留 threading
import time
from collections import defaultdict
import reply


from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
    jsonify
)
from flask_talisman import Talisman
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging.exceptions import ApiException
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from werkzeug.middleware.proxy_fix import ProxyFix

from database import db  # db 物件現在是 MS SQL Server 的接口
# F401: 下面兩個匯入在此檔案中未使用，通常在 app.py 中調用
# from equipment_scheduler import start_scheduler
# from initial_data import initialize_equipment_data

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

# 密鑰文件路徑可由環境變數覆蓋
SECRET_KEY_FILE = os.getenv("SECRET_KEY_FILE", "data/secret_key.txt")


def get_or_create_secret_key():
    """獲取或創建一個固定的 secret key"""
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

    if current_time - last_cleanup_time < 3600:
        return

    with request_counts_lock:
        ips_to_remove = [
            ip for ip, timestamps in request_counts.items()
            if not timestamps or current_time - max(timestamps) > 3600
        ]
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


ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")


def admin_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "templates"
        ),
        static_folder=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "static"
        )
    )
    app.secret_key = get_or_create_secret_key()
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

    csp = {
        "default-src": "'self'",
        "script-src": ["'self'", "'unsafe-inline'", "https://cdn.powerbi.com"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:"],
        "frame-src": ["https://app.powerbi.com"],
        "connect-src": [
            "'self'", "https://api.powerbi.com", "https://login.microsoftonline.com"
        ],
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

configuration = Configuration(access_token=channel_access_token)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)
handler = WebhookHandler(channel_secret)


def register_routes(app_instance):  # 傳入 app 實例
    @app_instance.route("/callback", methods=["POST"])
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

    @app_instance.route("/")
    def index():
        return render_template("index.html")

    @app_instance.route("/admin/login", methods=["GET", "POST"])
    def admin_login():
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                session["admin_logged_in"] = True
                session.permanent = True  # 可選：使 session 持久
                app_instance.permanent_session_lifetime = datetime.timedelta(days=7)  # 可選：設定持久時間
                return redirect(request.args.get("next") or url_for("admin_dashboard"))
            else:
                flash("登入失敗，請確認帳號密碼是否正確", "error")
        return render_template("admin_login.html")

    @app_instance.route("/admin/logout")
    def admin_logout():
        session.pop("admin_logged_in", None)
        return redirect(url_for("admin_login"))

    @app_instance.route("/admin/dashboard")
    @admin_required
    def admin_dashboard():
        # 直接使用 db 物件的方法
        conversation_stats = db.get_conversation_stats()
        recent_conversations = db.get_recent_conversations(limit=20)  # 使用 user_id
        system_info = {
            "ollama_endpoint": f"{os.getenv('OLLAMA_HOST', '127.0.0.1')}:{os.getenv('OLLAMA_PORT', '11434')}",
            "ollama_model": os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
            "line_channel_secret": "已設置" if os.getenv("LINE_CHANNEL_SECRET") else "未設置",
            "db_server": os.getenv("DB_SERVER", "localhost"),
            "db_name": os.getenv("DB_NAME", "conversations")
        }
        return render_template(
            "admin_dashboard.html",
            stats=conversation_stats,
            recent=recent_conversations,  # recent 列表中的 user_id (原 sender_id)
            system_info=system_info,
        )

    @app_instance.route("/admin/conversation/<user_id>")  # 這裡的 user_id 是正確的
    @admin_required
    def admin_view_conversation(user_id):
        # 直接使用 db 物件的方法
        # get_conversation_history 以 user_id (即 sender_id) 查詢
        conversation = db.get_conversation_history(user_id, limit=50)
        user_info = db.get_user_preference(user_id)
        return render_template(
            "admin_conversation.html",
            conversation=conversation,
            user_id=user_id,
            user_info=user_info,
        )

    @app_instance.template_filter("nl2br")
    def nl2br(value):
        if not value:
            return ""
        return value.replace("\n", "<br>")

    @app_instance.context_processor
    def utility_processor():
        def now_func():
            return datetime.datetime.now()
        return dict(now=now_func)

    @app_instance.route("/alarms", methods=["POST"])
    def alarms():
        """接收警報訊息"""
        data = request.get_json(force=True, silent=True)
        key = ("equipment_id", "detected_anomaly_type", "severity_level")
        if data and all(k in data for k in key):
            data["created_time"] = str(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            db.insert_alert_history(log_data=data)
            equipment_id = data["equipment_id"]
            subscribers = db.get_subscribed_users(equipment_id)
            if subscribers:
                message_text = (
                    f"設備 {equipment_id} 在 {data['created_time']} 時發生 {data['detected_anomaly_type']} 警報，"  # 新增發生異常時間
                    f"嚴重程度 {data['severity_level']}。"
                )
                for user in subscribers:
                    send_notification(user, message_text)
            else:
                logger.info(f"No subscribers found for equipment {equipment_id}")

        logger.info("Received JSON from client:", data)
        return jsonify({"status": "success"}), 200

    @app_instance.route("/resolvealarms", methods=["POST"])
    def resolve_alarms():
        """接收警報解決訊息"""
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "message": "No JSON data received."}), 400
        # 檢查必要的 key (新增detected_anomaly_type 跟 equipment_id 讓定位更加準確)
        required_keys = ("error_id", "detected_anomaly_type", "equipment_id", "resolved_by")
        if not all(k in data for k in required_keys):
            return jsonify({
                "status": "error", "message": "Missing required keys: error_id, detected_anomaly_type, equipment_id, resolved_by."
            }), 400
        try:
            # 呼叫 database.py 中 resolve_alert_history 函式，會回傳三種可能的結果
            db_result = db.resolve_alert_history(log_data=data)

            # 情況1:找不到對應的 error_id 和 detected_anomaly_type 和 equipment_id，直接回傳錯誤
            if db_result is None:
                logger.warning(
                    f"嘗試解決警報失敗，找不到 error_id: {data['error_id']} / "
                    f"detected_anomaly_type: {data['detected_anomaly_type']} / "
                    f"equipment_id: {data['equipment_id']}。"
                )
                return jsonify({
                    "status": "error",
                    "message": (
                        f"Alarm with error_id {data['error_id']} "
                        f"and detected_anomaly_type {data['detected_anomaly_type']} "
                        f"and equipment_id {data['equipment_id']} not found."
                    )
                }), 404
            # 情況2:警報先前已被解決，不發送通知
            elif isinstance(db_result, tuple):
                logger.info(
                    f"警報 {data['error_id']} / detected_anomaly_type: {data['detected_anomaly_type']} / "
                    f"equipment_id: {data['equipment_id']} 先前已被解決，不發送通知。")
                return jsonify({
                    "status": "success", "message": "Alarm was already resolved. No notification sent."
                }), 200
            # 情況3:成功更新警報，只有這種情況才發送通知
            else:
                # 準備訊息內容
                equipment_id = data['equipment_id']
                detected_anomaly_type = data['detected_anomaly_type']
                resolved_time_from_db = db_result

                # 查找訂閱者
                subscribers = db.get_subscribed_users(equipment_id)
                if subscribers:
                    # 建立新的通知訊息
                    message_text = (
                        f"設備 {equipment_id} 發生 {detected_anomaly_type} 警報，"
                        f"在 {resolved_time_from_db.strftime('%Y-%m-%d %H:%M:%S')} 由 {data['resolved_by']} 解決。"
                        f"解決說明: {data.get('resolution_notes') or '無'}"
                    )
                    # 發送通知
                    for user in subscribers:
                        send_notification(user, message_text)
                else:
                    logger.info(f"No subscribers found for equipment {equipment_id}")

                return jsonify({"status": "success", "message": "Alarm resolved and notification sent."}), 200

        except Exception as e:
            logger.error(f"處理警報解決請求時發生錯誤: {e}")
            return jsonify({"status": "error", "message": "An internal error occurred."}), 500


register_routes(app)


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    text = event.message.text.strip()
    text_lower = text.lower()
    user_id = event.source.user_id  # 獲取 user_id

    db.get_user_preference(user_id)  # 如果不存在，會在 get_user_preference 中創建

    reply_message_obj = reply.dispatch_command(
        text_lower, db, user_id
    )
    if reply_message_obj is None:
        try:
            from main import reply_message as main_reply_message
            response_text = main_reply_message(event)
            reply_message_obj = TextMessage(text=response_text)
        except ImportError:
            logger.error("無法導入 src.main.reply_message")
            reply_message_obj = TextMessage(text="抱歉，AI 對話功能暫時無法使用。")
        except Exception as e:
            logger.error(f"調用 Ollama 回覆訊息失敗: {e}")
            reply_message_obj = TextMessage(
                text="抱歉，處理您的請求時發生了錯誤，AI 功能可能暫時無法使用。"
            )

    if reply_message_obj:
        if not send_reply_with_fallback(event, reply_message_obj, user_id):
            logger.error(
                "回覆訊息與推播皆失敗，使用者 %s，訊息內容型別 %s",
                user_id,
                type(reply_message_obj).__name__,
            )
    else:
        logger.info(f"未處理的訊息: {text} from user {user_id}")
        unknown_command_reply = TextMessage(
            text="抱歉，我不太明白您的意思。您可以輸入 'help' 查看我能做些什麼。"
        )
        if not send_reply_with_fallback(event, unknown_command_reply, user_id):
            logger.error(
                "未知命令回覆失敗且推播失敗，使用者 %s",
                user_id,
            )


def send_notification(user_id_to_notify, message_text):
    """發送 LINE 訊息給特定使用者"""
    try:
        message_obj = TextMessage(text=message_text)
        push_request = PushMessageRequest(to=user_id_to_notify, messages=[message_obj])
        line_bot_api.push_message_with_http_info(push_request)
        logger.info(f"通知已成功發送給 user_id: {user_id_to_notify}")
        return True
    except Exception as e:
        logger.error(f"發送通知給 user_id {user_id_to_notify} 失敗: {e}")
        return False


def send_reply_with_fallback(event, message_obj, user_id):
    """嘗試回覆訊息，若 reply token 無效則改用 Push 模式"""
    messages = [message_obj]
    try:
        reply_request = ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=messages,
        )
        line_bot_api.reply_message_with_http_info(reply_request)
        return True
    except ApiException as api_exc:
        error_body = getattr(api_exc, "body", "") or ""
        combined_error = f"{error_body} {api_exc}".lower()
        logger.error(f"最終回覆訊息失敗: {api_exc}")
        if api_exc.status == 400 and "invalid reply token" in combined_error:
            logger.warning(
                "Reply token 無效，改以 push 訊息方式通知使用者 %s",
                user_id,
            )
            try:
                push_request = PushMessageRequest(to=user_id, messages=messages)
                line_bot_api.push_message_with_http_info(push_request)
                logger.info("Push 訊息成功，使用者 %s", user_id)
                return True
            except Exception as push_exc:
                logger.error(
                    "回覆失敗且推播同樣失敗，使用者 %s，錯誤: %s",
                    user_id,
                    push_exc,
                )
                return False
        return False
    except Exception as exc:
        logger.error(f"回覆訊息時發生未預期錯誤: {exc}")
        return False


if __name__ == "__main__":
    logger.info("linebot_connect.py 被直接執行。建議透過 app.py 啟動應用程式。")
