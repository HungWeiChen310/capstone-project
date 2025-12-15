# src/services/line_service.py
import os
import logging
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
    MulticastRequest,
)
from linebot.v3.messaging.exceptions import ApiException
from ..database import db
from src.services.command_service import dispatch_command
from src.main import reply_message as main_reply_message

logger = logging.getLogger(__name__)

# LINE Bot API setup
channel_access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not channel_access_token:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN not set")

configuration = Configuration(access_token=channel_access_token)
api_client = ApiClient(configuration)
line_bot_api = MessagingApi(api_client)


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


def send_multicast_notification(user_ids_to_notify, message_text):
    """
    發送 LINE 訊息給多個使用者 (Multicast)。
    這比多次調用 send_notification 更有效率，因為它只發送一次 HTTP 請求。
    """
    if not user_ids_to_notify:
        return True

    try:
        # LINE Multicast 最多支援 500 個使用者 ID
        # 如果需要發送超過 500 人，應在此處進行分批處理

        message_obj = TextMessage(text=message_text)
        multicast_request = MulticastRequest(to=user_ids_to_notify, messages=[message_obj])
        line_bot_api.multicast(multicast_request)
        logger.info(f"Multicast 通知已成功發送給 {len(user_ids_to_notify)} 位使用者")
        return True
    except Exception as e:
        logger.error(f"發送 Multicast 通知失敗: {e}")
        # 如果 Multicast 失敗，可以考慮退回到逐一發送 (但這會很慢)，
        # 或者只記錄錯誤。這裡選擇記錄錯誤。
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


def process_message_in_background(event):
    """在背景處理耗時的訊息回覆"""
    text = event.message.text.strip()
    text_lower = text.lower()
    user_id = event.source.user_id

    db.get_user_preference(user_id)

    reply_message_obj = dispatch_command(
        text_lower, db, user_id
    )
    if reply_message_obj is None:
        try:
            response_text = main_reply_message(event)
            reply_message_obj = TextMessage(text=response_text)
        except ImportError:
            logger.error("無法導入 src.main.reply_message")
            reply_message_obj = TextMessage(text="抱歉，AI 對話功能暫時無法使用。")
        except Exception as e:
            logger.exception(f"調用 Ollama 回覆訊息失敗: {e}")
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
