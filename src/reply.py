"""
一個簡單的幫助函數，返回一個 TextMessage 物件，包含使用說明和快速回覆選項。

Returns:
    TextMessage: 包含使用說明和快速回覆選項的 TextMessage 物件。
"""
from linebot.v3.messaging import (
    CarouselColumn,
    CarouselTemplate,
    MessageAction,
    QuickReply,
    QuickReplyItem,
    TemplateMessage,
    TextMessage,
)
from typing import Callable, List, Tuple
import logging
import pyodbc
import re

logger = logging.getLogger(__name__)


def __help() -> TextMessage:
    """顯示幫助訊息"""
    quick_reply = QuickReply(
        items=[
            QuickReplyItem(action=MessageAction(label="查看報表", text="powerbi")),
            QuickReplyItem(action=MessageAction(label="我的訂閱", text="我的訂閱")),
            QuickReplyItem(action=MessageAction(label="訂閱設備", text="訂閱設備")),
            QuickReplyItem(action=MessageAction(label="設備狀態", text="設備狀態")),
            QuickReplyItem(action=MessageAction(label="使用說明", text="使用說明")),
        ]
    )
    return TextMessage(
        text="您可以選擇以下選項或直接輸入您的問題：", quick_reply=quick_reply
    )


def __guide() -> TextMessage:
    """顯示使用指南訊息"""
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
                text="輸入 'language:語言代碼' 更改語言。\n目前支援：\nlanguage:zh-Hant (繁中)",
                actions=[MessageAction(label="設定為繁體中文", text="language:zh-Hant")],
            ),
        ]
    )
    reply_message_obj = TemplateMessage(
        alt_text="使用說明", template=carousel_template
    )
    return reply_message_obj


def __about() -> TextMessage:
    """顯示關於訊息"""
    reply_message_obj = TextMessage(
            text=(
                "這是一個整合 LINE Bot 與 Ollama 的智能助理，"
                "可以回答您的技術問題、監控半導體設備狀態並展示。"
                "您可以輸入 'help' 查看更多功能。"
            )
        )
    return reply_message_obj


def __language() -> TextMessage:
    reply_message_obj = TextMessage(
            text=(
                "您可以通過輸入以下命令設置語言：\n\n"
                "language:zh-Hant - 繁體中文"
            )
        )
    return reply_message_obj


def __set_language(text: str, db, user_id) -> TextMessage:
    """設置語言"""
    try:
        lang_code_input = text.split(":", 1)[1].strip().lower()
        valid_langs = {"zh-hant": "zh-Hant", "zh": "zh-Hant"}
        lang_to_set = valid_langs.get(lang_code_input)

        if lang_to_set:
            if db.set_user_preference(user_id, language=lang_to_set):
                confirmation_map = {"zh-Hant": "語言已切換至 繁體中文"}
                reply_message_obj = TextMessage(
                    text=confirmation_map.get(lang_to_set, f"語言已設定為 {lang_to_set}")
                )
            else:
                reply_message_obj = TextMessage(text="語言設定失敗，請稍後再試。")
        else:
            reply_message_obj = TextMessage(
                text="不支援的語言代碼。目前支援：zh-Hant (繁體中文)"
            )
    except IndexError:
        reply_message_obj = TextMessage(text="格式錯誤。請使用: language:zh-Hant")
    return reply_message_obj


def __equipment_status(db) -> TextMessage:
    """顯示設備狀態訊息"""
    try:
        with db._get_connection() as conn:  # 使用 MS SQL Server 連線
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.equipment_type, COUNT(*) as total,
                        SUM(CASE WHEN e.status = 'normal' THEN 1 ELSE 0 END) as normal_count,
                        SUM(CASE WHEN e.status = 'warning' THEN 1 ELSE 0 END) as warning_count,
                        SUM(CASE WHEN e.status = 'critical' THEN 1 ELSE 0 END) as critical_count,
                        SUM(CASE WHEN e.status = 'emergency' THEN 1 ELSE 0 END) as emergency_count,
                        SUM(CASE WHEN e.status = 'offline' THEN 1 ELSE 0 END) as offline_count
                FROM equipment e
                GROUP BY e.equipment_type;
                """
            )
            stats = cursor.fetchall()
            if not stats:
                reply_message_obj = TextMessage(text="目前尚未設定任何設備。")
            else:
                response_text = "📊 設備狀態摘要：\n\n"
                for row in stats:
                    equipment_type_db, total, normal, warning, critical, emergency, offline = row
                    type_name = {"dicer": "切割機"}.get(equipment_type_db, equipment_type_db)
                    response_text += f"{type_name}：總數 {total}, 正常 {normal}"
                    if warning > 0:
                        response_text += f", 警告 {warning}"
                    if critical > 0:
                        response_text += f", 嚴重 {critical}"
                    if emergency > 0:
                        response_text += f", 緊急 {emergency}"
                    if offline > 0:
                        response_text += f", 離線 {offline}"
                    response_text += "\n"

                cursor.execute(
                    """
                    SELECT TOP 5 e.name, e.equipment_type, e.status, e.equipment_id,
                                 ah.detected_anomaly_type, ah.created_time
                    FROM equipment e
                    LEFT JOIN alert_history ah ON e.equipment_id = ah.equipment_id
                        AND ah.is_resolved = 0
                        AND ah.equipment_id = (
                            SELECT MAX(ah_inner.equipment_id)
                            FROM alert_history ah_inner
                            WHERE ah_inner.equipment_id = e.equipment_id AND ah_inner.is_resolved = 0
                        )
                    WHERE e.status NOT IN ('normal', 'offline')
                    ORDER BY CASE e.status
                        WHEN 'emergency' THEN 1
                        WHEN 'critical' THEN 2
                        WHEN 'warning' THEN 3
                        ELSE 4
                    END, ah.created_time DESC;
                    """
                )
                abnormal_equipments = cursor.fetchall()
                if abnormal_equipments:
                    response_text += "\n⚠️ 近期異常設備 (最多5筆)：\n\n"
                    for name_db, equipment_type, status, eq_id, alert_t, alert_time in abnormal_equipments:
                        type_name = {
                            "dicer": "切割機"
                        }.get(equipment_type, equipment_type)
                        status_emoji = {
                            "warning": "⚠️", "critical": "🔴", "emergency": "🚨"
                        }.get(status, "❓")
                        response_text += (
                            f"{name_db} ({type_name}) 狀態: {status_emoji} {status}\n"
                        )
                        if alert_t and alert_time:
                            response_text += (
                                f"  最新警告: {alert_t} "
                                f"於 {alert_time.strftime('%Y-%m-%d %H:%M')}\n"
                            )
                    response_text += "\n輸入「設備詳情 [設備名稱]」可查看更多資訊。"
                reply_message_obj = TextMessage(text=response_text)
    except pyodbc.Error as db_err:
        logger.error(f"取得設備狀態失敗 (MS SQL Server): {db_err}")
        reply_message_obj = TextMessage(text="取得設備狀態失敗，請稍後再試。")
    except Exception as e:
        logger.error(f"處理設備狀態查詢時發生未知錯誤: {e}")
        reply_message_obj = TextMessage(text="系統忙碌中，請稍候再試。")
    return reply_message_obj


def __subscribe_equipment(text, db, user_id: str) -> TextMessage:
    """訂閱設備"""
    # Improved splitting logic: split by any whitespace
    parts = text.split(maxsplit=1)

    if len(parts) < 2 or not parts[1].strip():  # 指令為 "訂閱設備" (無參數)
        try:
            with db._get_connection() as conn:  # 使用 MS SQL Server 連線
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT equipment_id, name, equipment_type, location "
                    "FROM equipment ORDER BY equipment_type, name;"
                )
                equipments = cursor.fetchall()
                if not equipments:
                    reply_message_obj = TextMessage(text="目前沒有可用的設備進行訂閱。")
                else:
                    quick_reply_items = []
                    response_text_header = (
                        "請選擇要訂閱的設備 (或輸入 '訂閱設備 [設備ID]'):\n\n"
                    )
                    response_text_list = ""
                    for eq_id, name_db, equipment_type, loc in equipments[:13]:  # LINE QuickReply 最多13個
                        type_name = {
                           "dicer": "切割機"
                        }.get(equipment_type, equipment_type)
                        label = f"{name_db} ({type_name})"
                        quick_reply_items.append(
                            QuickReplyItem(action=MessageAction(
                                label=label[:20], text=f"訂閱設備 {eq_id}"
                            ))
                        )
                        response_text_list += (
                            f"- {name_db} ({type_name}, {loc or 'N/A'}), "
                            f"ID: {eq_id}\n"
                        )
                    if quick_reply_items:
                        reply_message_obj = TextMessage(
                            text=response_text_header + response_text_list,
                            quick_reply=QuickReply(items=quick_reply_items)
                        )
                    else:
                        reply_message_obj = TextMessage(
                            text=(
                                f"{response_text_header}{response_text_list}\n"
                                "使用方式: 訂閱設備 [設備ID]\n例如: 訂閱設備 DB001"
                            )
                        )
        except pyodbc.Error as db_err:
            logger.error(f"獲取設備清單失敗 (MS SQL Server): {db_err}")
            reply_message_obj = TextMessage(text="獲取設備清單失敗，請稍後再試。")
        except Exception as e:
            logger.error(f"處理訂閱設備列表時發生未知錯誤: {e}")
            reply_message_obj = TextMessage(text="系統忙碌中，請稍候再試。")
    else:  # 指令為 "訂閱設備 [ID]"
        equipment_id_to_subscribe = parts[1].strip().upper()  # ID 通常大寫
        try:
            with db._get_connection() as conn:  # 使用 MS SQL Server 連線
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM equipment WHERE equipment_id = ?;",
                    (equipment_id_to_subscribe,)
                )
                equipment = cursor.fetchone()
                if not equipment:
                    reply_message_obj = TextMessage(
                        text=f"查無設備 ID「{equipment_id_to_subscribe}」。請檢查 ID 是否正確。"
                    )
                else:
                    equipment_name_db = equipment[0]
                    cursor.execute(
                        "SELECT equipment_id FROM user_equipment_subscriptions "
                        "WHERE user_id = ? AND equipment_id = ?;",
                        (user_id, equipment_id_to_subscribe)
                    )
                    if cursor.fetchone():
                        reply_message_obj = TextMessage(
                            text=f"您已訂閱設備 {equipment_name_db} ({equipment_id_to_subscribe})。"
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO user_equipment_subscriptions "
                            "(user_id, equipment_id, notification_level) "
                            "VALUES (?, ?, 'all');",
                            (user_id, equipment_id_to_subscribe)
                        )
                        conn.commit()
                        reply_message_obj = TextMessage(
                            text=f"已成功訂閱設備 {equipment_name_db} ({equipment_id_to_subscribe})！"
                        )
        except pyodbc.IntegrityError:
            logger.warning(
                f"嘗試重複訂閱設備 {equipment_id_to_subscribe} for user {user_id}"
            )
            reply_message_obj = TextMessage(
                text=f"您似乎已訂閱設備 {equipment_id_to_subscribe}。"
            )
        except pyodbc.Error as db_err:
            logger.error(f"訂閱設備失敗 (MS SQL Server): {db_err}")
            reply_message_obj = TextMessage(
                text="訂閱設備失敗，資料庫操作錯誤，請稍後再試。"
            )
        except Exception as e:
            logger.error(f"處理訂閱設備時發生未知錯誤: {e}")
            reply_message_obj = TextMessage(text="系統忙碌中，請稍候再試。")
    return reply_message_obj


def __unsubscribe_equipment(text: str, db, user_id: str) -> TextMessage:
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():  # 指令為 "取消訂閱"
        try:
            with db._get_connection() as conn:  # 使用 MS SQL Server 連線
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT s.equipment_id, e.name, e.equipment_type
                    FROM user_equipment_subscriptions s
                    JOIN equipment e ON s.equipment_id = e.equipment_id
                    WHERE s.user_id = ?
                    ORDER BY e.equipment_type, e.name;
                    """, (user_id,)
                )
                subscriptions = cursor.fetchall()
                if not subscriptions:
                    reply_message_obj = TextMessage(text="您目前沒有訂閱任何設備。")
                else:
                    quick_reply_items = []
                    response_text_header = (
                        "您已訂閱的設備 (點擊取消訂閱或輸入 '取消訂閱 [設備ID]'):\n\n"
                    )
                    response_text_list = ""
                    for eq_id, name_db, equipment_type in subscriptions[:13]:  # QuickReply上限
                        type_name = {
                            "dicer": "切割機"
                        }.get(equipment_type, equipment_type)
                        label = f"{name_db} ({type_name})"
                        quick_reply_items.append(
                            QuickReplyItem(action=MessageAction(
                                label=label[:20], text=f"取消訂閱 {eq_id}"
                            ))
                        )
                        response_text_list += f"- {name_db} ({type_name}), ID: {eq_id}\n"
                    if quick_reply_items:
                        reply_message_obj = TextMessage(
                            text=response_text_header + response_text_list,
                            quick_reply=QuickReply(items=quick_reply_items)
                        )
                    else:
                        reply_message_obj = TextMessage(
                            text=(
                                f"{response_text_header}{response_text_list}\n"
                                "使用方式: 取消訂閱 [設備ID]\n例如: 取消訂閱 DB001"
                            )
                        )
        except pyodbc.Error as db_err:
            logger.error(f"獲取訂閱清單失敗 (MS SQL Server): {db_err}")
            reply_message_obj = TextMessage(text="獲取訂閱清單失敗，請稍後再試。")
        except Exception as e:
            logger.error(f"處理取消訂閱列表時發生未知錯誤: {e}")
            reply_message_obj = TextMessage(text="系統忙碌中，請稍候再試。")
    else:  # 指令為 "取消訂閱 [ID]"
        equipment_id_to_unsubscribe = parts[1].strip().upper()
        try:
            with db._get_connection() as conn:  # 使用 MS SQL Server 連線
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM equipment WHERE equipment_id = ?;",
                    (equipment_id_to_unsubscribe,)
                )
                equipment_info = cursor.fetchone()
                if not equipment_info:
                    reply_message_obj = TextMessage(
                        text=f"查無設備 ID「{equipment_id_to_unsubscribe}」。"
                    )
                else:
                    # equipment_name_db = equipment_info[0] # 未使用
                    cursor.execute(
                        "DELETE FROM user_equipment_subscriptions "
                        "WHERE user_id = ? AND equipment_id = ?;",
                        (user_id, equipment_id_to_unsubscribe)
                    )
                    conn.commit()
                    if cursor.rowcount > 0:
                        reply_message_obj = TextMessage(
                            text=f"已成功取消訂閱設備 {equipment_id_to_unsubscribe}。"
                        )
                    else:
                        reply_message_obj = TextMessage(
                            text=f"您並未訂閱設備 {equipment_id_to_unsubscribe}。"
                        )
        except pyodbc.Error as db_err:
            logger.error(f"取消訂閱失敗 (MS SQL Server): {db_err}")
            reply_message_obj = TextMessage(text="取消訂閱設備失敗，請稍後再試。")
        except Exception as e:
            logger.error(f"處理取消訂閱時發生未知錯誤: {e}")
            reply_message_obj = TextMessage(text="系統忙碌中，請稍候再試。")
    return reply_message_obj


def __my_subscriptions(db, user_id: str) -> TextMessage:
    """顯示用戶訂閱"""
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.equipment_id, e.name, e.equipment_type, e.location, e.status
                FROM user_equipment_subscriptions s
                JOIN equipment e ON s.equipment_id = e.equipment_id
                WHERE s.user_id = ?
                ORDER BY e.equipment_type, e.name;
                """, (user_id,)
            )
            subscriptions = cursor.fetchall()
            if not subscriptions:
                response_text = (
                    "您目前沒有訂閱任何設備。\n\n"
                    "請使用「訂閱設備」指令查看可訂閱的設備列表。"
                )
            else:
                response_text = "您已訂閱的設備：\n\n"
                for equipment_id, name_db, equipment_type, loc, status in subscriptions:
                    type_name = {
                        "dicer": "切割機"
                    }.get(equipment_type, equipment_type)
                    # 這裡原本有status_emoji，但沒有實機所以移除，之後可再改成停機，運作，或保養狀態
                    response_text += (
                        f"- {name_db} ({type_name}, {loc or 'N/A'}), "
                        f"ID: {equipment_id}, 狀態: {status}\n"
                    )
                response_text += (
                    "\n管理訂閱:\n• 訂閱設備 [設備ID]\n• 取消訂閱 [設備ID]"
                )
            reply_message_obj = TextMessage(text=response_text)
    except pyodbc.Error as db_err:
        logger.error(f"獲取我的訂閱清單失敗 (MS SQL Server): {db_err}")
        reply_message_obj = TextMessage(text="獲取訂閱清單失敗，請稍後再試。")
    except Exception as e:
        logger.error(f"處理我的訂閱時發生未知錯誤: {e}")
        reply_message_obj = TextMessage(text="系統忙碌中，請稍候再試。")
    return reply_message_obj


def __equipment_details(text: str, db, user_id: str) -> TextMessage:
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        reply_message_obj = TextMessage(
            text="請指定設備名稱或ID，例如「設備詳情 黏晶機A1」或「設備詳情 DB001」"
        )
        return reply_message_obj

    equipment_name = parts[1].strip()

    if equipment_name:  # 確保 equipment_name 已被賦值
        try:
            with db._get_connection() as conn:  # 使用 MS SQL Server 連線
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT e.equipment_id, e.name, e.equipment_type, e.status,
                           e.location, e.last_updated
                    FROM equipment e
                    WHERE e.name LIKE ? OR e.equipment_id = ?;
                    """,
                    (f"%{equipment_name}%", equipment_name.upper())
                )
                equipment = cursor.fetchone()
                if not equipment:
                    reply_message_obj = TextMessage(
                        text=f"查無設備「{equipment_name}」的資料。"
                    )
                else:
                    eq_id, name_db, equipment_type, status, location, last_updated_db = equipment
                    type_name = {
                        "dicer": "切割機"
                    }.get(equipment_type, equipment_type)
                    status_emoji = {
                        "normal": "✅", "warning": "⚠️", "critical": "🔴",
                        "emergency": "🚨", "offline": "⚫"
                    }.get(status, "❓")
                    last_updated_str = (
                        last_updated_db.strftime('%Y-%m-%d %H:%M:%S')
                        if last_updated_db else '未記錄'
                    )
                    response_text = (
                        f"設備詳情： {name_db} ({eq_id})\n"
                        f"類型: {type_name}\n"
                        f"狀態: {status_emoji} {status}\n"
                        f"地點: {location or '未提供'}\n"
                        f"最後更新: {last_updated_str}\n\n"
                    )
                    cursor.execute(
                        """
                        WITH RankedMetrics AS (
                            SELECT
                                em.metric_type, em.value, em.unit, em.last_updated,
                                ROW_NUMBER() OVER(
                                    PARTITION BY em.metric_type ORDER BY em.last_updated DESC
                                ) as rn
                            FROM equipment_metrics em
                            WHERE em.equipment_id = ?
                        )
                        SELECT metric_type, value, unit, last_updated
                        FROM RankedMetrics
                        WHERE rn = 1
                        ORDER BY metric_type;
                        """, (eq_id,)
                    )
                    metrics = cursor.fetchall()
                    if metrics:
                        response_text += "📊 最新監測值：\n"
                        for metric_t, val, unit, ts in metrics:
                            response_text += (
                                f"  {metric_t}: {val:.2f} {unit or ''} "
                                f"({ts.strftime('%H:%M:%S')})\n"
                            )
                    else:
                        response_text += "暫無最新監測指標。\n"
                    cursor.execute(
                        """
                        SELECT TOP 3 detected_anomaly_type, severity_level, created_time, message
                        FROM alert_history
                        WHERE equipment_id = ? AND is_resolved = 0
                        ORDER BY created_time DESC;
                        """, (eq_id,)
                    )
                    alerts = cursor.fetchall()
                    if alerts:
                        response_text += "\n⚠️ 未解決的警報：\n"
                        for alert_t, severity_level, alert_time, _ in alerts:  # msg_content not used
                            sev_emoji = {
                                "warning": "⚠️", "critical": "🔴", "emergency": "🚨"
                            }.get(severity_level, "ℹ️")
                            response_text += (
                                f"  {sev_emoji} {alert_t} ({severity_level}) "
                                f"於 {alert_time.strftime('%Y-%m-%d %H:%M')}\n"
                            )
                    else:
                        response_text += "\n目前無未解決的警報。\n"
                    # 請注意:這裡原本有equipment_operation_logs顯示訂單資訊，但無實體訂單所以刪除
                    reply_message_obj = TextMessage(text=response_text.strip())
        except pyodbc.Error as db_err:
            logger.error(f"取得設備詳情失敗 (MS SQL Server): {db_err}")
            reply_message_obj = TextMessage(text="取得設備詳情失敗，請稍後再試。")
        except Exception as e:
            logger.error(f"處理設備詳情查詢時發生未知錯誤: {e}")
            reply_message_obj = TextMessage(text="系統忙碌中，請稍候再試。")
        return reply_message_obj


__commands = {
    "help": __help, "幫助": __help, "選單": __help, "menu": __help,
    "使用說明": __guide, "說明": __guide, "教學": __guide, "指南": __guide, "guide": __guide,
    "關於": __about, "about": __about,
    "language": __language, "語言": __language,
    "設備狀態": __equipment_status, "機台狀態": __equipment_status, "equipment status": __equipment_status,
    "我的訂閱": __my_subscriptions, "my subscriptions": __my_subscriptions,
}

# Improved regex-like matching using simpler string checks for now, can be upgraded to full regex if needed
__fuzzy_commands: List[Tuple[Callable[[str], bool], Callable[[str], TextMessage]]] = [
    (lambda text: text.startswith("language:") or text.startswith("語言:"), __set_language),
    (lambda text: "訂閱設備" in text or "subscribe equipment" in text, __subscribe_equipment),
    (lambda text: "取消訂閱" in text or "unsubscribe" in text, __unsubscribe_equipment),
    (lambda text: "設備詳情" in text or "機台詳情" in text, __equipment_details),
]


def __get_command(text: str) -> Callable[[str], TextMessage]:
    """根據輸入文字返回對應的命令函數"""
    # Case-insensitive cleanup
    normalized_text = text.strip().lower()

    if normalized_text in __commands:
        return __commands[normalized_text]

    # Check fuzzy commands with original text (some might need it, though we check lower inside lambda usually)
    # Adjust lambda to check normalized text if consistent
    for condition, command in __fuzzy_commands:
        if condition(normalized_text):
            return command
    return None


def dispatch_command(text: str, db, user_id: str):
    """根據輸入文字調度對應的命令函數，並返回 TextMessage物件"""
    cmd = __get_command(text)
    if cmd is None:
        return None

    # A more robust way to dispatch commands by inspecting their signature
    import inspect
    sig = inspect.signature(cmd)

    # Prepare arguments to pass to the command function
    kwargs = {}
    if 'text' in sig.parameters:
        kwargs['text'] = text
    if 'db' in sig.parameters:
        kwargs['db'] = db
    if 'user_id' in sig.parameters:
        kwargs['user_id'] = user_id

    return cmd(**kwargs)
