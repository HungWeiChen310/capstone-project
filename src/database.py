import logging
import os
import pyodbc
from config import Config


logger = logging.getLogger(__name__)


class Database:
    """處理對話記錄與使用者偏好儲存的資料庫處理程序"""

    def __init__(self, server=None, database=None):
        """初始化資料庫連線"""
        resolved_server = server if server is not None else Config.DB_SERVER
        resolved_database = database if database is not None else Config.DB_NAME
        self.connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={resolved_server};"
            f"DATABASE={resolved_database};"
            "Trusted_Connection=yes;"
        )
        self._initialize_db()

    def _get_connection(self):
        """建立並回傳資料庫連線"""
        return pyodbc.connect(self.connection_string)

    def _initialize_db(self):
        """
        如果資料表尚未存在，則建立必要的表格。
        此版本已加上主鍵與外鍵約束以確保資料完整性。
        """
        try:
            with self._get_connection() as conn:
                init_cur = conn.cursor()

            # 1. user_preferences
                user_preferences_cols = """
                    [user_id] NVARCHAR(255) NOT NULL PRIMARY KEY,
                    [language] VARCHAR(50) NULL,
                    [role] NVARCHAR(50) NULL,
                    [is_admin] BIT NULL,
                    [responsible_area] NVARCHAR(255) NULL,
                    [created_at] datetime2(2) NULL,
                    [display_name] NVARCHAR(255) NULL,
                    [email] NVARCHAR(255) NULL UNIQUE,
                    [last_active] datetime2(2) NULL
                """
                self._create_table_if_not_exists(init_cur, "user_preferences", user_preferences_cols)

                # 2. equipment
                equipment_cols = """
                    [id] INT NULL,
                    [equipment_id] NVARCHAR(255) NOT NULL PRIMARY KEY,
                    [name] NVARCHAR(255) NOT NULL,
                    [equipment_type] NVARCHAR(255) NULL,
                    [location] NVARCHAR(255) NULL,
                    [status] NVARCHAR(255) NULL,
                    [last_updated] datetime2(2) NULL
                """
                self._create_table_if_not_exists(init_cur, "equipment", equipment_cols)

                # 3. conversations
                conversations_cols = """
                    [message_id] NVARCHAR(255) NOT NULL PRIMARY KEY,
                    [sender_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES user_preferences(user_id),
                    [receiver_id] NVARCHAR(255) NULL,
                    [sender_role] NVARCHAR(50) NULL,
                    [content] NVARCHAR(MAX) NOT NULL,
                    [timestamp] datetime2(2) NULL DEFAULT GETDATE()
                """
                self._create_table_if_not_exists(init_cur, "conversations", conversations_cols)

                # 4. user_equipment_subscriptions
                user_equipment_subscriptions_cols = """
                    [subscription_id] INT IDENTITY(1,1) PRIMARY KEY,
                    [user_id] NVARCHAR(255) NOT NULL,
                    [notification_level] NVARCHAR(50) NULL,
                    [subscribed_at] datetime2(2) NULL DEFAULT GETDATE(),
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    CONSTRAINT FK_subscriptions_user FOREIGN KEY (user_id) REFERENCES user_preferences(user_id),
                    CONSTRAINT FK_subscriptions_equipment FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id),
                    CONSTRAINT UQ_user_equipment UNIQUE(user_id, equipment_id)
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "user_equipment_subscriptions",
                    user_equipment_subscriptions_cols
                )

                # 5. alert_history
                alert_history_cols = """
                    [error_id] INT NOT NULL PRIMARY KEY,
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [alert_type] NVARCHAR(255) NULL,
                    [severity] NVARCHAR(255) NULL,
                    [message] NVARCHAR(MAX) NULL,
                    [is_resolved] BIT NULL DEFAULT 0,
                    [created_time] datetime2(2) NULL,
                    [resolved_time] datetime2(2) NULL,
                    [resolved_by] NVARCHAR(255) NULL,
                    [resolution_notes] NVARCHAR(MAX) NULL
                """
                self._create_table_if_not_exists(init_cur, "alert_history", alert_history_cols)

                # 6. equipment_metrics
                equipment_metrics_cols = """
                    [id] INT NOT NULL PRIMARY KEY,
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [metric_type] NVARCHAR(255) NOT NULL,
                    [status] NVARCHAR(50) NULL,
                    [value] FLOAT NULL,
                    [threshold_min] FLOAT NULL,
                    [threshold_max] FLOAT NULL,
                    [unit] NVARCHAR(50) NULL,
                    [last_updated] datetime2(2) NULL DEFAULT GETDATE()
                """
                self._create_table_if_not_exists(init_cur, "equipment_metrics", equipment_metrics_cols)

                # 7. equipment_metric_thresholds
                # --- 關鍵修正 2: 新增了 normal_value 欄位 ---
                equipment_metric_thresholds_cols = """
                    [metric_type] NVARCHAR(50) NOT NULL PRIMARY KEY,
                    [normal_value] FLOAT NULL,
                    [warning_min] FLOAT NULL,
                    [warning_max] FLOAT NULL,
                    [critical_min] FLOAT NULL,
                    [critical_max] FLOAT NULL,
                    [emergency_op] NVARCHAR(10) NULL,
                    [emergency_min] FLOAT NULL,
                    [emergency_max] FLOAT NULL,
                    [last_updated] datetime2(2) NULL DEFAULT GETDATE()
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "equipment_metric_thresholds",
                    equipment_metric_thresholds_cols
                )

                # 8. error_logs
                error_logs_cols = """
                    [log_date] DATE NOT NULL,
                    [error_id] INT NOT NULL PRIMARY KEY,
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [deformation_mm] FLOAT NOT NULL,
                    [rpm] INT NOT NULL,
                    [event_time] datetime2(2) NOT NULL,
                    [detected_anomaly_type] NVARCHAR(MAX) NOT NULL,
                    [downtime_min] INT NULL,
                    [downtime_sec] INT NULL,
                    [resolved_time] datetime2(2) NULL,
                    [notes] NVARCHAR(MAX) NULL
                """

                self._create_table_if_not_exists(init_cur, "error_logs", error_logs_cols)

                # 9. stats_abnormal_monthly
                stats_abnormal_monthly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [month] INT NOT NULL,
                    [detected_anomaly_type] NVARCHAR(255) NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_hrs] FLOAT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, month, detected_anomaly_type)
                """
                self._create_table_if_not_exists(init_cur, "stats_abnormal_monthly", stats_abnormal_monthly_cols)

                # 10. stats_abnormal_quarterly
                stats_abnormal_quarterly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [quarter] INT NOT NULL,
                    [detected_anomaly_type] NVARCHAR(255) NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_hrs] FLOAT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, quarter, detected_anomaly_type)
                """
                self._create_table_if_not_exists(init_cur, "stats_abnormal_quarterly", stats_abnormal_quarterly_cols)

                # 11. stats_abnormal_yearly
                stats_abnormal_yearly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [detected_anomaly_type] NVARCHAR(255) NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_hrs] FLOAT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, detected_anomaly_type)
                """
                self._create_table_if_not_exists(init_cur, "stats_abnormal_yearly", stats_abnormal_yearly_cols)

                # 12. stats_operational_monthly
                stats_operational_monthly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [month] INT NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_hrs] FLOAT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, month)
                """
                self._create_table_if_not_exists(init_cur, "stats_operational_monthly", stats_operational_monthly_cols)

                # 13. stats_operational_quarterly
                stats_operational_quarterly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [quarter] INT NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_hrs] FLOAT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, quarter)
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "stats_operational_quarterly",
                    stats_operational_quarterly_cols
                )

                # 14. stats_operational_yearly
                stats_operational_yearly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL,
                    [year] INT NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_hrs] FLOAT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year),
                    CONSTRAINT FK_stats_op_yearly_equip FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                """
                self._create_table_if_not_exists(init_cur, "stats_operational_yearly", stats_operational_yearly_cols)

                conn.commit()
                logger.info(
                    "資料庫表格初始化/檢查完成 (已建立主鍵與外鍵約束)。"
                )
        except pyodbc.Error as e:
            logger.exception(f"資料庫初始化期間發生 pyodbc 錯誤: {e}")
            raise
        except Exception as ex:
            logger.exception(f"資料庫初始化期間發生非預期錯誤: {ex}")
            raise

    def _create_table_if_not_exists(self, cursor, table_name, columns_definition):
        """通用方法，用於檢查並建立資料表"""
        check_table_sql = (
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?;"
        )
        cursor.execute(check_table_sql, (table_name,))
        if cursor.fetchone()[0] == 0:
            create_table_sql = f"CREATE TABLE {table_name} ({columns_definition});"
            cursor.execute(create_table_sql)
            logger.info(f"資料表 '{table_name}' 已建立。")
        else:
            logger.info(f"資料表 '{table_name}' 已存在，跳過建立。")

    def add_message(self, sender_id, receiver_id, sender_role, content):
        """加入一筆新的對話記錄（包含發送者角色）"""
        try:
            with self._get_connection() as conn:
                conv_add_cur = conn.cursor()
                conv_add_cur.execute(
                    """
                    INSERT INTO conversations
                        (sender_id, receiver_id, sender_role, content)
                    VALUES (?, ?, ?, ?);
                    """,
                    (sender_id, receiver_id, sender_role, content)
                )
            conn.commit()
            return True
        except pyodbc.Error as e:
            logger.exception(f"新增對話記錄失敗: {e}")
            return False

    def get_conversation_history(self, sender_id, limit=10):
        """取得指定 sender 的對話記錄"""
        try:
            with self._get_connection() as conn:
                conv_hist_cur = conn.cursor()
                # 注意：原本您的程式碼這裡的 sender_id 應該是 user_id，此處保持與原程式碼一致的命名
                # 但通常對話歷史是針對某個用戶 (user_id)
                # 如果 sender_id 就是 user_id，那沒問題
                conv_hist_cur.execute(
                    """
                    SELECT TOP (?) sender_role, content
                    FROM conversations
                    WHERE sender_id = ?
                    ORDER BY timestamp DESC;
                    """,
                    (limit, sender_id)
                )
                # 從資料庫讀取是 DESC，但通常聊天室顯示是 ASC (舊的在上面)
                # 所以先 reverse
                messages = [
                    # 統一鍵名為 'role' 以符合 OpenAI 格式
                    {"role": sender_role, "content": content}
                    for sender_role, content in conv_hist_cur.fetchall()
                ]
                messages.reverse()  # 反轉順序，讓最新的訊息在最後
                return messages
        except pyodbc.Error as e:
            logger.exception(f"取得對話記錄失敗: {e}")
            return []

    def get_conversation_stats(self):
        """取得對話記錄統計資料"""
        try:
            with self._get_connection() as conn:
                conv_stats_cur = conn.cursor()
                conv_stats_cur.execute("SELECT COUNT(*) FROM conversations;")
                total_messages = conv_stats_cur.fetchone()[0]
                conv_stats_cur.execute(
                    "SELECT COUNT(DISTINCT sender_id) FROM conversations;"
                )
                unique_senders = conv_stats_cur.fetchone()[0]
                conv_stats_cur.execute(
                    """
                    SELECT COUNT(*) FROM conversations
                    WHERE timestamp >= DATEADD(day, -1, GETDATE());
                    """
                )
                last_24h = conv_stats_cur.fetchone()[0]
                conv_stats_cur.execute(
                    "SELECT sender_role, COUNT(*) FROM conversations "
                    "GROUP BY sender_role;"
                )
                role_counts = dict(conv_stats_cur.fetchall())
                return {
                    "total_messages": total_messages,
                    "unique_users": unique_senders,
                    "last_24h": last_24h,
                    "user_messages": role_counts.get("user", 0),
                    "assistant_messages": role_counts.get("assistant", 0),
                    "system_messages": role_counts.get("system", 0),
                    "other_messages": sum(
                        count for role, count in role_counts.items()
                        if role not in ["user", "assistant", "system"]
                    )
                }
        except pyodbc.Error as e:
            logger.exception(f"取得對話統計資料失敗: {e}")
            return {
                "total_messages": 0,
                "unique_users": 0,
                "last_24h": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0,
                "other_messages": 0,
            }

    def get_recent_conversations(self, limit=20):
        """取得最近的對話列表（依 sender_id，通常是 user_id）"""
        try:
            with self._get_connection() as conn:
                recent_conv_cur = conn.cursor()
                # 這裡的 sender_id 實際上是指 user_id
                sql_query = """
                    SELECT DISTINCT TOP (?)
                        c.sender_id,   -- 這其實是 user_id
                        p.language,
                        MAX(c.timestamp) as last_activity_ts   -- 改名以區分 last_message 內容
                    FROM conversations c
                    LEFT JOIN user_preferences p ON c.sender_id = p.user_id   -- 連接基於 sender_id = user_id
                    GROUP BY c.sender_id, p.language
                    ORDER BY last_activity_ts DESC;
                """
                recent_conv_cur.execute(sql_query, (limit,))
                results = []
                for user_id_val, language, timestamp_val in recent_conv_cur.fetchall():
                    # sender_id 即 user_id
                    recent_conv_cur.execute(
                        "SELECT COUNT(*) FROM conversations WHERE sender_id = ?;",
                        (user_id_val,)
                    )
                    message_count = recent_conv_cur.fetchone()[0]
                    recent_conv_cur.execute(
                        """
                        SELECT TOP 1 content FROM conversations
                        WHERE sender_id = ? AND sender_role = 'user'   -- 通常看 user 的最後一句話
                        ORDER BY timestamp DESC;
                        """,
                        (user_id_val,)
                    )
                    last_message_content = recent_conv_cur.fetchone()
                    results.append({
                        "user_id": user_id_val,  # 改名為 user_id
                        "language": language or "zh-Hant",  # 預設語言
                        "last_activity": timestamp_val,  # 直接使用 timestamp
                        "message_count": message_count,
                        "last_message": last_message_content[0] if last_message_content else "",
                    })
                return results
        except pyodbc.Error as e:
            logger.exception(f"取得最近對話失敗: {e}")
            return []

    def set_user_preference(self, user_id, language=None, role=None):
        """設定或更新使用者偏好與角色"""
        try:
            with self._get_connection() as conn:
                user_pref_set_cur = conn.cursor()
                user_pref_set_cur.execute(
                    "SELECT user_id FROM user_preferences WHERE user_id = ?;",
                    (user_id,)
                )
                user_exists = user_pref_set_cur.fetchone()

                if user_exists:
                    # 更新現有使用者
                    update_parts = []
                    params = []
                    if language is not None:
                        update_parts.append("language = ?")
                        params.append(language)
                    if role is not None:
                        update_parts.append("role = ?")
                        params.append(role)

                    # 如果沒有要更新的欄位，至少更新 last_active
                    if not update_parts:
                        user_pref_set_cur.execute(
                            "UPDATE user_preferences SET last_active = GETDATE()"
                            " WHERE user_id = ?;",
                            (user_id,)
                        )
                    else:
                        sql = (
                            "UPDATE user_preferences SET last_active = GETDATE(), "
                            + ", ".join(update_parts)
                            + " WHERE user_id = ?;"
                        )
                        params.append(user_id)
                        user_pref_set_cur.execute(sql, tuple(params))
                else:
                    # 新增使用者
                    user_pref_set_cur.execute(
                        """
                        INSERT INTO user_preferences
                            (user_id, language, role, last_active,
                             is_admin, responsible_area)
                        VALUES (?, ?, ?, GETDATE(), 0, NULL);
                        """,
                        (user_id, language or "zh-Hant", role or "user")
                    )
                conn.commit()
                return True
        except pyodbc.Error as e:
            logger.exception(f"設定使用者偏好失敗: {e}")
            return False

    # 加回 get_user_preference 方法
    def get_user_preference(self, user_id):
        """取得使用者偏好與角色"""
        try:
            with self._get_connection() as conn:
                user_pref_get_cur = conn.cursor()
                user_pref_get_cur.execute(
                    "SELECT language, role, is_admin, responsible_area "
                    "FROM user_preferences WHERE user_id = ?;",
                    (user_id,)
                )
                result = user_pref_get_cur.fetchone()
                if result:
                    return {
                        "language": result[0],
                        "role": result[1],
                        "is_admin": result[2],
                        "responsible_area": result[3]
                    }
                # 如果未找到則創建預設偏好
                logger.info(
                    f"User {user_id} not found in preferences, "
                    "creating with defaults."
                )
                self.set_user_preference(user_id)  # 這會創建預設的 user, zh-Hant
                return {
                    "language": "zh-Hant",
                    "role": "user",
                    "is_admin": False,
                    "responsible_area": None
                }
        except pyodbc.Error as e:
            logger.exception(f"取得使用者偏好失敗: {e}")
            # 發生錯誤時回傳一個安全的預設值
            return {
                "language": "zh-Hant",
                "role": "user",
                "is_admin": False,
                "responsible_area": None
            }

    def insert_alert_history(self, log_data: dict):
        """
        將單筆機台異常資料寫入 alert_history 表格。
        """
        sql = """
            INSERT INTO alert_history (
                error_id, equipment_id, alert_type,
                severity, created_time
            ) VALUES (?, ?, ?, ?, ?);
        """
        # 取得目前最大的 error_id，並加 1 作為新的 error_id
        sql_get_max = "SELECT ISNULL(MAX(error_id), 0) FROM alert_history;"

        conn = None
        try:
            # 用傳進來的 db 去拿連線
            conn = db._get_connection()
            cursor = conn.cursor()

            cursor.execute(sql_get_max)
            latest_error_id = cursor.fetchone()[0] + 1

            cursor.execute(sql,
                           latest_error_id,
                           log_data["equipment_id"],
                           log_data["alert_type"],
                           log_data["severity"],
                           log_data["created_time"]
                           )
            conn.commit()
            logger.info(f"成功寫入一筆異常紀錄，equipment_id: {log_data['equipment_id']}")
        except pyodbc.Error as ex:
            logger.error(f"資料庫寫入時發生錯誤: {ex}")
            if conn:
                conn.rollback()
                logger.warning("交易已回滾。")
            raise
        finally:
            if "cursor" in locals() and cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_subscribed_users(self, equipment_id: str):
        """取得訂閱指定設備的所有使用者 ID"""
        sql = (
            "SELECT user_id FROM user_equipment_subscriptions WHERE equipment_id = ?;"
        )
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (equipment_id,))
                return [row[0] for row in cursor.fetchall()]
        except pyodbc.Error as e:
            logger.error(f"取得設備 {equipment_id} 訂閱者失敗: {e}")
            return []


# 在測試環境下避免連線到實際資料庫
if os.environ.get("TESTING", "False").lower() != "true":
    db = Database()
else:
    db = None
