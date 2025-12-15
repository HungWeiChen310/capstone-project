import contextlib
import datetime
import logging
import os
import sys
import threading
from queue import Empty, Full, Queue

import pyodbc

if __package__ is None or __package__ == "":
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .config import Config  # noqa: E402


logger = logging.getLogger(__name__)


class SimpleConnectionPool:
    """A simple thread-safe connection pool for PyODBC connections."""

    def __init__(self, connect_func, max_size=10, timeout=5):
        self.connect_func = connect_func
        self.max_size = max_size
        self.timeout = timeout
        self.pool = Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = threading.Lock()

    def get_connection(self):
        """Retrieves a connection from the pool, creating a new one if
        necessary."""
        try:
            return self.pool.get(block=False)
        except Empty:
            with self._lock:
                if self._created_count < self.max_size:
                    try:
                        conn = self.connect_func()
                        self._created_count += 1
                        return conn
                    except Exception as e:
                        logger.error(
                            "Failed to create new DB connection: %s",
                            e,
                        )
                        raise

            # If pool is full, wait for a connection to be returned
            try:
                return self.pool.get(timeout=self.timeout)
            except Empty:
                logger.error("Database connection pool exhausted.")
                raise Exception("Database connection pool exhausted")

    def return_connection(self, conn):
        """Returns a connection to the pool, rolling back any pending
        transaction."""
        if conn is None:
            return

        try:
            # Ensure clean state for the next user
            conn.rollback()
            self.pool.put(conn, block=False)
        except (Full, pyodbc.Error, Exception):
            # If pool is full or connection is dead, close it
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created_count -= 1

    def close_all(self):
        """Closes all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get(block=False)
                conn.close()
            except (Empty, Exception):
                pass
        with self._lock:
            self._created_count = 0


class Database:
    """處理對話記錄與使用者偏好儲存的資料庫處理程序"""

    def __init__(self, server=None, database=None, user=None, password=None):
        """初始化資料庫連線"""
        resolved_server = server if server is not None else Config.DB_SERVER
        resolved_database = (
            database if database is not None else Config.DB_NAME
        )
        resolved_user = user if user is not None else Config.DB_USER
        resolved_password = (
            password if password is not None else Config.DB_PASSWORD
        )

        logger.info("初始化資料庫連線字串...")
        windows_login = (os.getenv("Windows_login") or "").lower()
        logger.info(
            "使用的伺服器: %s, 伺服器: %s, 資料庫: %s",
            windows_login,
            resolved_server,
            resolved_database,
        )

        if not bool(os.getenv("Windows_login")):
            self.connection_string = (
                f"DRIVER={{{Config.DB_ODBC_DRIVER}}};"
                f"SERVER={resolved_server};"
                f"DATABASE={resolved_database};"
                f"UID={resolved_user};"
                f"PWD={resolved_password};"
                "Trusted_Connection=no;"
                "Encrypt=no;"
                "TrustServerCertificate=yes;"
            )
        else:
            self.connection_string = (
                "DRIVER={ODBC Driver 17 for SQL Server};"
                f"SERVER={resolved_server};"
                f"DATABASE={resolved_database};"
                "Trusted_Connection=yes;"
            )

        # Initialize connection pool
        self.pool = SimpleConnectionPool(
            self._create_new_connection,
            max_size=10,
        )
        self._initialize_db()

    def _create_new_connection(self):
        """建立一個新的資料庫連線 (供 Pool 使用)"""
        return pyodbc.connect(self.connection_string)

    @contextlib.contextmanager
    def _get_connection(self):
        """
        取得資料庫連線的 Context Manager (使用 Connection Pool)。
        Usage:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                ...
        """
        conn = self.pool.get_connection()
        try:
            yield conn
            # Commit on success (imitating pyodbc context manager behavior)
            conn.commit()
        except Exception:
            # Rollback on error
            conn.rollback()
            raise
        finally:
            # Always return to pool
            self.pool.return_connection(conn)

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
                    [last_active] datetime2(2) NULL
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "user_preferences",
                    user_preferences_cols,
                )

                # 2. equipment
                equipment_cols = """
                    [id] INT NULL,
                    [equipment_id] NVARCHAR(255) NOT NULL PRIMARY KEY,
                    [name] NVARCHAR(255) NOT NULL,
                    [equipment_type] NVARCHAR(255) NULL,
                    [location] NVARCHAR(255) NULL,
                    [status] NVARCHAR(255) NOT NULL,
                    [last_updated] datetime2(2) NULL
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "equipment",
                    equipment_cols,
                )

                # 3. conversations
                conversations_cols = """
                    [sender_id] NVARCHAR(255) NOT NULL ,
                    [receiver_id] NVARCHAR(255) NOT NULL,
                    [sender_role] NVARCHAR(50) NOT NULL,
                    [content] NVARCHAR(MAX) NOT NULL,
                    [timestamp] datetime2(2) NOT NULL DEFAULT GETDATE()
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "conversations",
                    conversations_cols,
                )

                # 4. user_equipment_subscriptions
                user_equipment_subscriptions_cols = """
                    [subscription_id] INT IDENTITY(1,1) PRIMARY KEY,
                    [user_id] NVARCHAR(255) NOT NULL,
                    [notification_level] NVARCHAR(50) NULL,
                    [subscribed_at] datetime2(2) NULL DEFAULT GETDATE(),
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    CONSTRAINT FK_subscriptions_user
                        FOREIGN KEY (user_id)
                        REFERENCES user_preferences(user_id),
                    CONSTRAINT FK_subscriptions_equipment
                        FOREIGN KEY (equipment_id)
                        REFERENCES equipment(equipment_id),
                    CONSTRAINT UQ_user_equipment UNIQUE(user_id, equipment_id)
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "user_equipment_subscriptions",
                    user_equipment_subscriptions_cols,
                )

                # 5. alert_history
                alert_history_cols = """
                    [error_id] INT NOT NULL PRIMARY KEY,
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [detected_anomaly_type] NVARCHAR(255) NULL,
                    [severity_level] NVARCHAR(255) NULL,
                    [is_resolved] BIT NULL DEFAULT 0,
                    [created_time] datetime2(2) NULL,
                    [resolved_time] datetime2(2) NULL,
                    [resolved_by] NVARCHAR(255) NULL,
                    [resolution_notes] NVARCHAR(MAX) NULL
                """
                # 此表欄位原有message欄位，因純粹為重複其他欄位內容，不須保留，所以移除
                self._create_table_if_not_exists(
                    init_cur,
                    "alert_history",
                    alert_history_cols,
                )

                # 6. equipment_metrics
                equipment_metrics_cols = """
                    [id] INT NOT NULL PRIMARY KEY,
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [metric_type] NVARCHAR(255) NOT NULL,
                    [status] NVARCHAR(50) NULL,
                    [value] FLOAT NULL,
                    [threshold_min] FLOAT NULL,
                    [threshold_max] FLOAT NULL,
                    [unit] NVARCHAR(50) NULL,
                    [last_updated] datetime2(2) NULL DEFAULT GETDATE()
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "equipment_metrics",
                    equipment_metrics_cols,
                )

                # 7. equipment_metric_thresholds
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
                    equipment_metric_thresholds_cols,
                )

                # 8. error_logs
                error_logs_cols = """
                    [log_date] DATE NOT NULL,
                    [error_id] INT NOT NULL PRIMARY KEY,
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [deformation_mm] FLOAT NOT NULL,
                    [rpm] INT NOT NULL,
                    [event_time] datetime2(2) NOT NULL,
                    [detected_anomaly_type] NVARCHAR(MAX) NOT NULL,
                    [downtime_sec] INT NULL,
                    [resolved_time] datetime2(2) NULL,
                    [severity_level] NVARCHAR(MAX) NULL
                """
                self._create_table_if_not_exists(
                    init_cur, "error_logs", error_logs_cols
                )

                # 9. stats_abnormal_monthly
                stats_abnormal_monthly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [month] INT NOT NULL,
                    [detected_anomaly_type] NVARCHAR(255) NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_sec] INT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (
                        equipment_id, year, month, detected_anomaly_type
                    )
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "stats_abnormal_monthly",
                    stats_abnormal_monthly_cols,
                )

                # 10. stats_abnormal_quarterly
                stats_abnormal_quarterly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [quarter] INT NOT NULL,
                    [detected_anomaly_type] NVARCHAR(255) NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_sec] INT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (
                        equipment_id, year, quarter, detected_anomaly_type
                    )
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "stats_abnormal_quarterly",
                    stats_abnormal_quarterly_cols,
                )

                # 11. stats_abnormal_yearly
                stats_abnormal_yearly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [detected_anomaly_type] NVARCHAR(255) NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_sec] INT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, detected_anomaly_type)
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "stats_abnormal_yearly",
                    stats_abnormal_yearly_cols,
                )

                # 12. stats_operational_monthly
                stats_operational_monthly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [month] INT NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_sec] INT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, month)
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "stats_operational_monthly",
                    stats_operational_monthly_cols,
                )

                # 13. stats_operational_quarterly
                stats_operational_quarterly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL
                        FOREIGN KEY REFERENCES equipment(equipment_id),
                    [year] INT NOT NULL,
                    [quarter] INT NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_sec] INT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year, quarter)
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "stats_operational_quarterly",
                    stats_operational_quarterly_cols,
                )

                # 14. stats_operational_yearly
                stats_operational_yearly_cols = """
                    [equipment_id] NVARCHAR(255) NOT NULL,
                    [year] INT NOT NULL,
                    [total_operation_hrs] INT NULL,
                    [downtime_sec] INT NULL,
                    [downtime_rate_percent] NVARCHAR(255) NULL,
                    [notes] NVARCHAR(MAX) NULL,
                    PRIMARY KEY (equipment_id, year),
                    CONSTRAINT FK_stats_op_yearly_equip
                        FOREIGN KEY (equipment_id)
                        REFERENCES equipment(equipment_id)
                """
                self._create_table_if_not_exists(
                    init_cur,
                    "stats_operational_yearly",
                    stats_operational_yearly_cols,
                )

                # conn.commit() # Handled by _get_connection context manager
                logger.info("資料庫表格初始化/檢查完成 (已建立主鍵與外鍵約束)。")

            self._ensure_system_user()
        except pyodbc.Error as e:
            logger.exception(f"資料庫初始化期間發生 pyodbc 錯誤: {e}")
            raise
        except Exception as ex:
            logger.exception(f"資料庫初始化期間發生非預期錯誤: {ex}")
            raise

    def _create_table_if_not_exists(
        self,
        cursor,
        table_name,
        columns_definition,
    ):
        """通用方法，用於檢查並建立資料表"""
        check_table_sql = (
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?;"
        )
        cursor.execute(check_table_sql, (table_name,))
        if cursor.fetchone()[0] == 0:
            create_table_sql = (
                f"CREATE TABLE {table_name} ({columns_definition});"
            )
            cursor.execute(create_table_sql)
            logger.info(f"資料表 '{table_name}' 已建立。")
        else:
            logger.info(f"資料表 '{table_name}' 已存在，跳過建立。")

    def _ensure_system_user(self):
        """Ensure the system bot account exists for conversation history."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM user_preferences WHERE user_id = ?;",
                    ("bot",),
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(
                        """
                        INSERT INTO user_preferences
                            (user_id, language, role, is_admin,
                             responsible_area,
                             created_at, display_name, last_active)
                        VALUES (?, ?, ?, 0, NULL, GETDATE(), ?, GETDATE());
                        """,
                        ("bot", "zh-Hant", "assistant", "System Bot"),
                    )
                    # conn.commit() # Handled by _get_connection
        except pyodbc.Error as e:
            logger.exception(f"確保系統 bot 使用者存在時發生錯誤: {e}")

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
                    (sender_id, receiver_id, sender_role, content),
                )
                # conn.commit() # Handled by _get_connection
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
                    (limit, sender_id),
                )
                # 從資料庫讀取是 DESC，但通常聊天室顯示是 ASC (舊的在上面)
                # 所以先 reverse
                messages = [
                    # 統一鍵名為 'role' 以符合 Ollama 格式
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
                        count
                        for role, count in role_counts.items()
                        if role not in ("user", "assistant", "system")
                    ),
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
                        MAX(c.timestamp) as last_activity_ts
                        -- 改名以區分 last_message 內容
                    FROM conversations c
                    LEFT JOIN user_preferences p ON c.sender_id = p.user_id
                    -- 連接基於 sender_id = user_id
                    GROUP BY c.sender_id, p.language
                    ORDER BY last_activity_ts DESC;
                """
                recent_conv_cur.execute(sql_query, (limit,))
                results = []
                for user_id_val, language, timestamp_val in (
                    recent_conv_cur.fetchall()
                ):
                    # sender_id 即 user_id
                    recent_conv_cur.execute(
                        (
                            "SELECT COUNT(*) FROM conversations "
                            "WHERE sender_id = ?;"
                        ),
                        (user_id_val,),
                    )
                    message_count = recent_conv_cur.fetchone()[0]
                    recent_conv_cur.execute(
                        """
                        SELECT TOP 1 content FROM conversations
                        WHERE sender_id = ? AND sender_role = 'user'
                        -- 通常看 user 的最後一句話
                        ORDER BY timestamp DESC;
                        """,
                        (user_id_val,),
                    )
                    last_message_content = recent_conv_cur.fetchone()
                    results.append(
                        {
                            "user_id": user_id_val,  # 改名為 user_id
                            "language": language or "zh-Hant",  # 預設語言
                            "last_activity": timestamp_val,  # 直接使用 timestamp
                            "message_count": message_count,
                            "last_message": (
                                (
                                    last_message_content[0]
                                    if last_message_content
                                    else ""
                                )
                            ),
                        }
                    )
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
                    (user_id,),
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
                            (
                                "UPDATE user_preferences "
                                "SET last_active = GETDATE() "
                                "WHERE user_id = ?;"
                            ),
                            (user_id,),
                        )
                    else:
                        sql = (
                            "UPDATE user_preferences "
                            "SET last_active = GETDATE(), "
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
                        (user_id, language or "zh-Hant", role or "user"),
                    )
                # conn.commit() # Handled by _get_connection
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
                    (user_id,),
                )
                result = user_pref_get_cur.fetchone()
                if result:
                    return {
                        "language": result[0],
                        "role": result[1],
                        "is_admin": result[2],
                        "responsible_area": result[3],
                    }
                # 如果未找到則創建預設偏好
                logger.info(
                    "User %s not found in preferences, "
                    "creating with defaults.",
                    user_id,
                )
                self.set_user_preference(user_id)  # 這會創建預設的 user, zh-Hant
                return {
                    "language": "zh-Hant",
                    "role": "user",
                    "is_admin": False,
                    "responsible_area": None,
                }
        except pyodbc.Error as e:
            logger.exception(f"取得使用者偏好失敗: {e}")
            # 發生錯誤時回傳一個安全的預設值
            return {
                "language": "zh-Hant",
                "role": "user",
                "is_admin": False,
                "responsible_area": None,
            }

    def insert_alert_history(self, log_data: dict):
        """
        在單筆紀錄中，同時新增警報到 alert_history 和日誌到 error_logs。
        """
        sql_alert_history = """
            INSERT INTO alert_history (
                error_id, equipment_id, detected_anomaly_type,
                severity_level, created_time
            ) VALUES (?, ?, ?, ?, ?);
        """
        # 新增 error_log 寫入統計資料
        sql_error_log = """
            INSERT INTO error_logs (
                log_date, error_id, equipment_id, deformation_mm,
                rpm, event_time, detected_anomaly_type, severity_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        # 取得目前最大的 error_id，並加 1 作為新的 error_id
        sql_get_max = "SELECT ISNULL(MAX(error_id), 0) FROM alert_history;"

        try:
            # Modified to use context manager correctly
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 共用 error_id 和 event_time
                cursor.execute(sql_get_max)
                latest_error_id = cursor.fetchone()[0] + 1
                event_time = datetime.datetime.now()

                # 寫入 alert_history
                cursor.execute(
                    sql_alert_history,
                    latest_error_id,
                    log_data["equipment_id"],
                    log_data["detected_anomaly_type"],
                    log_data["severity_level"],
                    event_time,
                )

                # 寫入 error_logs
                cursor.execute(
                    sql_error_log,
                    event_time.date(),
                    latest_error_id,
                    log_data["equipment_id"],
                    log_data.get("deformation_mm", 0),
                    log_data.get("rpm", 30000),  # 預設30000
                    event_time,
                    log_data["detected_anomaly_type"],
                    log_data["severity_level"],
                )

                # conn.commit() # Handled by _get_connection context manager
                logger.info(
                    "成功寫入一筆異常紀錄，equipment_id: %s",
                    log_data["equipment_id"],
                )
                return {
                    "error_id": latest_error_id,
                    "created_time": event_time,
                }
        except pyodbc.Error as ex:
            logger.error(f"資料庫寫入時發生錯誤: {ex}")
            # rollback handled by _get_connection
            raise

    def get_alert_info(self, error_id: int, detected_anomaly_type: str):
        """用 error_id 跟 detected_anomaly_type 取得單筆警報的資訊"""
        sql = (
            "SELECT equipment_id, detected_anomaly_type FROM alert_history "
            "WHERE error_id = ? AND detected_anomaly_type =  ?;"
        )
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 執行SQL查詢 並將 error_id 跟 detected_anomaly_type 作為參數傳入
                cursor.execute(sql, error_id, detected_anomaly_type)
                row = cursor.fetchone()  # 從查詢結果中取出唯一一筆資料
                if row:  # 檢查是否有成功取回資料
                    return {"equipment_id": row[0]}
                return None
        except pyodbc.Error as e:
            logger.error(
                "查詢警報資訊 (error_id: %s), detected_anomaly_type: %s) 失敗: %s",
                error_id,
                detected_anomaly_type,
                e,
            )
            return None

    def resolve_alert_history(self, log_data: dict):
        """
        將指定的警報紀錄更新為已解決狀態
        """
        # 更新指定 alert_history 欄位內容，依照 error_id 跟 detected_anomaly_type
        # 跟 equipment_id 作為條件
        sql_alert_history = """
        UPDATE alert_history
           SET is_resolved = 1,
               resolved_time = GETDATE(),
               resolved_by = ?,
               resolution_notes = ?
        OUTPUT inserted.resolved_time
         WHERE error_id = ? AND detected_anomaly_type = ? AND equipment_id = ?
           AND (is_resolved = 0);
        """
        # 更新指定 error_logs 欄位內容
        sql_error_log = """
        UPDATE error_logs
           SET resolved_time = GETDATE(),
               downtime_sec = DATEDIFF(second, event_time, GETDATE())
         WHERE error_id = ?;
        """

        try:
            # Modified to use context manager correctly
            with self._get_connection() as conn:
                cursor = conn.cursor()
                notes = log_data.get("resolution_notes")
                if notes == "":
                    notes = None
                # 確保 log_data 包含必要欄位
                cursor.execute(
                    sql_alert_history,
                    log_data["resolved_by"],
                    notes,
                    log_data["error_id"],
                    log_data["detected_anomaly_type"],
                    log_data["equipment_id"],
                )

                newly_resolved_time = cursor.fetchone()  # 取得更新後 OUTPUT 的時間
                if newly_resolved_time:
                    # alert_history 成功更新，才更新 error_logs
                    cursor.execute(sql_error_log, log_data["error_id"])
                    # conn.commit() # Handled by _get_connection
                    logger.info(
                        "成功將 error_id: %s / detected_anomaly_type: %s / "
                        "equipment_id: %s 的警報標示為已解決。",
                        log_data["error_id"],
                        log_data["detected_anomaly_type"],
                        log_data["equipment_id"],
                    )
                    return newly_resolved_time[0]

                # 檢查這筆警報是否是已解決
                check_sql = (
                    "SELECT resolved_time FROM alert_history "
                    "WHERE error_id = ? AND detected_anomaly_type = ? "
                    "AND equipment_id = ? AND is_resolved = 1;"
                )
                cursor.execute(
                    check_sql,
                    log_data["error_id"],
                    log_data["detected_anomaly_type"],
                    log_data["equipment_id"],
                )
                already_resolved_time = cursor.fetchone()

                if already_resolved_time:
                    # 警報先前已是解決狀態
                    logger.info(
                        "嘗試解決的 error_id: %s / equipment_id: %s / "
                        "detected_anomaly_type: %s 先前已被解決。",
                        log_data["error_id"],
                        log_data["equipment_id"],
                        log_data["detected_anomaly_type"],
                    )
                    return (already_resolved_time[0], "already_resolved")

                # 資料庫不存在這筆 error_id
                logger.warning(
                    "嘗試更新警報，但找不到對應的 error_id: %s / "
                    "detected_anomaly_type: %s。和equipment_id: %s。",
                    log_data["error_id"],
                    log_data["detected_anomaly_type"],
                    log_data["equipment_id"],
                )
                return None

        except pyodbc.Error as ex:
            # 取得 error_id 或預設 N/A
            error_id_val = log_data.get("error_id", "N/A")
            detected_anomaly_type_val = log_data.get(
                "detected_anomaly_type",
                "N/A",
            )  # 取得 detected_anomaly_type 或預設 N/A
            equipment_id_val = log_data.get(
                "equipment_id",
                "N/A",
            )
            logger.error(
                "更新警報 (error_id: %s), detected_anomaly_type: %s, "
                "equipment_id: %s) 時發生資料庫錯誤: %s",
                error_id_val,
                detected_anomaly_type_val,
                equipment_id_val,
                ex,
            )
            # rollback handled by _get_connection
            raise

    def get_subscribed_users(self, equipment_id: str):
        """取得訂閱指定設備的所有使用者 ID"""
        sql = (
            "SELECT user_id FROM user_equipment_subscriptions "
            "WHERE equipment_id = ?;"
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
