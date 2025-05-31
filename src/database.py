import logging
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
            "DRIVER={ODBC Driver 17 for SQL Server};"  # Escaped braces for f-string
            f"SERVER={resolved_server};"
            f"DATABASE={resolved_database};"
            "Trusted_Connection=yes;"  # Keep trusted connection for now
        )
        try:
            self._initialize_db()
        except pyodbc.Error:
            # Error is logged in _initialize_db or _get_connection
            # Re-raise to signal that DB initialization failed critically
            raise RuntimeError("資料庫初始化失敗，應用程式無法啟動。")


    def _get_connection(self):
        """建立並回傳資料庫連線。若連線失敗則拋出例外。"""
        try:
            conn = pyodbc.connect(self.connection_string)
            return conn
        except pyodbc.Error as e:
            logger.critical(f"資料庫連線失敗: {self.connection_string}", exc_info=True)
            raise  # Re-raise the error to be caught by the caller


    def _create_indexes(self, cursor):
        """建立資料庫索引以提升查詢效能 (SQL Server specific)"""
        index_definitions = [
            {"name": "idx_conversations_sender_id", "table": "conversations", "columns": "(sender_id)"},
            {"name": "idx_conversations_timestamp", "table": "conversations", "columns": "(timestamp)"},
            {"name": "idx_conversations_receiver_id", "table": "conversations", "columns": "(receiver_id)"},
            {"name": "idx_equipment_name", "table": "equipment", "columns": "(name)"},
            {"name": "idx_equipment_type", "table": "equipment", "columns": "(type)"},
            {"name": "idx_equipment_status", "table": "equipment", "columns": "(status)"},
            {"name": "idx_abnormal_logs_equipment_id", "table": "abnormal_logs", "columns": "(equipment_id)"},
            {"name": "idx_abnormal_logs_event_date", "table": "abnormal_logs", "columns": "(event_date)"},
            {"name": "idx_abnormal_logs_abnormal_type", "table": "abnormal_logs", "columns": "(abnormal_type)"},
            {"name": "idx_alert_history_equipment_id", "table": "alert_history", "columns": "(equipment_id)"},
            {"name": "idx_alert_history_created_at", "table": "alert_history", "columns": "(created_at)"},
            {"name": "idx_alert_history_is_resolved", "table": "alert_history", "columns": "(is_resolved)"},
            {"name": "idx_alert_history_alert_type", "table": "alert_history", "columns": "(alert_type)"},
            {"name": "idx_alert_history_severity", "table": "alert_history", "columns": "(severity)"},
            {"name": "idx_user_equipment_subscriptions_user_id", "table": "user_equipment_subscriptions", "columns": "(user_id)"},
            {"name": "idx_user_equipment_subscriptions_equipment_id", "table": "user_equipment_subscriptions", "columns": "(equipment_id)"},
            {"name": "idx_equipment_metrics_equipment_id", "table": "equipment_metrics", "columns": "(equipment_id)"},
            {"name": "idx_equipment_metrics_metric_type", "table": "equipment_metrics", "columns": "(metric_type)"},
            {"name": "idx_equipment_metrics_timestamp", "table": "equipment_metrics", "columns": "(timestamp)"},
            {"name": "idx_eq_op_logs_equipment_id", "table": "equipment_operation_logs", "columns": "(equipment_id)"},
            {"name": "idx_eq_op_logs_start_time", "table": "equipment_operation_logs", "columns": "(start_time)"},
            {"name": "idx_eq_op_logs_end_time", "table": "equipment_operation_logs", "columns": "(end_time)"},
            {"name": "idx_op_stats_monthly_eq_year_month", "table": "operation_stats_monthly", "columns": "(equipment_id, year, month)"},
            {"name": "idx_op_stats_quarterly_eq_year_quarter", "table": "operation_stats_quarterly", "columns": "(equipment_id, year, quarter)"},
            {"name": "idx_op_stats_yearly_eq_year", "table": "operation_stats_yearly", "columns": "(equipment_id, year)"},
            {"name": "idx_fault_stats_monthly_eq_year_month_type", "table": "fault_stats_monthly", "columns": "(equipment_id, year, month, abnormal_type)"},
            {"name": "idx_fault_stats_quarterly_eq_year_quarter_type", "table": "fault_stats_quarterly", "columns": "(equipment_id, year, quarter, abnormal_type)"},
            {"name": "idx_fault_stats_yearly_eq_year_type", "table": "fault_stats_yearly", "columns": "(equipment_id, year, abnormal_type)"},
        ]

        for idx in index_definitions:
            check_sql = f"""
                IF NOT EXISTS (
                    SELECT * FROM sys.indexes
                    WHERE name = '{idx['name']}' AND object_id = OBJECT_ID('{idx['table']}')
                )
            """
            create_sql = f"CREATE INDEX {idx['name']} ON {idx['table']} {idx['columns']};"

            try:
                cursor.execute(check_sql + create_sql)
                logger.debug(f"成功執行索引語句 (或索引已存在): {idx['name']} ON {idx['table']}")
            except pyodbc.Error as e:
                logger.error(f"建立索引 {idx['name']} ON {idx['table']} 失敗: {e}", exc_info=True)


    def _initialize_db(self):
        """如果資料表尚未存在，則建立必要的表格與索引"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # --- Table Creation DDLs (Full list) ---
                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_preferences')
                    CREATE TABLE user_preferences (
                        user_id NVARCHAR(255) PRIMARY KEY,
                        language NVARCHAR(10) DEFAULT N'zh-Hant',
                        last_active DATETIME2 DEFAULT GETDATE(),
                        is_admin BIT DEFAULT 0,
                        responsible_area NVARCHAR(255),
                        role NVARCHAR(50) DEFAULT N'user'
                    )""")

                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'conversations')
                    CREATE TABLE conversations (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        sender_id NVARCHAR(255) NOT NULL,
                        receiver_id NVARCHAR(255) NOT NULL,
                        sender_role NVARCHAR(50) NOT NULL,
                        content NVARCHAR(MAX) NOT NULL,
                        timestamp DATETIME2 DEFAULT GETDATE(),
                        FOREIGN KEY (sender_id) REFERENCES user_preferences(user_id) ON DELETE CASCADE,
                        FOREIGN KEY (receiver_id) REFERENCES user_preferences(user_id) ON DELETE NO ACTION -- Or ON DELETE SET NULL / CASCADE depending on desired behavior
                    )""")

                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'equipment')
                    CREATE TABLE equipment (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL UNIQUE,
                        name NVARCHAR(255) NOT NULL,
                        type NVARCHAR(255) NOT NULL,
                        location NVARCHAR(255),
                        status NVARCHAR(255) DEFAULT N'normal',
                        last_updated DATETIME2 DEFAULT GETDATE()
                    )""")

                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'abnormal_logs')
                    CREATE TABLE abnormal_logs (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        event_date DATETIME2 NOT NULL,
                        equipment_id NVARCHAR(255) NOT NULL,
                        deformation_mm FLOAT,
                        rpm INT,
                        event_time TIME,
                        abnormal_type NVARCHAR(255),
                        downtime INT,
                        recovered_time TIME,
                        notes NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE
                    )""")

                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'alert_history')
                    CREATE TABLE alert_history (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        alert_type NVARCHAR(255) NOT NULL,
                        severity NVARCHAR(255) NOT NULL,
                        message NVARCHAR(MAX) NOT NULL,
                        is_resolved BIT DEFAULT 0,
                        created_at DATETIME2 DEFAULT GETDATE(),
                        resolved_at DATETIME2,
                        resolved_by NVARCHAR(255),
                        resolution_notes NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE
                    )""")

                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_equipment_subscriptions')
                    CREATE TABLE user_equipment_subscriptions (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        user_id NVARCHAR(255) NOT NULL,
                        equipment_id NVARCHAR(255) NOT NULL,
                        notification_level NVARCHAR(255) DEFAULT N'all', -- e.g., 'all', 'critical_only'
                        subscribed_at DATETIME2 DEFAULT GETDATE(),
                        CONSTRAINT UQ_user_equipment UNIQUE(user_id, equipment_id),
                        FOREIGN KEY (user_id) REFERENCES user_preferences(user_id) ON DELETE CASCADE,
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE
                    )""")

                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'equipment_metrics')
                    CREATE TABLE equipment_metrics (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        metric_type NVARCHAR(100) NOT NULL,
                        value FLOAT NOT NULL,
                        threshold_min FLOAT,
                        threshold_max FLOAT,
                        unit NVARCHAR(50),
                        timestamp DATETIME2 DEFAULT GETDATE(),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE
                    )""")

                cursor.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'equipment_operation_logs')
                    CREATE TABLE equipment_operation_logs (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        operation_type NVARCHAR(100) NOT NULL, -- e.g., 'processing_lot', 'maintenance', 'idle'
                        start_time DATETIME2 NOT NULL,
                        end_time DATETIME2,
                        lot_id NVARCHAR(100),
                        product_id NVARCHAR(100),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE
                    )""")

                # Statistical Tables
                stats_tables_ddl = {
                    "operation_stats_monthly": """(id INT IDENTITY(1,1) PRIMARY KEY, equipment_id NVARCHAR(255) NOT NULL, year INT, month INT, total_operation_time INT, total_downtime INT, downtime_rate FLOAT, description NVARCHAR(MAX), FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE, CONSTRAINT UQ_op_monthly UNIQUE(equipment_id, year, month))""",
                    "operation_stats_quarterly": """(id INT IDENTITY(1,1) PRIMARY KEY, equipment_id NVARCHAR(255) NOT NULL, year INT, quarter INT, total_operation_time INT, total_downtime INT, downtime_rate FLOAT, description NVARCHAR(MAX), FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE, CONSTRAINT UQ_op_quarterly UNIQUE(equipment_id, year, quarter))""",
                    "operation_stats_yearly": """(id INT IDENTITY(1,1) PRIMARY KEY, equipment_id NVARCHAR(255) NOT NULL, year INT, total_operation_time INT, total_downtime INT, downtime_rate FLOAT, description NVARCHAR(MAX), FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE, CONSTRAINT UQ_op_yearly UNIQUE(equipment_id, year))""",
                    "fault_stats_monthly": """(id INT IDENTITY(1,1) PRIMARY KEY, equipment_id NVARCHAR(255) NOT NULL, year INT, month INT, abnormal_type NVARCHAR(255), downtime INT, downtime_rate FLOAT, description NVARCHAR(MAX), FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE, CONSTRAINT UQ_fault_monthly UNIQUE(equipment_id, year, month, abnormal_type))""",
                    "fault_stats_quarterly": """(id INT IDENTITY(1,1) PRIMARY KEY, equipment_id NVARCHAR(255) NOT NULL, year INT, quarter INT, abnormal_type NVARCHAR(255), downtime INT, downtime_rate FLOAT, description NVARCHAR(MAX), FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE, CONSTRAINT UQ_fault_quarterly UNIQUE(equipment_id, year, quarter, abnormal_type))""",
                    "fault_stats_yearly": """(id INT IDENTITY(1,1) PRIMARY KEY, equipment_id NVARCHAR(255) NOT NULL, year INT, abnormal_type NVARCHAR(255), downtime INT, downtime_rate FLOAT, description NVARCHAR(MAX), FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id) ON DELETE CASCADE, CONSTRAINT UQ_fault_yearly UNIQUE(equipment_id, year, abnormal_type))"""
                }
                for table_name, ddl_columns in stats_tables_ddl.items():
                    cursor.execute(f"IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = '{table_name}') CREATE TABLE {table_name} {ddl_columns}")

                logger.info("所有資料表結構已確認/建立。")

                self._create_indexes(cursor) # Call to create indexes
                logger.info("所有資料庫索引已確認/建立。")

                conn.commit()
                logger.info("資料庫初始化成功 (包含資料表與索引)。")
        except pyodbc.Error as e:
            logger.critical("資料庫初始化過程中發生嚴重錯誤。", exc_info=True)
            raise # Re-raise after logging


    def add_message(self, sender_id, receiver_id, sender_role, content):
        """加入一筆新的對話記錄（包含發送者角色）"""
        try:
            with self._get_connection() as conn:
                conv_add_cur = conn.cursor()
                conv_add_cur.execute(
                    """
                    INSERT INTO conversations
                        (sender_id, receiver_id, sender_role, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (sender_id, receiver_id, sender_role, content)
                )
                conn.commit()
                logger.debug(f"成功新增訊息，由 {sender_id} 發送給 {receiver_id}")
                return True
        except pyodbc.Error as e:
            logger.exception(f"為發送者 {sender_id} 新增對話記錄到資料庫時失敗: {e}")
            # Attempt to rollback if connection is still alive and in a transaction
            try:
                if conn and conn.autocommit is False: # Check if in transaction
                    conn.rollback()
                    logger.info(f"為發送者 {sender_id} 新增訊息失敗後，已執行 Rollback。")
            except pyodbc.Error as rb_err:
                logger.error(f"為發送者 {sender_id} 新增訊息失敗後，Rollback 也失敗: {rb_err}")
            return False

    def get_conversation_history(self, sender_id, limit=10):
        """取得指定 sender 的對話記錄"""
        try:
            with self._get_connection() as conn:
                conv_hist_cur = conn.cursor()
                conv_hist_cur.execute(
                    """
                    SELECT TOP (?) sender_role, content
                    FROM conversations
                    WHERE sender_id = ?
                    ORDER BY timestamp DESC
                    """,
                    (limit, sender_id)
                )
                messages = [
                    {"sender_role": sender_role, "content": content}
                    for sender_role, content in conv_hist_cur.fetchall()
                ]
                messages.reverse() # chronological order
                logger.debug(f"成功為 {sender_id} 取得 {len(messages)} 筆對話記錄。")
                return messages
        except pyodbc.Error as e:
            logger.exception(f"為發送者 {sender_id} 取得對話記錄時發生資料庫錯誤: {e}")
            return []
        except Exception as e_gen: # Catch other potential errors
            logger.exception(f"為發送者 {sender_id} 取得對話記錄時發生非預期錯誤: {e_gen}")
            return []


    def get_conversation_stats(self):
        """取得對話記錄統計資料"""
        try:
            with self._get_connection() as conn:
                conv_stats_cur = conn.cursor()
                conv_stats_cur.execute("SELECT COUNT(*) FROM conversations")
                # Safely get results, defaulting to 0 if query fails or returns None
                row = conv_stats_cur.fetchone()
                total_messages = row[0] if row else 0

                conv_stats_cur.execute("SELECT COUNT(DISTINCT sender_id) FROM conversations")
                row = conv_stats_cur.fetchone()
                unique_senders = row[0] if row else 0

                conv_stats_cur.execute("SELECT COUNT(*) FROM conversations WHERE timestamp >= DATEADD(day, -1, GETDATE())")
                row = conv_stats_cur.fetchone()
                last_24h = row[0] if row else 0

                conv_stats_cur.execute("SELECT sender_role, COUNT(*) FROM conversations GROUP BY sender_role")
                role_counts_raw = conv_stats_cur.fetchall()
                role_counts = {role: count for role, count in role_counts_raw} if role_counts_raw else {}

                stats = {
                    "total_messages": total_messages or 0,
                    "unique_senders": unique_senders or 0,
                    "last_24h": last_24h or 0,
                    "user_messages": role_counts.get("user", 0),
                    "assistant_messages": role_counts.get("assistant", 0),
                    "system_messages": role_counts.get("system", 0),
                    "other_messages": sum(count for role, count in role_counts.items() if role not in ["user", "assistant", "system"])
                }
                logger.debug(f"成功取得對話統計資料: {stats}")
                return stats
        except pyodbc.Error as e:
            logger.exception(f"取得對話統計資料時發生資料庫錯誤: {e}")
            return {
                "total_messages": 0,
                "unique_senders": 0,
                "last_24h": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0,
                "other_messages": 0,
            }

    def get_recent_conversations(self, limit=20):
        """取得最近的對話列表（依 sender_id）"""
        try:
            with self._get_connection() as conn:
                recent_conv_cur = conn.cursor()
                recent_conv_cur.execute(
                    """
                    SELECT DISTINCT TOP (?)
                        c.sender_id,
                        p.language,
                        MAX(c.timestamp) as last_message
                    FROM conversations c
                    LEFT JOIN user_preferences p ON c.sender_id = p.user_id
                    GROUP BY c.sender_id, p.language
                    ORDER BY last_message DESC
                    """,
                    (limit,)
                )
                results = []
                fetched_conversations = recent_conv_cur.fetchall()
                results = []
                if fetched_conversations:
                    for sender_id_db, language, timestamp in fetched_conversations:
                        message_count_row = recent_conv_cur.execute(
                            "SELECT COUNT(*) FROM conversations WHERE sender_id = ?", (sender_id_db,)
                        ).fetchone()
                        message_count = message_count_row[0] if message_count_row else 0

                        last_message_row = recent_conv_cur.execute(
                            "SELECT TOP 1 content FROM conversations WHERE sender_id = ? AND sender_role = 'user' ORDER BY timestamp DESC",
                            (sender_id_db,)
                        ).fetchone()
                        last_message_content = last_message_row[0] if last_message_row else ""

                        results.append({
                            "sender_id": sender_id_db,
                            "language": language or "zh-Hant",
                            "last_activity": timestamp,
                            "message_count": message_count,
                            "last_message": last_message_content,
                        })
                logger.debug(f"成功取得 {len(results)} 筆最近對話。")
                return results
        except pyodbc.Error as e:
            logger.exception(f"取得最近對話列表時發生資料庫錯誤: {e}")
            return []
        except Exception as e_gen: # Catch other potential errors
            logger.exception(f"取得最近對話列表時發生非預期錯誤: {e_gen}")
            return []


    def set_user_preference(self, user_id, language=None, role=None):
        """設定或更新使用者偏好與角色"""
        try:
            with self._get_connection() as conn:
                user_pref_set_cur = conn.cursor()
                user_pref_set_cur.execute(
                    "SELECT user_id FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                user_exists = user_pref_set_cur.fetchone()
                if user_exists:
                    sql = "UPDATE user_preferences SET last_active = GETDATE()"
                    params = []
                    if language:
                        sql += ", language = ?"
                        params.append(language)
                    if role:
                        sql += ", role = ?"
                        params.append(role)
                    sql += " WHERE user_id = ?"
                    params.append(user_id)
                    user_pref_set_cur.execute(sql, tuple(params))
                else:
                    user_pref_set_cur.execute(
                        """
                        INSERT INTO user_preferences (user_id, language, role)
                        VALUES (?, ?, ?)
                        """,
                        (user_id, language or "zh-Hant", role or "user")
                    )
                conn.commit()
                logger.info(f"成功為使用者 {user_id} 設定偏好 (語言: {language}, 角色: {role})。")
                return True
        except pyodbc.Error as e:
            logger.exception(f"為使用者 {user_id} 設定偏好時發生資料庫錯誤: {e}")
            try:
                if conn and conn.autocommit is False:
                    conn.rollback()
                    logger.info(f"為使用者 {user_id} 設定偏好失敗後，已執行 Rollback。")
            except pyodbc.Error as rb_err:
                logger.error(f"為使用者 {user_id} 設定偏好失敗後，Rollback 也失敗: {rb_err}")
            return False
        except Exception as e_gen: # Catch other potential errors
            logger.exception(f"為使用者 {user_id} 設定偏好時發生非預期錯誤: {e_gen}")
            return False


    def get_user_preference(self, user_id):
        """取得使用者偏好與角色"""
        try:
            with self._get_connection() as conn:
                user_pref_get_cur = conn.cursor()
                user_pref_get_cur.execute(
                    "SELECT language, role FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                result = user_pref_get_cur.fetchone()
                if result:
                    logger.debug(f"成功取得使用者 {user_id} 的偏好設定。")
                    return {"language": result[0], "role": result[1]}
                else:
                    logger.info(f"找不到使用者 {user_id} 的偏好設定，將嘗試創建預設值。")
                    # Attempt to set default preferences.
                    # This might fail if DB is read-only or user_id is problematic for insert.
                    if self.set_user_preference(user_id): # Default language and role are set within set_user_preference
                        logger.info(f"已為使用者 {user_id} 創建預設偏好設定。")
                        return {"language": "zh-Hant", "role": "user"} # Return what was attempted to be set
                    else:
                        logger.error(f"為使用者 {user_id} 創建預設偏好設定失敗。回傳系統預設值。")
                        # Fallback to system default if creation fails
                        return {"language": "zh-Hant", "role": "user"}
        except pyodbc.Error as e:
            logger.exception(f"為使用者 {user_id} 取得偏好設定時發生資料庫錯誤: {e}")
            return {"language": "zh-Hant", "role": "user"} # Fallback to system default
        except Exception as e_gen: # Catch other potential errors
            logger.exception(f"為使用者 {user_id} 取得偏好設定時發生非預期錯誤: {e_gen}")
            return {"language": "zh-Hant", "role": "user"}


db = Database()
