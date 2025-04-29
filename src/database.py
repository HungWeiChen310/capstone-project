import logging
import os
import pyodbc

# 設定日誌紀錄器
logger = logging.getLogger(__name__)


class Database:
    """處理對話記錄與使用者偏好儲存的資料庫處理程序"""

    def __init__(self, server="localhost", database="conversations"):
        """初始化資料庫連線"""
        self.connection_string = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
        )
        self._initialize_db()

    def _get_connection(self):
        """建立資料庫連線"""
        return pyodbc.connect(self.connection_string)

    def _initialize_db(self):
        """如果資料表尚未存在，則建立必要的表格"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 建立對話記錄表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'conversations')
                    CREATE TABLE conversations (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        user_id NVARCHAR(255) NOT NULL,
                        role NVARCHAR(50) NOT NULL,
                        content NVARCHAR(MAX) NOT NULL,
                        timestamp DATETIME2 DEFAULT GETDATE()
                    )
                    """
                )
                # 建立使用者偏好表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_preferences')
                    CREATE TABLE user_preferences (
                        user_id NVARCHAR(255) PRIMARY KEY,
                        language NVARCHAR(10) DEFAULT N'zh-Hant',
                        last_active DATETIME2 DEFAULT GETDATE(),
                        is_admin BIT DEFAULT 0,
                        responsible_area NVARCHAR(255)
                    )
                    """
                )
                # 建立設備表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'equipment')
                    CREATE TABLE equipment (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL UNIQUE,
                        name NVARCHAR(255) NOT NULL,
                        type NVARCHAR(100) NOT NULL,
                        location NVARCHAR(255),
                        status NVARCHAR(50) DEFAULT N'normal',
                        last_updated DATETIME2 DEFAULT GETDATE()
                    )
                    """
                )
                # 建立設備監測指標表
                cursor.execute(
                    """
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
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    )
                    """
                )
                # 建立設備運轉紀錄表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'equipment_operation_logs')
                    CREATE TABLE equipment_operation_logs (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        operation_type NVARCHAR(100) NOT NULL,
                        start_time DATETIME2,
                        end_time DATETIME2,
                        lot_id NVARCHAR(255),
                        product_id NVARCHAR(255),
                        yield_rate FLOAT,
                        operator_id NVARCHAR(255),
                        notes NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    )
                    """
                )
                # 建立警報記錄表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'alert_history')
                    CREATE TABLE alert_history (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        alert_type NVARCHAR(100) NOT NULL,
                        severity NVARCHAR(50) NOT NULL,
                        message NVARCHAR(MAX) NOT NULL,
                        is_resolved BIT DEFAULT 0,
                        created_at DATETIME2 DEFAULT GETDATE(),
                        resolved_at DATETIME2,
                        resolved_by NVARCHAR(255),
                        resolution_notes NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    )
                    """
                )
                # 建立使用者訂閱設備表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'user_equipment_subscriptions')
                    CREATE TABLE user_equipment_subscriptions (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        user_id NVARCHAR(255) NOT NULL,
                        equipment_id NVARCHAR(255) NOT NULL,
                        notification_level NVARCHAR(50) DEFAULT N'all',
                        subscribed_at DATETIME2 DEFAULT GETDATE(),
                        CONSTRAINT UQ_user_equipment UNIQUE(user_id, equipment_id)
                    )
                    """
                )
                conn.commit()
                logger.info("資料庫初始化成功，包含設備監控相關資料表")
        except Exception as e:
            logger.exception(f"資料庫初始化失敗：{e}")
            raise

    def add_message(self, user_id, role, content):
        """加入一筆新的對話記錄"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
                    (user_id, role, content)
                )
                conn.commit()
                return True
        except Exception:
            logger.exception("新增對話記錄失敗")
            return False

    def get_conversation_history(self, user_id, limit=10):
        """取得指定使用者的對話記錄"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT TOP (?) role, content 
                    FROM conversations 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC
                    """,
                    (limit, user_id)
                )
                messages = [
                    {"role": role, "content": content}
                    for role, content in cursor.fetchall()
                ]
                messages.reverse()  # 反轉順序讓最舊的訊息排在前面
                return messages
        except Exception:
            logger.exception("取得對話記錄失敗")
            return []

    def set_user_preference(self, user_id, language=None):
        """設定或更新使用者偏好"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # 先檢查使用者是否已存在
                cursor.execute(
                    "SELECT user_id FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                user_exists = cursor.fetchone()
                if user_exists:
                    if language:
                        cursor.execute(
                            """
                            UPDATE user_preferences 
                            SET language = ?, last_active = GETDATE()
                            WHERE user_id = ?
                            """,
                            (language, user_id)
                        )
                else:
                    cursor.execute(
                        """
                        INSERT INTO user_preferences (user_id, language)
                        VALUES (?, ?)
                        """,
                        (user_id, language or 'zh-Hant')
                    )
                conn.commit()
                return True
        except Exception:
            logger.exception("設定使用者偏好失敗")
            return False

    def get_user_preference(self, user_id):
        """取得使用者偏好"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT language FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                if result:
                    return {"language": result[0]}
                else:
                    # 如未找到則創建預設偏好
                    self.set_user_preference(user_id)
                    return {"language": "zh-Hant"}
        except Exception:
            logger.exception("取得使用者偏好失敗")
            return {"language": "zh-Hant"}

    def get_conversation_stats(self):
        """取得對話記錄統計資料"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conversations")
                total_messages = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                unique_users = cursor.fetchone()[0]
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM conversations 
                    WHERE timestamp >= DATEADD(day, -1, GETDATE())
                    """
                )
                last_24h = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT role, COUNT(*) as count FROM conversations GROUP BY role"
                )
                role_counts = dict(cursor.fetchall())
                return {
                    "total_messages": total_messages,
                    "unique_users": unique_users,
                    "last_24h": last_24h,
                    "user_messages": role_counts.get("user", 0),
                    "assistant_messages": role_counts.get("assistant", 0),
                    "system_messages": role_counts.get("system", 0),
                }
        except Exception:
            logger.exception("取得對話統計資料失敗")
            return {
                "total_messages": 0,
                "unique_users": 0,
                "last_24h": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0,
            }

    def get_recent_conversations(self, limit=20):
        """取得最近的對話列表"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT TOP (?)
                        c.user_id,
                        p.language,
                        MAX(c.timestamp) as last_message
                    FROM conversations c
                    LEFT JOIN user_preferences p ON c.user_id = p.user_id
                    GROUP BY c.user_id, p.language
                    ORDER BY last_message DESC
                    """,
                    (limit,)
                )
                results = []
                for user_id, language, timestamp in cursor.fetchall():
                    cursor.execute(
                        "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
                        (user_id,)
                    )
                    message_count = cursor.fetchone()[0]
                    cursor.execute(
                        """
                        SELECT TOP 1 content FROM conversations
                        WHERE user_id = ? AND role = 'user'
                        ORDER BY timestamp DESC
                        """,
                        (user_id,)
                    )
                    last_message = cursor.fetchone()
                    results.append({
                        "user_id": user_id,
                        "language": language or "zh-Hant",
                        "last_activity": timestamp,
                        "message_count": message_count,
                        "last_message": last_message[0] if last_message else "",
                    })
                return results
        except Exception:
            logger.exception("取得最近對話失敗")
            return []


# 建立單例資料庫實例
db = Database()
