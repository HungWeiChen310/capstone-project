import logging
import os
import sqlite3

# 設定日誌紀錄器
logger = logging.getLogger(__name__)


class Database:
    """處理對話記錄與使用者偏好儲存的資料庫處理程序"""

    def __init__(self, db_path="data/conversations.db"):
        """初始化資料庫連線"""
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """如果資料表尚未存在，則建立必要的表格"""
        try:
            # 確保資料夾存在
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # 建立對話記錄表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # 建立使用者偏好表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT PRIMARY KEY,
                        language TEXT DEFAULT "zh-Hant",
                        last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # 建立設備表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS equipment (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_id TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        location TEXT,
                        status TEXT DEFAULT "normal",
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # 建立設備監測指標表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS equipment_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_id TEXT NOT NULL,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        threshold_min REAL,
                        threshold_max REAL,
                        unit TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    )
                    """
                )
                # 建立設備運轉紀錄表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS equipment_operation_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_id TEXT NOT NULL,
                        operation_type TEXT NOT NULL,
                        start_time DATETIME,
                        end_time DATETIME,
                        lot_id TEXT,
                        product_id TEXT,
                        yield_rate REAL,
                        operator_id TEXT,
                        notes TEXT,
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    )
                    """
                )
                # 建立警報記錄表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equipment_id TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        is_resolved INTEGER DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        resolved_at DATETIME,
                        resolved_by TEXT,
                        resolution_notes TEXT,
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    )
                    """
                )
                # 建立使用者訂閱設備表
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_equipment_subscriptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        equipment_id TEXT NOT NULL,
                        notification_level TEXT DEFAULT "all",
                        subscribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, equipment_id)
                    )
                    """
                )
                # 為 user_preferences 表新增 is_admin 與 responsible_area 欄位（若尚未存在）
                try:
                    cursor.execute("SELECT is_admin FROM user_preferences LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute(
                        "ALTER TABLE user_preferences ADD COLUMN is_admin INTEGER DEFAULT 0"
                    )
                try:
                    cursor.execute("SELECT responsible_area FROM user_preferences LIMIT 1")
                except sqlite3.OperationalError:
                    cursor.execute(
                        "ALTER TABLE user_preferences ADD COLUMN responsible_area TEXT"
                    )
                conn.commit()
                logger.info("資料庫初始化成功，包含設備監控相關資料表")
        except Exception as e:
            logger.exception(f"資料庫初始化失敗：{e}")
            raise

    def add_message(self, user_id, role, content):
        """加入一筆新的對話記錄"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
                    (user_id, role, content),
                )
                conn.commit()
                return True
        except Exception:
            logger.exception("新增對話記錄失敗")
            return False

    def get_conversation_history(self, user_id, limit=10):
        """取得指定使用者的對話記錄"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT role, content FROM conversations WHERE user_id = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit),
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 先檢查使用者是否已存在
                cursor.execute(
                    "SELECT user_id FROM user_preferences WHERE user_id = ?",
                    (user_id,),
                )
                user_exists = cursor.fetchone()
                if user_exists:
                    updates = []
                    params = []
                    if language:
                        updates.append("language = ?")
                        params.append(language)
                    if updates:
                        # 將 last_active 欄位更新為 CURRENT_TIMESTAMP
                        updates.append("last_active = CURRENT_TIMESTAMP")
                        query = (
                            "UPDATE user_preferences SET " + ", ".join(updates) +
                            " WHERE user_id = ?"
                        )
                        params.append(user_id)
                        cursor.execute(query, params)
                else:
                    fields = ["user_id"]
                    values = [user_id]
                    if language:
                        fields.append("language")
                        values.append(language)
                    fields_str = ", ".join(fields)
                    placeholders = ", ".join(["?"] * len(values))
                    query = f"INSERT INTO user_preferences ({fields_str}) VALUES ({placeholders})"
                    cursor.execute(query, tuple(values))
                conn.commit()
                return True
        except Exception:
            logger.exception("設定使用者偏好失敗")
            return False

    def get_user_preference(self, user_id):
        """取得使用者偏好"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT language FROM user_preferences WHERE user_id = ?",
                    (user_id,),
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conversations")
                total_messages = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                unique_users = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT COUNT(*) FROM conversations "
                    "WHERE timestamp > datetime('now', '-1 day')"
                )
                last_24h = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT role, COUNT(*) FROM conversations GROUP BY role"
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
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT c.user_id, p.language, MAX(c.timestamp) as last_message
                    FROM conversations c
                    LEFT JOIN user_preferences p ON c.user_id = p.user_id
                    GROUP BY c.user_id
                    ORDER BY last_message DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                results = []
                for user_id, language, timestamp in cursor.fetchall():
                    cursor.execute(
                        "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
                        (user_id,),
                    )
                    message_count = cursor.fetchone()[0]
                    cursor.execute(
                        """
                        SELECT content FROM conversations
                        WHERE user_id = ? AND role = 'user'
                        ORDER BY timestamp DESC LIMIT 1
                        """,
                        (user_id,),
                    )
                    last_message = cursor.fetchone()
                    results.append(
                        {
                            "user_id": user_id,
                            "language": language or "zh-Hant",
                            "last_activity": timestamp,
                            "message_count": message_count,
                            "last_message": last_message[0] if last_message else "",
                        }
                    )
                return results
        except Exception:
            logger.exception("取得最近對話失敗")
            return []


# 建立單例資料庫實例
db = Database()
