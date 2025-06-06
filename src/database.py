import logging
import pyodbc
from config import Config

logger = logging.getLogger(__name__)


class Database:
    """處理對話記錄與使用者偏好儲存的資料庫處理程序"""

    def __init__(self, server=None, database=None):
        """初始化資料庫連線"""
        # 修改回從 Config 讀取預設值
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
        """如果資料表尚未存在，則建立必要的表格"""
        try:
            with self._get_connection() as conn:
                init_cur = conn.cursor()

                # --- 1. 建立無依賴的基礎資料表 ---

                # 建立使用者偏好表 (被 conversations 和 user_equipment_subscriptions 參考)
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'user_preferences'
                    )
                    CREATE TABLE user_preferences (
                        user_id NVARCHAR(255) PRIMARY KEY,
                        language NVARCHAR(10) DEFAULT N'zh-Hant',
                        last_active DATETIME2 DEFAULT GETDATE(),
                        is_admin BIT DEFAULT 0,
                        responsible_area NVARCHAR(255),
                        role NVARCHAR(50) DEFAULT N'user'
                    );
                """)

                # 建立設備表 (被多個資料表參考)
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'equipment'
                    )
                    CREATE TABLE equipment (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL UNIQUE,
                        name NVARCHAR(255) NOT NULL,
                        type NVARCHAR(255) NOT NULL,
                        location NVARCHAR(255),
                        status NVARCHAR(255) DEFAULT N'normal',
                        last_updated DATETIME2 DEFAULT GETDATE()
                    );
                """)

                # --- 2. 建立依賴基礎資料表的從屬資料表 ---

                # 建立對話記錄表 (依賴 user_preferences)
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'conversations'
                    )
                    CREATE TABLE conversations (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        sender_id NVARCHAR(255) NOT NULL,
                        receiver_id NVARCHAR(255) NOT NULL,
                        sender_role NVARCHAR(50) NOT NULL,
                        content NVARCHAR(MAX) NOT NULL,
                        timestamp DATETIME2 DEFAULT GETDATE(),
                        FOREIGN KEY (sender_id) REFERENCES user_preferences(user_id),
                        FOREIGN KEY (receiver_id) REFERENCES user_preferences(user_id)
                    );
                """)

                # 建立異常紀錄表 (依賴 equipment)
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'abnormal_logs'
                    )
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
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 建立警報記錄表 (依賴 equipment)
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'alert_history'
                    )
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
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 使用者訂閱設備表 (邏輯上依賴 user_preferences 和 equipment)
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables
                        WHERE name = 'user_equipment_subscriptions'
                    )
                    CREATE TABLE user_equipment_subscriptions (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        user_id NVARCHAR(255) NOT NULL,
                        equipment_id NVARCHAR(255) NOT NULL,
                        notification_level NVARCHAR(255) DEFAULT N'all',
                        subscribed_at DATETIME2 DEFAULT GETDATE(),
                        CONSTRAINT UQ_user_equipment UNIQUE(user_id, equipment_id),
                        FOREIGN KEY (user_id) REFERENCES user_preferences(user_id),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 建立設備指標表 (依賴 equipment)
                init_cur.execute("""
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
                    );
                """)

                # 建立設備運作記錄表 (依賴 equipment)
                init_cur.execute("""
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'equipment_operation_logs')
                    CREATE TABLE equipment_operation_logs (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        operation_type NVARCHAR(100) NOT NULL,
                        start_time DATETIME2 NOT NULL,
                        end_time DATETIME2,
                        lot_id NVARCHAR(100),
                        product_id NVARCHAR(100),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # --- 3. 建立統計相關資料表 (皆依賴 equipment) ---

                # 運作統計（月）
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'operation_stats_monthly'
                    )
                    CREATE TABLE operation_stats_monthly (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        year INT,
                        month INT,
                        total_operation_time INT,
                        total_downtime INT,
                        downtime_rate FLOAT,
                        description NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 運作統計（季）
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'operation_stats_quarterly'
                    )
                    CREATE TABLE operation_stats_quarterly (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        year INT,
                        quarter INT,
                        total_operation_time INT,
                        total_downtime INT,
                        downtime_rate FLOAT,
                        description NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 運作統計（年）
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'operation_stats_yearly'
                    )
                    CREATE TABLE operation_stats_yearly (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        year INT,
                        total_operation_time INT,
                        total_downtime INT,
                        downtime_rate FLOAT,
                        description NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 各異常統計（月）
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'fault_stats_monthly'
                    )
                    CREATE TABLE fault_stats_monthly (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        year INT,
                        month INT,
                        abnormal_type NVARCHAR(255),
                        downtime INT,
                        downtime_rate FLOAT,
                        description NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 各異常統計（季）
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'fault_stats_quarterly'
                    )
                    CREATE TABLE fault_stats_quarterly (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        year INT,
                        quarter INT,
                        abnormal_type NVARCHAR(255),
                        downtime INT,
                        downtime_rate FLOAT,
                        description NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                # 各異常統計（年）
                init_cur.execute("""
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'fault_stats_yearly'
                    )
                    CREATE TABLE fault_stats_yearly (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        equipment_id NVARCHAR(255) NOT NULL,
                        year INT,
                        abnormal_type NVARCHAR(255),
                        downtime INT,
                        downtime_rate FLOAT,
                        description NVARCHAR(MAX),
                        FOREIGN KEY (equipment_id) REFERENCES equipment(equipment_id)
                    );
                """)

                conn.commit()
                logger.info("資料庫初始化成功，包含所有自訂資料表")
        except pyodbc.Error as exc:
            logger.exception(f"資料庫初始化失敗：{exc}")
            raise

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
                unique_senders = conv_stats_cur.fetchone()[0]  # 這裡的 sender_id 應該是指 user_id
                conv_stats_cur.execute(
                    """
                    SELECT COUNT(*) FROM conversations
                    WHERE timestamp >= DATEADD(day, -1, GETDATE());
                    """
                )
                last_24h = conv_stats_cur.fetchone()[0]
                conv_stats_cur.execute(
                    "SELECT sender_role, COUNT(*) FROM conversations GROUP BY sender_role;"
                )
                role_counts = dict(conv_stats_cur.fetchall())
                return {
                    "total_messages": total_messages,
                    "unique_users": unique_senders,  # 改名為 unique_users 更清晰
                    "last_24h": last_24h,
                    "user_messages": role_counts.get("user", 0),
                    "assistant_messages": role_counts.get("assistant", 0),
                    # 如果您有 system role 的訊息
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

    # 加回 set_user_preference 方法
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
                            "UPDATE user_preferences SET last_active = GETDATE() "
                            "WHERE user_id = ?;",
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
                            (user_id, language, role, last_active, is_admin, responsible_area)
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
                logger.info(f"User {user_id} not found in preferences, creating with defaults.")
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


db = Database()
