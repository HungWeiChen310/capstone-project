import datetime
import json
import logging
import os
import pyodbc  # 引入 pyodbc
from database import db  # 從 database 模組匯入 db 實例

logger = logging.getLogger(__name__)


class Analytics:
    """分析模組，用於追蹤與分析使用者行為與系統使用狀況 (使用 MS SQL Server)"""

    def __init__(self):
        """初始化分析模組，並確保分析相關的資料表已建立"""
        # stats_dir 和 self.stats_path 仍然用於匯出 JSON 統計檔案，可以保留
        # 可以基於 db 的某個屬性或固定路徑
        stats_dir = os.path.join(os.path.dirname(db.connection_string), "stats")
        if "SERVER" not in db.connection_string:  # 簡易判斷是否為有效連線字串
            # 如果 db 連線字串不尋常，給一個預設
            stats_dir = os.path.join(os.getcwd(), "data", "stats")

        os.makedirs(stats_dir, exist_ok=True)
        self.stats_path = os.path.join(stats_dir, "usage_stats.json")
        self._initialize_analytics_tables()

    def _initialize_analytics_tables(self):
        """初始化分析用的資料表 (MS SQL Server 版本)"""
        try:
            with db._get_connection() as conn:  # 使用全域 db 實例的連線
                cursor = conn.cursor()
                # analytics_events 表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'analytics_events')
                    CREATE TABLE analytics_events (
                        id INT IDENTITY(1,1) PRIMARY KEY,
                        event_type NVARCHAR(255) NOT NULL,
                        user_id NVARCHAR(255),
                        timestamp DATETIME2 DEFAULT GETDATE(),
                        metadata NVARCHAR(MAX) NULL  -- 允許 NULL
                    );
                    """
                )
                # daily_stats 表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'daily_stats')
                    CREATE TABLE daily_stats (
                        date DATE PRIMARY KEY,  -- 使用 DATE 型別
                        total_messages INT DEFAULT 0,
                        unique_users INT DEFAULT 0,
                        avg_response_time FLOAT DEFAULT 0,  -- 使用 FLOAT
                        data NVARCHAR(MAX) NULL  -- 儲存 JSON 字串
                    );
                    """
                )
                # keyword_stats 表
                cursor.execute(
                    """
                    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'keyword_stats')
                    CREATE TABLE keyword_stats (
                        keyword NVARCHAR(255) PRIMARY KEY,
                        count INT DEFAULT 0,
                        last_used DATETIME2 DEFAULT GETDATE()
                    );
                    """
                )
                conn.commit()
                logger.info("分析相關資料表已在 MS SQL Server 中初始化或確認存在。")
        except pyodbc.Error as e:
            logger.exception(f"初始化 MS SQL Server 分析表失敗: {e}")
        except Exception as e:
            logger.exception(f"初始化 MS SQL Server 分析表時發生未知錯誤: {e}")

    def track_event(self, event_type, user_id=None, metadata=None):
        """
        記錄一個事件

        參數:
            event_type: 事件類型 (例如: message, powerbi_view, language_change)
            user_id: 使用者 ID
            metadata: 額外資訊，會以 JSON 格式儲存
        """
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            with db._get_connection() as conn:  # 使用全域 db 實例的連線
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO analytics_events (event_type, user_id, metadata) "
                    "VALUES (?, ?, ?);",
                    (event_type, user_id, metadata_json),
                )
                conn.commit()
            return True
        except pyodbc.Error as e:
            logger.exception(f"記錄事件 [%s] 到 MS SQL Server 失敗: {e}", event_type)
            return False
        except Exception as e:
            logger.exception(f"記錄事件 [%s] 時發生未知錯誤: {e}", event_type)
            return False

    def track_keywords(self, text, increment=1):
        """
        追蹤常用關鍵字

        參數:
            text: 要分析的文字
            increment: 增加的計數
        """
        if not text or not isinstance(text, str):
            return False
        keywords = [word for word in text.lower().split() if len(word) > 1]
        try:
            with db._get_connection() as conn:  # 使用全域 db 實例的連線
                cursor = conn.cursor()
                for keyword in keywords:
                    # 檢查關鍵字是否存在
                    cursor.execute(
                        "SELECT count FROM keyword_stats WHERE keyword = ?;", (keyword,)
                    )
                    result = cursor.fetchone()
                    if result:
                        cursor.execute(
                            "UPDATE keyword_stats SET count = count + ?, "
                            "last_used = GETDATE() WHERE keyword = ?;",
                            (increment, keyword),
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO keyword_stats (keyword, count, last_used) "
                            "VALUES (?, ?, GETDATE());",
                            (keyword, increment),
                        )
                conn.commit()
            return True
        except pyodbc.Error as e:
            logger.exception(f"追蹤關鍵字到 MS SQL Server 失敗: {e}")
            return False
        except Exception as e:
            logger.exception(f"追蹤關鍵字時發生未知錯誤: {e}")
            return False

    def get_top_keywords(self, limit=20):
        """取得最常使用的關鍵字"""
        try:
            with db._get_connection() as conn:  # 使用全域 db 實例的連線
                cursor = conn.cursor()
                # MS SQL Server 使用 TOP
                cursor.execute(
                    "SELECT TOP (?) keyword, count FROM keyword_stats "
                    "ORDER BY count DESC;",
                    (limit,),
                )
                return [(keyword, count) for keyword, count in cursor.fetchall()]
        except pyodbc.Error as e:
            logger.exception(f"從 MS SQL Server 取得熱門關鍵字失敗: {e}")
            return []
        except Exception as e:
            logger.exception(f"取得熱門關鍵字時發生未知錯誤: {e}")
            return []

    def generate_daily_stats(self, date_str=None):  # 參數名改為 date_str 以清晰
        """
        生成每日統計數據

        參數:
            date_str: 日期字串 (YYYY-MM-DD)，若未提供則使用今天
        """
        if date_str is None:
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")

        # 確保 date_str 是 YYYY-MM-DD 格式
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.error(f"無效的日期格式: {date_str}. 請使用 YYYY-MM-DD.")
            return None

        try:
            with db._get_connection() as conn:  # 使用全域 db 實例的連線
                cursor = conn.cursor()

                # 總訊息數
                cursor.execute(
                    "SELECT COUNT(*) FROM conversations "
                    "WHERE CONVERT(date, timestamp) = ?;",
                    (date_str,),
                )
                total_messages = cursor.fetchone()[0]

                # 唯一使用者數 (基於 conversations 表的 sender_id)
                cursor.execute(
                    "SELECT COUNT(DISTINCT sender_id) FROM conversations "
                    "WHERE CONVERT(date, timestamp) = ?;",
                    (date_str,),
                )
                unique_users = cursor.fetchone()[0]

                # 事件計數
                cursor.execute(
                    "SELECT event_type, COUNT(*) FROM analytics_events "
                    "WHERE CONVERT(date, timestamp) = ? GROUP BY event_type;",
                    (date_str,),
                )
                event_counts = dict(cursor.fetchall())

                # 語言分佈 (從 user_preferences 獲取)
                cursor.execute(
                    "SELECT language, COUNT(*) FROM user_preferences GROUP BY language;"
                )
                language_distribution = dict(cursor.fetchall())

                stats_data = {
                    "date": date_str,
                    "total_messages": total_messages,
                    "unique_users": unique_users,
                    "events": event_counts,
                    "language_distribution": language_distribution,
                }
                stats_json = json.dumps(stats_data)

                cursor.execute(
                    "SELECT date FROM daily_stats WHERE date = ?;", (date_str,)
                )
                if cursor.fetchone():
                    cursor.execute(
                        "UPDATE daily_stats SET total_messages = ?, unique_users = ?, "
                        "data = ? WHERE date = ?;",
                        (total_messages, unique_users, stats_json, date_str),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO daily_stats (date, total_messages, "
                        "unique_users, data) VALUES (?, ?, ?, ?);",
                        (date_str, total_messages, unique_users, stats_json),
                    )
                conn.commit()
                return stats_data
        except pyodbc.Error as e:
            logger.exception(
                f"生成 MS SQL Server 每日統計 ({date_str}) 失敗: {e}"
            )
            return None
        except Exception as e:
            logger.exception(f"生成每日統計 ({date_str}) 時發生未知錯誤: {e}")
            return None

    def get_usage_trends(self, days=30):
        """
        取得使用趨勢數據

        參數:
            days: 要回溯的天數
        """
        end_date = datetime.datetime.now()
        # 包含今天，所以是 days-1
        start_date = end_date - datetime.timedelta(days=days - 1)

        date_range_str = []
        current_dt = start_date
        while current_dt <= end_date:
            date_range_str.append(current_dt.strftime("%Y-%m-%d"))
            current_dt += datetime.timedelta(days=1)

        message_trend_map = {day_str: 0 for day_str in date_range_str}
        user_trend_map = {day_str: 0 for day_str in date_range_str}

        try:
            with db._get_connection() as conn:  # 使用全域 db 實例的連線
                cursor = conn.cursor()
                # 訊息趨勢
                sql_message_trend = """
                    SELECT CONVERT(date, timestamp) as day, COUNT(*) as count
                    FROM conversations
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY CONVERT(date, timestamp)
                    ORDER BY day;
                """
                cursor.execute(
                    sql_message_trend,
                    (
                        start_date.strftime("%Y-%m-%d %H:%M:%S"),
                        end_date.strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                for row in cursor.fetchall():
                    day_str = (
                        row[0].strftime("%Y-%m-%d")
                        if isinstance(row[0], (datetime.date, datetime.datetime))
                        else str(row[0])
                    )
                    if day_str in message_trend_map:
                        message_trend_map[day_str] = row[1]

                # 使用者趨勢
                sql_user_trend = """
                    SELECT CONVERT(date, timestamp) as day, COUNT(DISTINCT sender_id) as count
                    FROM conversations
                    WHERE timestamp BETWEEN ? AND ?
                    GROUP BY CONVERT(date, timestamp)
                    ORDER BY day;
                """
                cursor.execute(
                    sql_user_trend,
                    (
                        start_date.strftime("%Y-%m-%d %H:%M:%S"),
                        end_date.strftime("%Y-%m-%d %H:%M:%S")
                    )
                )
                for row in cursor.fetchall():
                    day_str = (
                        row[0].strftime("%Y-%m-%d")
                        if isinstance(row[0], (datetime.date, datetime.datetime))
                        else str(row[0])
                    )
                    if day_str in user_trend_map:
                        user_trend_map[day_str] = row[1]

                trends = {
                    "dates": date_range_str,
                    "messages": [message_trend_map[day_str] for day_str in date_range_str],
                    "users": [user_trend_map[day_str] for day_str in date_range_str],
                }
                return trends
        except pyodbc.Error as e:
            logger.exception(f"從 MS SQL Server 取得使用趨勢失敗: {e}")
            return {
                "dates": date_range_str,
                "messages": [0]*len(date_range_str),
                "users": [0]*len(date_range_str)
            }
        except Exception as e:
            logger.exception(f"取得使用趨勢時發生未知錯誤: {e}")
            return {
                "dates": date_range_str,
                "messages": [0]*len(date_range_str),
                "users": [0]*len(date_range_str)
            }

    def export_stats(self, format_type="json"):  # 參數名改為 format_type
        """
        匯出統計數據

        參數:
            format_type: 輸出格式 (目前僅支援 json)
        """
        if format_type != "json":
            raise ValueError("目前僅支援 JSON 格式匯出")
        try:
            conversation_stats = self._get_conversation_stats()
            user_stats = self._get_user_stats()
            keyword_stats = self.get_top_keywords(50)
            usage_trends = self.get_usage_trends(30)
            export_data = {
                "generated_at": datetime.datetime.now().isoformat(),
                "conversation_stats": conversation_stats,
                "user_stats": user_stats,
                "top_keywords": dict(keyword_stats),  # 轉換為 dict
                "usage_trends": usage_trends,
            }
            # 確保 self.stats_path 是有效的
            if not self.stats_path:
                logger.error("統計檔案路徑 (self.stats_path) 未設定。無法匯出。")
                return None

            os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)  # 確保目錄存在
            with open(self.stats_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            logger.info(f"統計數據已成功匯出至: {self.stats_path}")
            return self.stats_path
        except Exception as e:
            logger.exception(f"匯出 MS SQL Server 統計數據失敗: {e}")
            return None

    def _get_conversation_stats(self):
        """取得對話統計數據 (MS SQL Server 版本)"""
        try:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conversations;")
                total_messages = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT sender_role, COUNT(*) FROM conversations GROUP BY sender_role;"
                )
                role_counts = dict(cursor.fetchall())
                # 最近24小時訊息
                cursor.execute(
                    "SELECT COUNT(*) FROM conversations "
                    "WHERE timestamp >= DATEADD(day, -1, GETDATE());"
                )
                last_24h = cursor.fetchone()[0]
                return {
                    "total_messages": total_messages,
                    "role_distribution": role_counts,
                    "last_24h": last_24h,
                }
        except pyodbc.Error as e:
            logger.exception(f"從 MS SQL Server 取得對話統計數據失敗: {e}")
            return {}
        except Exception as e:
            logger.exception(f"取得對話統計數據時發生未知錯誤: {e}")
            return {}

    def _get_user_stats(self):
        """取得使用者統計數據 (MS SQL Server 版本)"""
        try:
            with db._get_connection() as conn:
                cursor = conn.cursor()
                # 總使用者數 (基於 user_preferences 表)
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_preferences;")
                total_users = cursor.fetchone()[0]

                # 最近7天活躍使用者 (基於 conversations 表的 sender_id)
                cursor.execute(
                    "SELECT COUNT(DISTINCT sender_id) FROM conversations "
                    "WHERE timestamp >= DATEADD(day, -7, GETDATE());"
                )
                active_users_7d = cursor.fetchone()[0]

                # 語言分佈
                cursor.execute(
                    "SELECT language, COUNT(*) FROM user_preferences GROUP BY language;"
                )
                language_distribution = dict(cursor.fetchall())
                return {
                    "total_users": total_users,
                    "active_users_last_7_days": active_users_7d,
                    "language_distribution": language_distribution,
                }
        except pyodbc.Error as e:
            logger.exception(f"從 MS SQL Server 取得使用者統計數據失敗: {e}")
            return {}
        except Exception as e:
            logger.exception(f"取得使用者統計數據時發生未知錯誤: {e}")
            return {}


# 全域 analytics 實例
analytics = Analytics()
