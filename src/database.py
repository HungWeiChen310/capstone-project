"""Database module for handling conversation records and user preferences."""

import logging
import os
import pyodbc

# 設定日誌紀錄器
logger = logging.getLogger(__name__)


class Database:
    """Handle database operations for conversation records and user preferences."""

    def __init__(self, server="localhost", database="conversations"):
        """Initialize database connection.

        Args:
            server (str): Database server address. Defaults to "localhost".
            database (str): Database name. Defaults to "conversations".
        """
        self.connection_string = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
        )
        self._initialize_db()

    def _get_connection(self):
        """Create and return a database connection.

        Returns:
            pyodbc.Connection: A connection to the database.
        """
        return pyodbc.connect(self.connection_string)

    def _initialize_db(self):
        """Create necessary tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create conversations table
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

                # Create user_preferences table
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

                # Create equipment table
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

                # Create equipment_metrics table
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

                # Create equipment_operation_logs table
                cursor.execute(
                    """
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'equipment_operation_logs'
                    )
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

                # Create alert_history table
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

                # Create user_equipment_subscriptions table
                cursor.execute(
                    """
                    IF NOT EXISTS (
                        SELECT * FROM sys.tables WHERE name = 'user_equipment_subscriptions'
                    )
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
                logger.info("Database initialized successfully with equipment monitoring tables")
        except Exception as e:
            logger.exception(f"Database initialization failed: {e}")
            raise

    def add_message(self, user_id, role, content):
        """Add a new conversation message.

        Args:
            user_id (str): The user identifier.
            role (str): The role of the message sender.
            content (str): The message content.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO conversations (user_id, role, content)
                    VALUES (?, ?, ?)
                    """,
                    (user_id, role, content)
                )
                conn.commit()
                return True
        except Exception:
            logger.exception("Failed to add conversation message")
            return False

    def get_conversation_history(self, user_id, limit=10):
        """Get conversation history for a specific user.

        Args:
            user_id (str): The user identifier.
            limit (int, optional): Maximum number of messages to return. Defaults to 10.

        Returns:
            list: List of conversation messages, each containing role and content.
        """
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
                messages.reverse()  # Reverse order to show oldest messages first
                return messages
        except Exception:
            logger.exception("Failed to retrieve conversation history")
            return []

    def set_user_preference(self, user_id, language=None):
        """Set or update user preferences.

        Args:
            user_id (str): The user identifier.
            language (str, optional): Preferred language. Defaults to None.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
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
            logger.exception("Failed to set user preferences")
            return False

    def get_user_preference(self, user_id):
        """Get user preferences.

        Args:
            user_id (str): The user identifier.

        Returns:
            dict: User preferences including language setting.
        """
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

                # Create default preferences if not found
                self.set_user_preference(user_id)
                return {"language": "zh-Hant"}
        except Exception:
            logger.exception("Failed to retrieve user preferences")
            return {"language": "zh-Hant"}

    def get_conversation_stats(self):
        """Get conversation statistics.

        Returns:
            dict: Statistics including total messages, unique users, and message counts
                 by role.
        """
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
            logger.exception("Failed to retrieve conversation statistics")
            return {
                "total_messages": 0,
                "unique_users": 0,
                "last_24h": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0,
            }

    def get_recent_conversations(self, limit=20):
        """Get list of recent conversations.

        Args:
            limit (int, optional): Maximum number of conversations to return.
                                 Defaults to 20.

        Returns:
            list: List of recent conversations with user details and message counts.
        """
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
            logger.exception("Failed to retrieve recent conversations")
            return []


# Create singleton database instance
db = Database()
