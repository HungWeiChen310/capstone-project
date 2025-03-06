import sqlite3
import logging
import os
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

class Database:
    """Database handler for storing conversation history and user preferences"""
    
    def __init__(self, db_path="data/conversations.db"):
        """Initialize the database connection"""
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create necessary tables if they don't exist"""
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create conversations table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create user preferences table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    language TEXT DEFAULT "zh-Hant",
                    last_active DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def add_message(self, user_id, role, content):
        """Add a new message to the conversation history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (user_id, role, content) VALUES (?, ?, ?)",
                    (user_id, role, content)
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding message to database: {e}")
            return False
    
    def get_conversation_history(self, user_id, limit=10):
        """Retrieve conversation history for a specific user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT role, content FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit)
                )
                # Reverse the order to get oldest messages first
                messages = [{"role": role, "content": content} for role, content in cursor.fetchall()]
                messages.reverse()
                return messages
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    
    def set_user_preference(self, user_id, language=None):
        """Set or update user preferences"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user exists
                cursor.execute("SELECT user_id FROM user_preferences WHERE user_id = ?", (user_id,))
                user_exists = cursor.fetchone()
                
                if user_exists:
                    # Update existing user
                    updates = []
                    params = []
                    
                    if language:
                        updates.append("language = ?")
                        params.append(language)
                    
                    if updates:
                        updates.append("last_active = CURRENT_TIMESTAMP")
                        query = f"UPDATE user_preferences SET {', '.join(updates)} WHERE user_id = ?"
                        params.append(user_id)
                        cursor.execute(query, params)
                else:
                    # Create new user
                    fields = ["user_id"]
                    values = [user_id]
                    
                    if language:
                        fields.append("language")
                        values.append(language)
                    
                    placeholders = ", ".join(["?"] * len(values))
                    query = f"INSERT INTO user_preferences ({', '.join(fields)}) VALUES ({placeholders})"
                    cursor.execute(query, values)
                
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error setting user preference: {e}")
            return False
    
    def get_user_preference(self, user_id):
        """Get user preferences"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT language FROM user_preferences WHERE user_id = ?",
                    (user_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    return {"language": result[0]}
                else:
                    # Create default preferences
                    self.set_user_preference(user_id)
                    return {"language": "zh-Hant"}
        except Exception as e:
            logger.error(f"Error retrieving user preference: {e}")
            return {"language": "zh-Hant"}
    
    def get_conversation_stats(self):
        """Get basic conversation statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total message count
                cursor.execute("SELECT COUNT(*) FROM conversations")
                total_messages = cursor.fetchone()[0]
                
                # Get unique user count
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                unique_users = cursor.fetchone()[0]
                
                # Get messages in last 24 hours
                cursor.execute("SELECT COUNT(*) FROM conversations WHERE timestamp > datetime('now', '-1 day')")
                last_24h = cursor.fetchone()[0]
                
                # Get user messages vs. assistant messages
                cursor.execute("SELECT role, COUNT(*) FROM conversations GROUP BY role")
                role_counts = dict(cursor.fetchall())
                
                return {
                    "total_messages": total_messages,
                    "unique_users": unique_users,
                    "last_24h": last_24h,
                    "user_messages": role_counts.get("user", 0),
                    "assistant_messages": role_counts.get("assistant", 0),
                    "system_messages": role_counts.get("system", 0)
                }
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {
                "total_messages": 0,
                "unique_users": 0,
                "last_24h": 0,
                "user_messages": 0,
                "assistant_messages": 0,
                "system_messages": 0
            }
    
    def get_recent_conversations(self, limit=20):
        """Get a list of recent conversations"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get users with recent conversations
                cursor.execute("""
                    SELECT DISTINCT c.user_id, p.language, MAX(c.timestamp) as last_message
                    FROM conversations c
                    LEFT JOIN user_preferences p ON c.user_id = p.user_id
                    GROUP BY c.user_id
                    ORDER BY last_message DESC
                    LIMIT ?
                """, (limit,))
                
                results = []
                for user_id, language, timestamp in cursor.fetchall():
                    # Get message count for this user
                    cursor.execute("SELECT COUNT(*) FROM conversations WHERE user_id = ?", (user_id,))
                    message_count = cursor.fetchone()[0]
                    
                    # Get preview of last message
                    cursor.execute("""
                        SELECT content FROM conversations 
                        WHERE user_id = ? AND role = 'user'
                        ORDER BY timestamp DESC LIMIT 1
                    """, (user_id,))
                    last_message = cursor.fetchone()
                    
                    results.append({
                        "user_id": user_id,
                        "language": language or "zh-Hant",
                        "last_activity": timestamp,
                        "message_count": message_count,
                        "last_message": last_message[0] if last_message else ""
                    })
                
                return results
        except Exception as e:
            logger.error(f"Error getting recent conversations: {e}")
            return []

# Create a singleton database instance
db = Database()