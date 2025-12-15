import logging
import os
import re
import time
import requests
from typing import Optional, List
from src.database import db
from .rag import get_default_knowledge_base, RetrievalResult
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Vanna is no longer used for direct SQL generation in the reply flow,
# 雖然目前可以使用RAG.py中的向量資料庫來輔助回答，如果想要更追求精準的SQL生成，可以考慮保留Vanna的連接功能
# but the instance might be useful for other potential features.
# from vanna.ollama import Ollama
# from vanna.chromadb import ChromaDB_VectorStore
# class DBVanna(ChromaDB_VectorStore, Ollama):
#     def __init__(self, config=None):
#         ChromaDB_VectorStore.__init__(self, config=config)
#         Ollama.__init__(self, config=config)
# vn = DBVanna(config={'model': 'gpt-oss:20b'})
# resolved_server = Config.DB_SERVER
# resolved_database = Config.DB_NAME
# connection_string = (
#     "DRIVER={ODBC Driver 17 for SQL Server};"
#     f"SERVER={resolved_server};"
#     f"DATABASE={resolved_database};"
#     "Trusted_Connection=yes;"
#         )
# vn.connect_to_mssql(odbc_conn_str=connection_string)


def sanitize_input(text):
    """
    Cleans user input to remove any potential XSS injections or harmful content.
    """
    if not isinstance(text, str):
        return ""
    sanitized = text.replace("<", "&lt;").replace(">", "&gt;")
    sanitized = re.sub(r'[^\w\s.,;?!@#$%^&*()-=+\[\]{}:"\'/\\<>`]', "", sanitized)
    if "&lt;" in sanitized or "&gt;" in sanitized:
        first_pos = len(sanitized)
        if "&lt;" in sanitized:
            first_pos = min(first_pos, sanitized.find("&lt;"))
        if "&gt;" in sanitized:
            first_pos = min(first_pos, sanitized.find("&gt;"))
        sanitized = sanitized[first_pos:]
    else:
        sanitized = sanitized.replace("`", "")
    return sanitized


def get_system_prompt(language="zh-Hant"):
    """Selects the appropriate system prompt based on the language."""
    system_prompts = {
        "zh-Hant": (
            "你是一個專業的技術顧問，專注於提供工程相關問題的解答。"
            "回答應該具體、實用且易於理解。\n"
            "請優先使用繁體中文回覆，除非使用者以其他語言提問。\n"
            "提供的建議應包含實踐性的步驟和解決方案。如果不確定答案，請誠實表明。\n"
            "禁止使用任何形式的代碼塊標記（如```）和emoji來回覆內容，直接以純文字形式提供回答。\n"
            "提示詞提供的資料皆為資料庫中搜索的內容，請務必根據這些資料來回答使用者的問題，並在回答中引用相關來源，"
            "禁止憑空編造資訊以及不要要求使用者自行查詢。若資料不足以回答使用者的問題，請誠實告知並說明缺少哪些資訊。"
        ),
    }
    return system_prompts.get(language, system_prompts["zh-Hant"])


class UserData:
    """Stores user conversation history, using a database and in-memory cache."""
    def __init__(self, max_users=1000, max_messages=40, inactive_timeout=3600):
        self.temp_conversations = {}
        self.user_last_active = {}
        self.max_users = max_users
        self.max_messages = max_messages
        self.inactive_timeout = inactive_timeout
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        import threading
        import time

        def cleanup_task():
            while True:
                time.sleep(1800)
                self.periodic_cleanup()
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def get_conversation(self, user_id):
        """
        Retrieves conversation history for a user.
        If not in memory, tries to load from the database.
        """
        self.user_last_active[user_id] = time.time()

        # If in memory, return directly
        if user_id in self.temp_conversations:
            return self.temp_conversations[user_id]

        # Cleanup if cache is full
        if len(self.temp_conversations) > self.max_users:
            self._cleanup_least_active_users()

        # Try to load from DB
        logging.info(f"Loading conversation history for user {user_id} from database.")
        history_from_db = db.get_conversation_history(user_id, limit=self.max_messages)

        # Format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        # Ensure system prompt is handled in get_response, so just load raw messages here.
        conversation = history_from_db if history_from_db else []

        self.temp_conversations[user_id] = conversation
        return conversation

    def add_message(self, user_id, role, content):
        sender = user_id if role == "user" else "bot"
        receiver = "bot" if role == "user" else user_id

        # Persist to DB
        db.add_message(sender, receiver, role, content)

        self.user_last_active[user_id] = time.time()
        conversation = self.get_conversation(user_id)
        conversation.append({"role": role, "content": content})

        if self.max_messages and self.max_messages > 0:
            has_system = bool(conversation and conversation[0].get("role") == "system")
            keep_limit = self.max_messages + (1 if has_system else 0)
            if len(conversation) > keep_limit:
                if has_system:
                    conversation[:] = [conversation[0]] + conversation[-self.max_messages:]
                else:
                    conversation[:] = conversation[-self.max_messages:]
        return conversation

    def _cleanup_least_active_users(self):
        sorted_users = sorted(self.user_last_active.items(), key=lambda x: x[1])
        users_to_remove = sorted_users[: int(len(sorted_users) * 0.2) or 1]
        for user_id, _ in users_to_remove:
            if user_id in self.temp_conversations:
                del self.temp_conversations[user_id]
            del self.user_last_active[user_id]

    def periodic_cleanup(self):
        current_time = time.time()
        users_to_remove = [
            user_id for user_id, last_active in list(self.user_last_active.items())
            if current_time - last_active > self.inactive_timeout
        ]
        for user_id in users_to_remove:
            if user_id in self.temp_conversations:
                del self.temp_conversations[user_id]
            if user_id in self.user_last_active:
                del self.user_last_active[user_id]


user_data = UserData()


class OllamaService:
    """Handle interactions with the local Ollama model, now with improved RAG."""

    def __init__(self, message, user_id):
        self.user_id = user_id
        self.message = sanitize_input(message)
        self.ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1")
        self.ollama_port = self._parse_int(os.getenv("OLLAMA_PORT"), default=11434)
        self.ollama_scheme = os.getenv("OLLAMA_SCHEME", "http")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.session = requests.Session()
        self.max_conversation_length = 10
        self.user_prefs = db.get_user_preference(user_id)
        self.language = self.user_prefs.get("language", "zh-Hant")

        # Get user subscriptions
        self.subscribed_machines = db.get_user_subscriptions(user_id)

        # RAG settings
        self.rag_enabled = os.getenv("ENABLE_RAG", "true").lower() not in {"false", "0", "no"}
        self.rag_top_k = self._parse_int(os.getenv("RAG_TOP_K"), default=3)
        # Increased default for better quality
        self.rag_min_score = self._parse_float(os.getenv("RAG_MIN_SCORE"), default=0.4)
        self.rag_max_context_chars = max(
            200,
            # Increased context size
            self._parse_int(os.getenv("RAG_MAX_CONTEXT_CHARS"), default=2500),
        )
        self.request_timeout = max(
            5.0,
            self._parse_float(os.getenv("OLLAMA_TIMEOUT"), default=300.0)
        )

    def get_fallback_response(self, error=None):
        # Fallback responses remain the same
        fallback_responses = {
            "zh-Hant": "抱歉，目前無法處理您的請求，可能是本地 Ollama 服務忙碌或連線異常。請稍後再試，或輸入 'help' 查看其他功能。",
        }
        return fallback_responses.get(self.language, fallback_responses["zh-Hant"])

    def get_response(self):
        """Send a chat request to the local Ollama API and return the reply."""
        conversation = user_data.get_conversation(self.user_id)
        system_prompt = get_system_prompt(self.language)

        # Ensure system prompt exists
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, {"role": "system", "content": system_prompt})
        elif conversation[0].get("content") != system_prompt:
            conversation[0]["content"] = system_prompt

        user_data.add_message(self.user_id, "user", self.message)

        # Build access control instruction
        access_instruction = ""
        if self.subscribed_machines:
            machine_list = ", ".join(self.subscribed_machines)
            access_instruction = (
                f"\n[IMPORTANT ACCESS CONTROL]\n"
                f"The user is subscribed ONLY to the following machines: {machine_list}.\n"
                f"You MUST NOT provide any information about machines not in this list.\n"
                f"If the user asks about an unsubscribed machine, politely refuse to answer, "
                f"stating they do not have subscription access to it.\n"
            )
        else:
            access_instruction = (
                f"\n[IMPORTANT ACCESS CONTROL]\n"
                f"The user is NOT subscribed to any machines.\n"
                f"You MUST NOT provide specific machine status or details.\n"
                f"Politely inform the user they need to subscribe to a machine to view its details.\n"
            )

        context_message = self._build_context_message()

        # Prepare a clean snapshot for the API call
        conversation_snapshot = self._prepare_conversation_snapshot(conversation)

        # Combine access instruction and RAG context
        full_system_inject = ""
        if access_instruction:
            full_system_inject += access_instruction + "\n"
        if context_message:
            full_system_inject += context_message

        if full_system_inject:
            # Inject context after the system prompt
            conversation_snapshot.insert(1, {"role": "system", "content": full_system_inject})

        try:
            # Retry logic remains the same
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    base_url = f"{self.ollama_scheme}://{self.ollama_host}"
                    if ":" not in self.ollama_host.split("]")[-1]:
                        base_url += f":{self.ollama_port}"
                    api_url = f"{base_url}/api/chat"

                    payload = {"model": self.ollama_model, "messages": conversation_snapshot, "stream": False}

                    response = self.session.post(api_url, json=payload, timeout=self.request_timeout)
                    response.raise_for_status()
                    data = response.json()

                    ai_message = data.get("message", {}).get("content") or data.get("response", "")
                    if not ai_message:
                        raise ValueError("Empty response from Ollama chat API")

                    user_data.add_message(self.user_id, "assistant", ai_message)
                    return ai_message

                except (requests.RequestException, ValueError) as exc:
                    logging.warning("Ollama chat request failed (attempt %s/%s): %s", attempt + 1, max_retries, exc)
                    if attempt < max_retries - 1:
                        time.sleep(1)

            # If all retries fail
            fallback_message = self.get_fallback_response()
            user_data.add_message(self.user_id, "assistant", fallback_message)
            return fallback_message

        except Exception as e:
            logging.error(f"Ollama service error: {e}", exc_info=True)
            fallback_message = self.get_fallback_response(e)
            user_data.add_message(self.user_id, "assistant", fallback_message)
            return fallback_message

    def _build_context_message(self) -> Optional[str]:
        """Builds a context message by retrieving relevant content from the new vector knowledge base."""
        if not self.rag_enabled or not self.message:
            return None
        try:
            knowledge_base = get_default_knowledge_base()
            if not knowledge_base.is_ready:
                logging.warning("RAG knowledge base is not ready or empty.")
                return None
        except Exception as exc:
            logging.exception("Failed to initialize RAG knowledge base: %s", exc)
            return None

        try:
            results: List[RetrievalResult] = knowledge_base.search(
                self.message,
                top_k=self.rag_top_k,
                min_score=self.rag_min_score,
            )
        except Exception as exc:
            logging.exception("Error during RAG search: %s", exc)
            return None

        if not results:
            logging.info("RAG search returned no results for query: '%s'", self.message)
            return None

        formatted_sections = []
        sources_for_log = []
        for result in results:
            metadata = result.document.metadata
            source = metadata.get("source", "Unknown")

            # Improved source display
            display_source = source
            if metadata.get("origin") == "text" and "chunk_index" in metadata:
                display_source = f"{source} (Part {metadata['chunk_index']}/{metadata['chunk_count']})"
            elif metadata.get("origin") == "database":
                display_source = f"Database: {metadata.get('source_table', 'table')}"

            sources_for_log.append(f"{display_source} (Score: {result.score:.4f})")
            formatted_sections.append(
                f"Source: {display_source}\nContent: {result.document.content.strip()}"
            )

        logging.info("RAG retrieved sources:\n%s", "\n".join(sources_for_log))

        context_header = (
            "Based on the retrieved knowledge, here is some relevant information to help answer "
            "the user's question. If the information is insufficient, state what is missing."
        )
        context = f"{context_header}\n\n" + "\n\n---\n\n".join(formatted_sections)

        if len(context) > self.rag_max_context_chars:
            context = context[:self.rag_max_context_chars - 4] + "\n..."
        logging.info("RAG reply: %s", context)
        return context

    @staticmethod
    def _parse_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _prepare_conversation_snapshot(self, conversation):
        if not conversation:
            return []

        max_history = max(1, self.max_conversation_length * 2)
        system_message = conversation[0] if conversation[0].get("role") == "system" else None
        history = conversation[1:] if system_message else list(conversation)

        if max_history and len(history) > max_history:
            history = history[-max_history:]

        while history and history[0].get("role") != "user":
            history.pop(0)

        snapshot = [dict(system_message)] if system_message else []
        snapshot.extend(dict(msg) for msg in history)
        return snapshot


def reply_message(event):
    """Handles user messages and returns an AI-generated response using the unified RAG system."""
    user_message = event.message.text
    user_id = event.source.user_id

    # The Vanna SQL generation logic is now removed from the primary reply flow.
    # The new RAG system, which includes database content, provides a unified context source.

    # logging.info(f"Generating SQL query to respond to user's message")
    # try:
    #     sql = vn.generate_sql(user_message, allow_llm_to_see_data=True)
    #     logging.info(f"Executing SQL query: {sql}")
    #     df = vn.run_sql(sql)
    #     if df is not None and not df.empty:
    #         user_message += f"\nUse the following query result to provide an answer:\n{df.to_string(index=False)}"
    # except Exception as e:
    #     logging.warning("Vanna SQL generation failed: %s", e)
    #     pass

    # Use the Ollama service with the integrated RAG to generate a response
    ollama_service = OllamaService(message=user_message, user_id=user_id)
    response = ollama_service.get_response()
    return response
