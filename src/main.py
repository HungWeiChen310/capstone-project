import logging
import os
import re
import time
import requests
from database import db
from rag import get_default_knowledge_base


def sanitize_input(text):
    """
    清理使用者輸入，移除任何可能的 XSS 注入或有害內容
    """
    if not isinstance(text, str):
        return ""
    # 只跳脫尖括號，避免改變其他合法字元
    sanitized = text.replace("<", "&lt;").replace(">", "&gt;")
    # 先允許保留反引號，稍後若無尖括號再移除
    sanitized = re.sub(r'[^\w\s.,;?!@#$%^&*()-=+\[\]{}:"\'/\\<>`]', "", sanitized)
    # 若字串包含被轉義的尖括號，僅保留自第一個尖括號之後的內容
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
    """根據語言選擇適當的系統提示"""
    system_prompts = {
        "zh-Hant": """你是一個專業的技術顧問，專注於提供工程相關問題的解答。回答應該具體、實用且易於理解。
                   請優先使用繁體中文回覆，除非使用者以其他語言提問。
                   提供的建議應包含實踐性的步驟和解決方案。如果不確定答案，請誠實表明。""",
        "zh-Hans": """你是一个专业的技术顾问，专注于提供工程相关问题的解答。回答应该具体、实用且易于理解。
                    请优先使用简体中文回复，除非用户以其他语言提问。
                    提供的建议应包含实践性的步骤和解决方案。如果不确定答案，请诚实表明。""",
        "en": """You are a professional technical consultant, focused on providing answers to engineering-related \
            questions. Your answers should be specific, practical, and easy to understand.
               Please respond primarily in English unless the user asks in another language.
               The advice you provide should include practical steps and solutions. If you're unsure about an answer, \
                   please be honest about it.""",
        "ja": """あなたは専門技術コンサルタントで、エンジニアリング関連の質問に答えることに焦点を当てています。回答は具体的で実用的かつ理解しやすいものであるべきです。
               ユーザーが他の言語で質問しない限り、日本語で回答してください。
               提供するアドバイスには、実践的なステップや解決策を含めてください。回答に自信がない場合は、正直に述べてください。""",
        "ko": """귀하는 엔지니어링 관련 질문에 대한 답변을 제공하는 데 중점을 둔 전문 기술 컨설턴트입니다. 답변은 구체적이고 실용적이며 이해하기 쉬워야 합니다.
               사용자가 다른 언어로 질문하지 않는 한 한국어로 응답하십시오.
               제공하는 조언에는 실용적인 단계와 솔루션이 포함되어야 합니다. 답변이 확실하지 않은 경우 정직하게 말씀해 주십시오.""",
    }
    return system_prompts.get(language, system_prompts["zh-Hant"])
# Ollama integration for chat responses


class UserData:
    """存儲用戶對話記錄的類 - 使用資料庫與記憶體快取"""

    def __init__(self, max_users=1000, max_messages=20, inactive_timeout=3600):
        self.temp_conversations = {}  # 暫存記憶體中的對話
        self.user_last_active = {}  # 記錄用戶最後活動時間
        self.max_users = max_users  # 最大快取用戶數
        self.max_messages = max_messages  # 每個用戶保留的最大訊息數
        self.inactive_timeout = inactive_timeout  # 不活躍超時時間(秒)
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """啟動清理線程"""
        import threading
        import time

        def cleanup_task():
            while True:
                time.sleep(1800)  # 每30分鐘清理一次
                self.periodic_cleanup()
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()

    def get_conversation(self, user_id):
        """取得特定用戶的對話記錄，若不存在則初始化"""
        # 更新最後活動時間
        self.user_last_active[user_id] = time.time()
        # 如果用戶數超過上限，清理最不活躍的用戶
        if len(self.temp_conversations) > self.max_users:
            self._cleanup_least_active_users()
        # 先檢查記憶體快取
        if user_id in self.temp_conversations:
            return self.temp_conversations[user_id]
        # 若不在記憶體中，從資料庫取得
        conversation = db.get_conversation_history(user_id)
        # 快取到記憶體
        self.temp_conversations[user_id] = conversation
        return conversation

    def add_message(self, user_id, role, content):
        """新增一則訊息到用戶的對話記錄中 (同時儲存到資料庫)"""
        # 加入資料庫
        db.add_message(user_id, role, content)
        # 更新最後活動時間
        self.user_last_active[user_id] = time.time()
        # 更新記憶體快取
        conversation = self.get_conversation(user_id)
        conversation.append({"role": role, "content": content})
        # 限制對話長度 (保留系統提示)
        if len(conversation) > self.max_messages + 1:
            # 保留第一條系統提示和最近的訊息
            if conversation[0]["role"] == "system":
                conversation = [conversation[0]] + conversation[-(self.max_messages):]
            else:
                conversation = conversation[-(self.max_messages):]
            self.temp_conversations[user_id] = conversation
        return conversation

    def _cleanup_least_active_users(self):
        """清理最不活躍的用戶"""
        # 按最後活動時間排序
        sorted_users = sorted(self.user_last_active.items(), key=lambda x: x[1])
        # 清理 20% 最不活躍的用戶
        users_to_remove = sorted_users[
            : int(len(sorted_users) * 0.2) or 1
        ]  # 至少移除1個
        for user_id, _ in users_to_remove:
            if user_id in self.temp_conversations:
                del self.temp_conversations[user_id]
            del self.user_last_active[user_id]

    def periodic_cleanup(self):
        """定期清理不活躍用戶的記憶體快取"""
        current_time = time.time()
        users_to_remove = []
        for user_id, last_active in list(self.user_last_active.items()):
            if current_time - last_active > self.inactive_timeout:
                users_to_remove.append(user_id)
        for user_id in users_to_remove:
            if user_id in self.temp_conversations:
                del self.temp_conversations[user_id]
            if user_id in self.user_last_active:
                del self.user_last_active[user_id]


user_data = UserData()


class OllamaService:
    """Handle interactions with the local Ollama model."""

    def __init__(self, message, user_id):
        self.user_id = user_id  # Changed: sanitize_input removed for user_id
        self.message = sanitize_input(message)  # No change for message
        self.ollama_host = os.getenv("OLLAMA_HOST", "127.0.0.1")
        self.ollama_port = self._parse_int(os.getenv("OLLAMA_PORT"), default=11434)
        self.ollama_scheme = os.getenv("OLLAMA_SCHEME", "http")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.session = requests.Session()
        self.max_conversation_length = 10  # 保留最近 10 輪對話
        # 讀取使用者語言偏好
        self.user_prefs = db.get_user_preference(user_id)
        self.language = self.user_prefs.get("language", "zh-Hant")
        self.rag_enabled = os.getenv("ENABLE_RAG", "true").lower() not in {"false", "0", "no"}
        self.rag_top_k = self._parse_int(os.getenv("RAG_TOP_K"), default=3)
        self.rag_min_score = self._parse_float(os.getenv("RAG_MIN_SCORE"), default=0.05)
        self.rag_max_context_chars = max(
            200, self._parse_int(os.getenv("RAG_MAX_CONTEXT_CHARS"), default=1800)
        )
        self.request_timeout = max(
            5.0,
            self._parse_float(os.getenv("OLLAMA_TIMEOUT"), default=30.0)
        )
    def get_fallback_response(self, error=None):
        """Provide a graceful reply when the Ollama API fails."""
        fallback_responses = {
            "zh-Hant": "抱歉，目前無法處理您的請求，可能是本地 Ollama 服務忙碌或連線異常。請稍後再試，或輸入 'help' 查看其他功能。",
            "zh-Hans": "抱歉，目前无法处理您的请求，可能是本地 Ollama 服务忙碌或连接异常。请稍后再试，或输入 'help' 查看其他功能。",
            "en": "Sorry, I cannot process your request right now. The local Ollama service might be busy or unreachable. Please try again later or type 'help' to see other features.",
            "ja": "申し訳ありませんが、現在リクエストを処理できません。ローカルの Ollama サービスが混雑しているか接続できない可能性があります。しばらくしてから再度お試しいただくか、『help』と入力して他の機能をご確認ください。",
            "ko": "죄송하지만 현재 요청을 처리할 수 없습니다. 로컬 Ollama 서비스가 바쁘거나 연결이 원활하지 않을 수 있습니다. 잠시 후 다시 시도하시거나 'help'를 입력해 다른 기능을 확인해 주세요.",
        }
        # 使用對應語言的回覆，無則使用繁體中文
        return fallback_responses.get(self.language, fallback_responses["zh-Hant"])

    def get_response(self):
        """Send a chat request to the local Ollama API and return the reply."""
        # 取得對話歷史
        conversation = user_data.get_conversation(self.user_id)
        # 確保對話不會超過 max_conversation_length
        if (
            len(conversation) >= self.max_conversation_length * 2
        ):  # 乘以 2 是為每輪對話保留使用者與助手訊息
            # 保留系統提示與最近對話
            conversation = (
                conversation[:1]
                + conversation[-(self.max_conversation_length * 2 - 1):]
            )
        # 檢查是否包含系統提示，若無則加入
        if not conversation or conversation[0]["role"] != "system":
            system_prompt = get_system_prompt(self.language)
            conversation.insert(0, {"role": "system", "content": system_prompt})
        # 新增使用者訊息
        user_data.add_message(self.user_id, "user", self.message)
        conversation_snapshot = [dict(msg) for msg in conversation]
        context_message = self._build_context_message()
        if context_message:
            insert_index = (
                1
                if conversation_snapshot and conversation_snapshot[0].get("role") == "system"
                else 0
            )
            conversation_snapshot.insert(
                insert_index,
                {"role": "system", "content": context_message},
            )
        try:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    api_url = f"{self.ollama_scheme}://{self.ollama_host}:{self.ollama_port}/api/chat"
                    payload = {
                        "model": self.ollama_model,
                        "messages": conversation_snapshot,
                        "stream": False,
                    }
                    response = self.session.post(
                        api_url,
                        json=payload,
                        timeout=self.request_timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    ai_message = ""
                    if isinstance(data, dict):
                        ai_message = data.get("message", {}).get("content") or data.get("response", "")
                    if not ai_message:
                        raise ValueError("Empty response from Ollama chat API")
                    # 將 AI 回覆存入對話歷史
                    user_data.add_message(self.user_id, "assistant", ai_message)
                    return ai_message
                except (requests.RequestException, ValueError) as exc:
                    retry_count += 1
                    logging.warning(
                        "Ollama chat request failed (attempt %s/%s): %s",
                        retry_count,
                        max_retries,
                        exc,
                    )
                    time.sleep(1)  # 等待 1 秒再重試
            # 所有重試都失敗，回傳預設訊息
            fallback_message = self.get_fallback_response()
            user_data.add_message(self.user_id, "assistant", fallback_message)
            return fallback_message
        except Exception as e:
            logging.error(f"Ollama service error: {e}")
            fallback_message = self.get_fallback_response(e)
            user_data.add_message(self.user_id, "assistant", fallback_message)
            return fallback_message

    def _build_context_message(self):
        """根據使用者訊息從知識庫擷取相關內容"""
        if not self.rag_enabled or not self.message:
            return None
        try:
            knowledge_base = get_default_knowledge_base()
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("初始化 RAG 知識庫失敗: %s", exc)
            return None
        if not knowledge_base.is_ready:
            return None
        results = knowledge_base.search(
            self.message,
            top_k=self.rag_top_k,
            min_score=self.rag_min_score,
        )
        if not results:
            return None
        formatted_sections = []
        sources_for_log = []
        for result in results:
            metadata = result.document.metadata
            source = metadata.get("source", result.document.doc_id)
            chunk_index = metadata.get("chunk_index")
            chunk_count = metadata.get("chunk_count")
            if chunk_index and chunk_count:
                display_source = f"{source} (節選 {chunk_index}/{chunk_count})"
            elif chunk_index:
                display_source = f"{source} (節選 {chunk_index})"
            else:
                display_source = source
            sources_for_log.append(display_source)
            formatted_sections.append(
                f"來源：{display_source}\n內容：{result.document.content.strip()}"
            )
        logging.debug("RAG 擷取來源：%s", sources_for_log)
        context = (
            "以下為知識庫擷取的相關內容，可作為回答使用者問題的參考。"
            "若資料不足以完整解答，請清楚說明缺少的資訊。\n\n"
            + "\n\n".join(formatted_sections)
        )
        if len(context) > self.rag_max_context_chars:
            context = context[: self.rag_max_context_chars - 3] + "..."
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


def reply_message(event):
    """處理用戶訊息並回傳 AI 回應"""
    # For LINE v3 API compatibility
    user_message = event.message.text
    user_id = event.source.user_id
    # 使用 Ollama 服務產生回應
    ollama_service = OllamaService(message=user_message, user_id=user_id)
    response = ollama_service.get_response()
    return response


# 如果直接執行此檔案，則啟動 Flask 應用
if __name__ == "__main__":
    # 避免循環引用問題
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(
        "linebot_connect", os.path.join(os.path.dirname(__file__), "linebot_connect.py")
    )
    linebot_connect = importlib.util.module_from_spec(spec)
    sys.modules["linebot_connect"] = linebot_connect
    spec.loader.exec_module(linebot_connect)
    port = int(os.environ.get("PORT", os.getenv("HTTPS_PORT", 443)))
    linebot_connect.app.run(
        ssl_context=(
            os.environ.get('SSL_CERT_PATH', 'certs/capstone-project.me-chain.pem'),
            os.environ.get('SSL_KEY_PATH', 'certs/capstone-project.me-key.pem')
        ),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=port,
        debug=False,
    )
