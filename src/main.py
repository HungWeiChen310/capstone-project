import logging
import os
import re
import time
from html import escape
import openai # Modified for specific exception imports
from database import db


def sanitize_input(text):
    """
    清理使用者輸入，移除任何可能的 XSS 注入或有害內容
    """
    if not isinstance(text, str):
        return ""
    # 基本 HTML 跳脫 - Helps prevent XSS if content is rendered in HTML
    sanitized = escape(text)
    # 移除或替換潛在有害或不需要的字元
    # Current regex removes characters not in the whitelist.
    # This includes backticks (`), tildes (~), etc.
    # If these characters are essential for user input, this regex needs adjustment.
    # For now, assuming current restrictiveness is intended.
    # Test cases should verify behavior with special characters and XSS payloads.
    sanitized = re.sub(r'[^\w\s.,;?!@#$%^&*()-=+\[\]{}:"\'/\\<>]', "", sanitized)
    # Consider if allowing < and > is intended after escape(), as the regex allows them.
    # If they are meant to be fully removed, add them to the regex removal set or handle separately.
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
# OpenAI integration for chat responses


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
        """新增一則訊息到用戶的對話記錄中 (同時儲存到資料庫)。返回 True 表示成功，False 表示失敗。"""
        # 加入資料庫
        if not db.add_message(user_id, role, content):
            logger.error(f"UserData: 資料庫新增訊息失敗 (使用者 ID: {user_id})。")
            return False # Indicate failure

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


class OpenAIService:
    """處理與 OpenAI API 的互動邏輯"""

    def __init__(self, message, user_id):
        # user_id is used for database lookups and OpenAI 'user' field, assumed to be safe.
        # Avoid logging user_id excessively if PII is a concern for log access.
        self.user_id = user_id
        self.message = sanitize_input(message)

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.critical("OpenAIService: OpenAI API 金鑰未在環境變數中設置。")
            raise ValueError("OpenAI API 金鑰未設置")
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.critical(f"OpenAIService: 初始化 OpenAI client 失敗: {e}", exc_info=True)
            raise RuntimeError(f"無法初始化 OpenAI client: {e}")

        self.max_conversation_length = 10  # 保留最近的 10 輪對話
        self.user_prefs = db.get_user_preference(user_id)
        self.language = self.user_prefs.get("language", "zh-Hant")

    def get_fallback_response(self, error=None):
        """提供 OpenAI API 失敗時的備用回應"""
        fallback_responses = {
            "zh-Hant": "抱歉，我暫時無法處理您的請求。可能是網路連線問題或系統忙碌。請稍後再試，或輸入 'help' 查看其他功能。",
            "zh-Hans": "抱歉，我暂时无法处理您的请求。可能是网络连接问题或系统忙碌。请稍后再试，或输入 'help' 查看其他功能。",
            "en": "Sorry, I cannot process your request at the moment. This might be due to connectivity issues or \
                system load. Please try again later or type 'help' to see other features.",
            "ja": "申し訳ありませんが、現在リクエストを処理できません。接続の問題やシステムの負荷が原因かもしれません。後でもう一度お試しいただくか、「help」と入力して他の機能をご覧ください。",
            "ko": "죄송합니다. 현재 요청을 처리할 수 없습니다. 연결 문제나 시스템 로드로 인한 것일 수 있습니다. 나중에 다시 시도하거나 'help'를 입력하여 다른 기능을 확인하세요.",
        }
        # 使用對應語言的回覆，若無則使用繁體中文
        return fallback_responses.get(self.language, fallback_responses["zh-Hant"])

    def get_response(self):
        """向 OpenAI API 發送請求並獲取回應"""
        # 取得對話歷史
        conversation = user_data.get_conversation(self.user_id)
        # 確保對話不會超過 max_conversation_length
        if (
            len(conversation) >= self.max_conversation_length * 2
        ):  # 乘以 2 因為每輪對話有使用者和助手各一條
            # 保留系統提示和最近的對話
            conversation = (
                conversation[:1]
                + conversation[-(self.max_conversation_length * 2 - 1):]
            )
        # 檢查是否有系統提示，若無則加入
        if not conversation or conversation[0]["role"] != "system":
            system_prompt = get_system_prompt(self.language)
            conversation.insert(0, {"role": "system", "content": system_prompt})

        # 添加用戶的新訊息到對話歷史 (記憶體與資料庫)
        if not user_data.add_message(self.user_id, "user", self.message):
            # Failed to save user message to DB, critical issue
            logger.error(f"OpenAIService: 無法將用戶 {self.user_id} 的訊息儲存到資料庫。中止 OpenAI 請求。")
            return self.get_fallback_response("Failed to record user message.")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Consider making model configurable
                    messages=conversation,
                    max_tokens=500,  # Consider making max_tokens configurable
                    timeout=20,      # Increased timeout
                    user=self.user_id # Pass user_id to OpenAI for monitoring/moderation
                )
                ai_message = response.choices[0].message.content.strip()

                if not user_data.add_message(self.user_id, "assistant", ai_message):
                    # Failed to save AI message to DB, but we got a response
                    logger.error(f"OpenAIService: 無法將 AI 回應儲存到資料庫 (使用者 ID: {self.user_id})。AI 回應仍會回傳。")
                return ai_message

            except openai.AuthenticationError as e:
                logger.critical(f"OpenAI API 驗證失敗 (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                # No retry for auth errors
                break
            except openai.RateLimitError as e:
                logger.warning(f"OpenAI API 速率限制 (Attempt {attempt + 1}/{max_retries}): {e}. 等待 {2 ** attempt} 秒後重試。", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else: # Last attempt failed
                    break
            except openai.APITimeoutError as e:
                logger.warning(f"OpenAI API 請求超時 (Attempt {attempt + 1}/{max_retries}): {e}. 等待 {2 ** attempt} 秒後重試。", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    break
            except openai.APIConnectionError as e:
                logger.warning(f"OpenAI API 連線錯誤 (Attempt {attempt + 1}/{max_retries}): {e}. 等待 {2 ** attempt} 秒後重試。", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    break
            except openai.APIError as e: # General OpenAI API error
                logger.error(f"OpenAI API 發生錯誤 (Attempt {attempt + 1}/{max_retries}): {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(1) # Simple backoff for other API errors
                else:
                    break
            except Exception as e: # Catch other unexpected errors during API call
                logger.exception(f"呼叫 OpenAI API 時發生非預期錯誤 (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else: # Last attempt failed
                    break

        # If all retries fail or a non-retryable error occurred
        logger.error(f"OpenAI API 請求在 {max_retries} 次嘗試後失敗 (使用者 ID: {self.user_id})。")
        fallback_message = self.get_fallback_response("API request failed after retries.")
        if not user_data.add_message(self.user_id, "assistant", fallback_message):
             logger.error(f"OpenAIService: 無法將備用訊息儲存到資料庫 (使用者 ID: {self.user_id})。")
        return fallback_message


def reply_message(event):
    """處理用戶訊息並回傳 AI 回應"""
    # For LINE v3 API compatibility
    user_message = event.message.text
    user_id = event.source.user_id
    # 使用 OpenAI 服務產生回應
    openai_service = OpenAIService(message=user_message, user_id=user_id)
    response = openai_service.get_response()
    return response


# 如果直接執行此檔案，則啟動 Flask 應用
if __name__ == "__main__":
    try:
        # 動態導入 linebot_connect 以避免潛在的循環引用
        # 並允許 linebot_connect 中的初始化（如 db）在 main 中的 db 實例化後進行
        import importlib.util
        import sys

        # Assuming linebot_connect.py is in the same directory (src)
        module_path = os.path.join(os.path.dirname(__file__), "linebot_connect.py")
        if not os.path.exists(module_path):
            logger.critical(f"無法找到 linebot_connect.py 於: {module_path}")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("linebot_connect", module_path)
        if spec is None or spec.loader is None:
            logger.critical(f"無法為 linebot_connect.py 創建模組規格。")
            sys.exit(1)

        linebot_connect = importlib.util.module_from_spec(spec)
        sys.modules["linebot_connect"] = linebot_connect # Add to sys.modules before exec
        spec.loader.exec_module(linebot_connect)

        # Initialize equipment data and scheduler from linebot_connect
        # These functions should ideally be idempotent or handle multiple calls safely
        if hasattr(linebot_connect, 'initialize_equipment_data'):
            try:
                logger.info("正在初始化設備數據...")
                linebot_connect.initialize_equipment_data()
            except Exception as e_init:
                logger.error(f"初始化設備數據時發生錯誤: {e_init}", exc_info=True)

        if hasattr(linebot_connect, 'start_scheduler'):
            try:
                logger.info("正在啟動排程器...")
                linebot_connect.start_scheduler()
            except Exception as e_sched:
                logger.error(f"啟動排程器時發生錯誤: {e_sched}", exc_info=True)

        # Get port and debug settings from Config object (src/config.py)
        from config import Config
        port = Config.PORT
        # SSL context should also ideally be managed via Config or be more robust
        ssl_cert_path = os.getenv('SSL_CERT_PATH', 'certs/capstone-project.me-chain.pem')
        ssl_key_path = os.getenv('SSL_KEY_PATH', 'certs/capstone-project.me-key.pem')

        if not os.path.exists(ssl_cert_path) or not os.path.exists(ssl_key_path):
            logger.warning(f"SSL 憑證或金鑰檔案找不到。路徑: Cert='{ssl_cert_path}', Key='{ssl_key_path}'.")
            logger.warning("Flask 應用程式將以 HTTP 模式啟動 (若 DEBUG=True 且非生產環境)。")
            # Only run without SSL if not in production or explicitly allowed for debug
            if Config.APP_ENV != "production" and Config.DEBUG:
                 linebot_connect.app.run(host="0.0.0.0", port=port, debug=Config.DEBUG)
            else:
                logger.critical("生產環境中 SSL 憑證遺失，無法啟動伺服器。")
                sys.exit(1)
        else:
            logger.info(f"Flask 應用程式將在 HTTPS模式下啟動於埠 {port}")
            linebot_connect.app.run(
                ssl_context=(ssl_cert_path, ssl_key_path),
                host="0.0.0.0", port=port, debug=Config.DEBUG
            )

    except ImportError as e_imp:
        logger.critical(f"啟動主應用程式時發生導入錯誤: {e_imp}", exc_info=True)
        sys.exit(1)
    except SystemExit as e_sys: # Catch sys.exit from Config validation or SSL issues
        logger.info(f"應用程式因 SystemExit 而關閉: {e_sys.code}") # Usually already logged by Config
    except Exception as e_main:
        logger.critical(f"啟動主應用程式時發生未預期錯誤: {e_main}", exc_info=True)
        sys.exit(1)
