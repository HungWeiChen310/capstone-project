import os
import json
import logging
import re
import time
from html import escape
from openai import OpenAI
from src.database import db

def sanitize_input(text):
    """
    清理使用者輸入，移除任何可能的 XSS 注入或有害內容
    """
    if not isinstance(text, str):
        return ""
    
    # 基本 HTML 跳脫
    sanitized = escape(text)
    
    # 移除潛在惡意字元
    sanitized = re.sub(r'[^\w\s.,;?!@#$%^&*()-=+\[\]{}:"\'/\\<>]', '', sanitized)
    
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
        
        "en": """You are a professional technical consultant, focused on providing answers to engineering-related questions. Your answers should be specific, practical, and easy to understand.
               Please respond primarily in English unless the user asks in another language.
               The advice you provide should include practical steps and solutions. If you're unsure about an answer, please be honest about it.""",
        
        "ja": """あなたは専門技術コンサルタントで、エンジニアリング関連の質問に答えることに焦点を当てています。回答は具体的で実用的かつ理解しやすいものであるべきです。
               ユーザーが他の言語で質問しない限り、日本語で回答してください。
               提供するアドバイスには、実践的なステップや解決策を含めてください。回答に自信がない場合は、正直に述べてください。""",
        
        "ko": """귀하는 엔지니어링 관련 질문에 대한 답변을 제공하는 데 중점을 둔 전문 기술 컨설턴트입니다. 답변은 구체적이고 실용적이며 이해하기 쉬워야 합니다.
               사용자가 다른 언어로 질문하지 않는 한 한국어로 응답하십시오.
               제공하는 조언에는 실용적인 단계와 솔루션이 포함되어야 합니다. 답변이 확실하지 않은 경우 정직하게 말씀해 주십시오."""
    }
    
    return system_prompts.get(language, system_prompts["zh-Hant"])

# OpenAI integration for chat responses
class UserData:
    """存儲用戶對話記錄的類 - 使用資料庫與記憶體快取"""
    def __init__(self):
        self.temp_conversations = {}  # 暫存記憶體中的對話
    
    def get_conversation(self, user_id):
        """取得特定用戶的對話記錄，若不存在則初始化"""
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
        
        # 更新記憶體快取
        conversation = self.get_conversation(user_id)
        conversation.append({"role": role, "content": content})
        
        return conversation

user_data = UserData()

class OpenAIService:
    """處理與 OpenAI API 的互動邏輯"""
    def __init__(self, message, user_id):
        self.user_id = sanitize_input(user_id)
        self.message = sanitize_input(message)
        # 從環境變數獲取 OpenAI API 金鑰
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
           raise ValueError("OpenAI API 金鑰未設置")
        self.client = OpenAI(api_key=self.api_key)
        self.max_conversation_length = 10  # 保留最近的 10 輪對話
        
        # 取得使用者語言偏好
        self.user_prefs = db.get_user_preference(user_id)
        self.language = self.user_prefs.get("language", "zh-Hant")

    def get_fallback_response(self, error=None):
        """提供 OpenAI API 失敗時的備用回應"""
        fallback_responses = {
            "zh-Hant": "抱歉，我暫時無法處理您的請求。可能是網路連線問題或系統忙碌。請稍後再試，或輸入 'help' 查看其他功能。",
            "zh-Hans": "抱歉，我暂时无法处理您的请求。可能是网络连接问题或系统忙碌。请稍后再试，或输入 'help' 查看其他功能。",
            "en": "Sorry, I cannot process your request at the moment. This might be due to connectivity issues or system load. Please try again later or type 'help' to see other features.",
            "ja": "申し訳ありませんが、現在リクエストを処理できません。接続の問題やシステムの負荷が原因かもしれません。後でもう一度お試しいただくか、「help」と入力して他の機能をご覧ください。",
            "ko": "죄송합니다. 현재 요청을 처리할 수 없습니다. 연결 문제나 시스템 로드로 인한 것일 수 있습니다. 나중에 다시 시도하거나 'help'를 입력하여 다른 기능을 확인하세요."
        }
        
        # 使用對應語言的回覆，若無則使用繁體中文
        return fallback_responses.get(self.language, fallback_responses["zh-Hant"])

    def get_response(self):
        """向 OpenAI API 發送請求並獲取回應"""
        # 取得對話歷史
        conversation = user_data.get_conversation(self.user_id)
        
        # 確保對話不會超過 max_conversation_length
        if len(conversation) >= self.max_conversation_length * 2:  # 乘以 2 因為每輪對話有使用者和助手各一條
            # 保留系統提示和最近的對話
            conversation = conversation[:1] + conversation[-(self.max_conversation_length * 2 - 1):]
        
        # 檢查是否有系統提示，若無則加入
        if not conversation or conversation[0]["role"] != "system":
            system_prompt = get_system_prompt(self.language)
            conversation.insert(0, {"role": "system", "content": system_prompt})
        
        # 添加用戶的新訊息
        user_data.add_message(self.user_id, "user", self.message)
        
        try:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # 呼叫 OpenAI API
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=conversation,
                        max_tokens=500,
                        timeout=10  # 設定超時時間
                    )
                    
                    # 取得 AI 回應
                    ai_message = response.choices[0].message.content
                    
                    # 將 AI 回應加入對話歷史
                    user_data.add_message(self.user_id, "assistant", ai_message)
                    
                    return ai_message
                except Exception as e:
                    retry_count += 1
                    logging.warning(f"API 重試 {retry_count}/{max_retries}: {e}")
                    time.sleep(1)  # 等待 1 秒再重試
            
            # 若所有重試都失敗，使用備用回應
            fallback_message = self.get_fallback_response()
            user_data.add_message(self.user_id, "assistant", fallback_message)
            return fallback_message
            
        except Exception as e:
            logging.error(f"OpenAI API 錯誤: {e}")
            fallback_message = self.get_fallback_response(e)
            user_data.add_message(self.user_id, "assistant", fallback_message)
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
    # 避免循環引用問題
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("linebot_connect", 
                                                 os.path.join(os.path.dirname(__file__), "linebot_connect.py"))
    linebot_connect = importlib.util.module_from_spec(spec)
    sys.modules["linebot_connect"] = linebot_connect
    spec.loader.exec_module(linebot_connect)
    
    port = int(os.environ.get("PORT", 5000))
    linebot_connect.app.run(host="0.0.0.0", port=port, debug=False)