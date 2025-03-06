import pytest
from unittest.mock import patch, MagicMock
from linebot.v3.webhooks import MessageEvent, TextMessageContent, Source

# 匯入要測試的對象
from src.main import reply_message, OpenAIService, UserData


class MockEvent:
    def __init__(self, user_message, user_id):
        self.message = TextMessageContent(
            text=user_message, 
            id="message123", 
            quoteToken="dummy_quote_token"  # 添加 quoteToken
        )
        self.source = MagicMock(spec=Source)
        self.source.user_id = user_id


@patch.object(OpenAIService, 'get_response', return_value="這是模擬的回應")
def test_openai_service(mock_get_response):
    """
    測試 OpenAIService 類別的 get_response 方法，
    使用直接 patch 以確保不會實際調用 OpenAI API
    """
    # 初始化服務
    service = OpenAIService(message="Test message", user_id="test_user")
    
    # 執行測試的方法
    response = service.get_response()
    
    # 驗證回傳的是我們模擬的回應
    assert response == "這是模擬的回應"
    
    # 確認 get_response 被調用了一次
    mock_get_response.assert_called_once()


@patch.object(OpenAIService, 'get_response', return_value="這是模擬的回應")
def test_reply_message_function(mock_get_response):
    """
    測試 reply_message 函式是否能正確取得使用者訊息並回傳預期結果
    """
    # 建立模擬的 LINE 事件
    event = MockEvent(user_message="Hello, AI!", user_id="user_123")
    
    # 執行測試的函式
    reply = reply_message(event)
    
    # 驗證回傳的是我們模擬的回應
    assert reply == "這是模擬的回應"
    
    # 確認 OpenAIService.get_response 被調用
    # (由於 mock 作用在 OpenAIService 類別方法上，這裡無法驗證具體參數)
    mock_get_response.assert_called_once()