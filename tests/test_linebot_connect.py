import pytest
from unittest.mock import patch, MagicMock
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import MessageEvent, TextMessageContent, Source
from linebot.v3.messaging import TextMessage, ReplyMessageRequest

from src.linebot_connect import app, handler, handle_message, line_bot_api

@pytest.fixture
def client():
    with app.test_client() as client:
        # Configure client to simulate HTTPS requests to bypass Talisman redirects
        client.environ_base = {'wsgi.url_scheme': 'https', 'HTTP_X_FORWARDED_PROTO': 'https'}
        client.follow_redirects = False  # Still don't follow redirects for testing
        yield client

@patch.object(handler, 'handle')
def test_callback_valid_signature(mock_handle, client):
    headers = {'X-Line-Signature': 'valid_signature'}
    data = '{"events": [{"type": "message","message":{"type":"text","text":"Hello"},"replyToken":"dummy_token","source":{"userId":"user123"}}]}'
    resp = client.post('/callback', data=data, headers=headers)
    assert resp.status_code == 200
    mock_handle.assert_called_once_with(data, 'valid_signature')

def test_callback_no_signature(client):
    data = '{"events":[]}'
    resp = client.post('/callback', data=data)
    assert resp.status_code == 400

@patch.object(handler, 'handle', side_effect=InvalidSignatureError)
def test_callback_invalid_signature(mock_handle, client):
    headers = {'X-Line-Signature': 'invalid_signature'}
    data = '{"events":[]}'
    resp = client.post('/callback', data=data, headers=headers)
    assert resp.status_code == 400
    mock_handle.assert_called_once()

# ------------------------------
# 以下新增對 handle_message 的測試
# ------------------------------

class DummyEvent:
    def __init__(self, text, reply_token="dummy_token", user_id="user123"):
        self.message = TextMessageContent(
            text=text, 
            id="message123", 
            quoteToken="dummy_quote_token"  # 添加 quoteToken
        )
        self.reply_token = reply_token
        self.source = MagicMock(spec=Source)
        self.source.user_id = user_id

@patch('src.linebot_connect.get_powerbi_embed_config')
@patch.object(line_bot_api, 'reply_message_with_http_info')
def test_handle_message_powerbi(mock_reply_message, mock_get_embed_config):
    # 模擬取得 PowerBI 嵌入設定
    fake_config = {
        "embedUrl": "http://fake.powerbi.url",
        "accessToken": "fake_token",
        "reportId": "dummy_report_id",
        "workspaceId": "dummy_workspace_id"
    }
    mock_get_embed_config.return_value = fake_config
    
    # 傳入 "powerbi" 指令（不區分大小寫）
    event = DummyEvent("powerbi")
    handle_message(event)
    
    # 預期回覆文字包含 PowerBI 報表連結
    expected_text = f"請點選下方連結查看 PowerBI 報表：{fake_config['embedUrl']}"
    mock_reply_message.assert_called_once()
    
    # 檢查 ReplyMessageRequest 參數
    args, _ = mock_reply_message.call_args
    reply_request = args[0]
    assert isinstance(reply_request, ReplyMessageRequest)
    assert reply_request.reply_token == event.reply_token
    assert len(reply_request.messages) == 1
    assert reply_request.messages[0].text == expected_text

@patch.object(line_bot_api, 'reply_message_with_http_info')
@patch('src.main.reply_message', return_value="Fake ChatGPT response")
def test_handle_message_chatgpt(mock_main_reply, mock_reply_message):
    # 傳入一般訊息（非 PowerBI 指令）
    event = DummyEvent("Hello")
    handle_message(event)
    
    expected_text = "Fake ChatGPT response"
    # 確認先調用了 ChatGPT 模組的回覆函式
    mock_main_reply.assert_called_once_with(event)
    mock_reply_message.assert_called_once()
    
    # 檢查 ReplyMessageRequest 參數
    args, _ = mock_reply_message.call_args
    reply_request = args[0]
    assert isinstance(reply_request, ReplyMessageRequest)
    assert reply_request.reply_token == event.reply_token
    assert len(reply_request.messages) == 1
    assert reply_request.messages[0].text == expected_text