import os
import sys
import pytest
import pyodbc
from unittest.mock import patch, MagicMock, mock_open

# Ensure src is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import the module to be tested
from src import linebot_connect

# Mock database atexit registration if it's happening at import time in database.py
# This can prevent issues with tests if the DB connection is not truly needed.
if 'database' in sys.modules:
    if hasattr(sys.modules['database'], 'db') and hasattr(sys.modules['database'].db, '_conn'):
        # If db object and its connection exist, mock atexit if it's part of its cleanup
        patch('atexit.register', MagicMock()).start()

# --- Tests for get_or_create_secret_key ---

@patch.dict(os.environ, {}, clear=True) # Ensure no env var for SECRET_KEY initially
@patch("src.linebot_connect.os.path.exists")
@patch("builtins.open", new_callable=mock_open)
def test_get_or_create_secret_key_creates_new_key_if_no_file(mock_file_open, mock_exists):
    mock_exists.return_value = False # File does not exist

    # Resetting relevant global state in linebot_connect if any, for this function
    # This function uses a global SECRET_KEY_FILE
    original_secret_key_file = linebot_connect.SECRET_KEY_FILE
    linebot_connect.SECRET_KEY_FILE = "data/test_secret_key.txt" # Use a test-specific path

    try:
        key = linebot_connect.get_or_create_secret_key()
        assert len(key) == 48 # secrets.token_hex(24)
        mock_file_open.assert_called_once_with(linebot_connect.SECRET_KEY_FILE, "w")
        handle = mock_file_open()
        handle.write.assert_called_once_with(key)
    finally:
        linebot_connect.SECRET_KEY_FILE = original_secret_key_file
        if os.path.exists(linebot_connect.SECRET_KEY_FILE): # Clean up created dummy file
             os.remove(linebot_connect.SECRET_KEY_FILE)
        if os.path.exists("data"): # Clean up created dummy dir
             if not os.listdir("data"): # only remove if empty
                 os.rmdir("data")


@patch.dict(os.environ, {}, clear=True)
@patch("src.linebot_connect.os.path.exists")
@patch("builtins.open", new_callable=mock_open, read_data="mysecretkeyfromfile")
def test_get_or_create_secret_key_reads_from_file(mock_file_open, mock_exists):
    mock_exists.return_value = True # File exists
    original_secret_key_file = linebot_connect.SECRET_KEY_FILE
    linebot_connect.SECRET_KEY_FILE = "data/test_secret_key_exists.txt"

    try:
        key = linebot_connect.get_or_create_secret_key()
        assert key == "mysecretkeyfromfile"
        mock_file_open.assert_called_once_with(linebot_connect.SECRET_KEY_FILE, "r")
    finally:
        linebot_connect.SECRET_KEY_FILE = original_secret_key_file


@patch.dict(os.environ, {"SECRET_KEY": "myenvsecretkey"}, clear=True)
def test_get_or_create_secret_key_uses_env_variable():
    key = linebot_connect.get_or_create_secret_key()
    assert key == "myenvsecretkey"


@patch.dict(os.environ, {}, clear=True)
@patch("src.linebot_connect.os.path.exists")
@patch("builtins.open", new_callable=mock_open, read_data="") # File exists but is empty
def test_get_or_create_secret_key_creates_new_if_file_empty(mock_file_open, mock_exists):
    mock_exists.return_value = True
    original_secret_key_file = linebot_connect.SECRET_KEY_FILE
    linebot_connect.SECRET_KEY_FILE = "data/test_secret_key_empty.txt"

    try:
        key = linebot_connect.get_or_create_secret_key()
        assert len(key) == 48 # New key generated
        # Check it was opened for read, then for write
        assert mock_file_open.call_count == 2
        mock_file_open.assert_any_call(linebot_connect.SECRET_KEY_FILE, "r")
        mock_file_open.assert_any_call(linebot_connect.SECRET_KEY_FILE, "w")
        handle = mock_file_open()
        handle.write.assert_called_once_with(key)
    finally:
        linebot_connect.SECRET_KEY_FILE = original_secret_key_file
        if os.path.exists(linebot_connect.SECRET_KEY_FILE):
             os.remove(linebot_connect.SECRET_KEY_FILE)
        if os.path.exists("data"):
             if not os.listdir("data"):
                 os.rmdir("data")

# --- Flask App and Route Tests (Basic Setup) ---

@pytest.fixture
def app_client():
    """Create and configure a new app instance for each test."""
    # Ensure config is loaded for a testing environment
    with patch.dict(os.environ, {"APP_ENV": "testing", "TESTING": "True"}, clear=True):
        # Reload config if it reads env vars at import time
        if "src.config" in sys.modules:
            importlib.reload(sys.modules["src.config"])

        # Reload linebot_connect to re-initialize app with new config
        importlib.reload(linebot_connect)

        app = linebot_connect.app
        app.config.update({
            "TESTING": True,
            # "SECRET_KEY": "test_secret_for_flask_session", # Already handled by get_or_create
            "WTF_CSRF_ENABLED": False # Disable CSRF for simpler form tests
        })
        with app.test_client() as client:
            yield client


# Basic route test
def test_index_route(app_client):
    response = app_client.get("/")
    assert response.status_code == 200
    assert b"Service Status: Active" in response.data # Assuming index.html has this

# --- More tests to be added for admin_login, handle_message etc. ---
# For handle_message, extensive mocking of line_bot_api and db will be needed.

# Example: Test for admin_login input validation (more detailed tests needed)
def test_admin_login_input_validation(app_client):
    # Test short username
    response = app_client.post("/admin/login", data={"username": "u", "password": "longenoughpassword"})
    assert b"用戶名長度必須在 3 到 50 個字符之間。" in response.data # Encoded

    # Test short password
    response = app_client.post("/admin/login", data={"username": "validuser", "password": "short"})
    assert b"密碼長度必須在 8 到 100 個字符之間。" in response.data

def test_admin_login_success(app_client):
    # Mock os.getenv for ADMIN_USERNAME and ADMIN_PASSWORD
    # These are checked directly in linebot_connect.py
    with patch.dict(os.environ, {
        "ADMIN_USERNAME": "testadmin",
        "ADMIN_PASSWORD": "testpassword"
    }):
        # Need to reload linebot_connect if it caches these at module level,
        # or ensure it calls os.getenv dynamically. It does.
        response = app_client.post("/admin/login", data={
            "username": "testadmin",
            "password": "testpassword"
        }, follow_redirects=True) # Follow redirect to dashboard
    assert response.status_code == 200
    assert b"admin_dashboard" in response.request.path # Check path after redirect
    assert b"管理後台" in response.data # Check for content from dashboard
    with app_client.session_transaction() as session:
        assert session["admin_logged_in"] is True

def test_admin_login_failure(app_client):
    with patch.dict(os.environ, {
        "ADMIN_USERNAME": "testadmin",
        "ADMIN_PASSWORD": "testpassword"
    }):
        response = app_client.post("/admin/login", data={
            "username": "wrongadmin",
            "password": "wrongpassword"
        })
    assert response.status_code == 200 # Usually login pages return 200 on failure
    assert b"登入失敗，請確認帳號密碼是否正確" in response.data
    with app_client.session_transaction() as session:
        assert "admin_logged_in" not in session

@patch("src.linebot_connect.db") # Mock the database module used by the route
def test_admin_view_conversation_valid_user(mock_db, app_client):
    # Simulate admin logged in
    with app_client.session_transaction() as sess:
        sess["admin_logged_in"] = True

    mock_db.get_conversation_history.return_value = [
        {"sender_role": "user", "content": "Hello"},
        {"sender_role": "assistant", "content": "Hi there"}
    ]
    mock_db.get_user_preference.return_value = {"language": "zh-Hant", "role": "user"}

    user_id = "U12345678901234567890123456789012" # Valid format
    response = app_client.get(f"/admin/conversation/{user_id}")

    assert response.status_code == 200
    assert b"Hello" in response.data
    assert b"Hi there" in response.data
    mock_db.get_conversation_history.assert_called_once_with(user_id, limit=50)
    mock_db.get_user_preference.assert_called_once_with(user_id)

@patch("src.linebot_connect.db")
def test_admin_view_conversation_invalid_user_id_format(mock_db, app_client):
    with app_client.session_transaction() as sess:
        sess["admin_logged_in"] = True

    invalid_user_id = "Ushort"
    response = app_client.get(f"/admin/conversation/{invalid_user_id}", follow_redirects=True)

    assert response.status_code == 200 # Redirects to dashboard
    assert b"admin_dashboard" in response.request.path
    assert b"無效的使用者 ID 格式。" in response.data # Check for flash message
    mock_db.get_conversation_history.assert_not_called()

@patch("src.linebot_connect.db")
def test_admin_view_conversation_db_error(mock_db, app_client):
    with app_client.session_transaction() as sess:
        sess["admin_logged_in"] = True

    mock_db.get_conversation_history.side_effect = pyodbc.Error("Simulated DB Error")
    user_id = "U12345678901234567890123456789012"
    response = app_client.get(f"/admin/conversation/{user_id}", follow_redirects=True)

    assert response.status_code == 200
    assert b"admin_dashboard" in response.request.path
    assert b"查詢對話記錄時發生資料庫錯誤。" in response.data
    mock_db.get_conversation_history.assert_called_once_with(user_id, limit=50)


# Example: Test for handle_message language command (basic)
# This requires significant mocking.
@patch("src.linebot_connect.line_bot_api") # Mock the Line Bot API
@patch("src.linebot_connect.db") # Mock the database
def test_handle_message_language_setting(mock_db, mock_line_api):
    # Create a mock event object
    mock_event = MagicMock()
    mock_event.message.text = "language:zh-Hant"
    mock_event.source.user_id = "test_user_123"
    mock_event.reply_token = "test_reply_token"

    # Call the handler
    linebot_connect.handle_message(mock_event)

    # Assert db.set_user_preference was called correctly
    # Assuming set_user_preference returns True on success
    mock_db.set_user_preference.return_value = True
    mock_db.set_user_preference.assert_called_once_with("test_user_123", language="zh-Hant")

    # Assert reply_message_with_http_info was called
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    reply_request = args[0]
    assert reply_request.reply_token == "test_reply_token"
    assert "語言已切換至 繁體中文" in reply_request.messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_language_setting_invalid_code(mock_db, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "language:invalid-lang"
    mock_event.source.user_id = "test_user_invalid_lang"
    mock_event.reply_token = "test_reply_token_invalid_lang"

    linebot_connect.handle_message(mock_event)

    mock_db.set_user_preference.assert_not_called() # Should not be called for invalid lang
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "不支援的語言。" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_language_setting_db_fail(mock_db, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "language:zh-Hant"
    mock_event.source.user_id = "test_user_db_fail"
    mock_event.reply_token = "test_reply_token_db_fail"

    mock_db.set_user_preference.return_value = False # Simulate DB failure

    linebot_connect.handle_message(mock_event)

    mock_db.set_user_preference.assert_called_once_with("test_user_db_fail", language="zh-Hant")
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "語言設定失敗，請稍後再試。" in args[0].messages[0].text


# --- Tests for "設備狀態" (Equipment Status) command ---
@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db") # Mock the entire db object from database.py
def test_handle_message_equipment_status_success(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "設備狀態"
    mock_event.source.user_id = "test_user_eq_status"
    mock_event.reply_token = "test_reply_token_eq_status"

    # Mock the database connection and cursor
    mock_cursor = MagicMock()
    # First query for overall stats
    mock_cursor.fetchall.return_value = [
        ("die_bonder", 5, 3, 1, 1, 0, 0),
        ("wire_bonder", 2, 2, 0, 0, 0, 0)
    ]
    # Second query for abnormal equipment (called if first query returns stats)
    # This will be the second call to fetchall if the first one is successful
    mock_cursor.execute.side_effect = [
        None, # First execute for overall stats
        None  # Second execute for abnormal equipment
    ]

    # Simulate multiple fetchall calls if execute is called multiple times
    # The first fetchall is for stats, second for abnormal_equipments, third for latest_alert
    abnormal_eq_data = [("Bonder01", "die_bonder", "critical", "DB001")]
    latest_alert_data = [("High Temperature", "2023-01-01 10:00:00")]

    # This setup is a bit complex due to multiple execute/fetchall calls in the handler.
    # A more robust way would be to have db methods that are mocked,
    # but we are testing the direct SQL in linebot_connect.
    call_count = 0
    def fetchall_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1: # Overall stats
            return [("die_bonder", 5, 3, 1, 1, 0, 0), ("wire_bonder", 2, 2, 0, 0, 0, 0)]
        elif call_count == 2: # Abnormal equipment
            return abnormal_eq_data
        return [] # Default for other calls

    def fetchone_side_effect(*args, **kwargs):
        # This is for the latest_alert query inside the loop
        return latest_alert_data[0] if latest_alert_data else None

    mock_cursor.fetchall.side_effect = fetchall_side_effect
    mock_cursor.fetchone.side_effect = fetchone_side_effect

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn # Mock the connection context manager

    linebot_connect.handle_message(mock_event)

    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    reply_text = args[0].messages[0].text
    assert "📊 設備狀態摘要：" in reply_text
    assert "黏晶機：總數 5, 正常 3, 警告 1, 嚴重 1" in reply_text
    assert "打線機：總數 2, 正常 2" in reply_text
    assert "⚠️ 異常設備：" in reply_text
    assert "Bonder01 (黏晶機) 狀態: 🔴" in reply_text
    assert "最新警告: High Temperature 於 2023-01-01 10:00:00" in reply_text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_equipment_status_no_data(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "設備狀態"
    mock_event.source.user_id = "test_user_eq_status_no_data"
    mock_event.reply_token = "test_reply_token_eq_status_no_data"

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [] # No equipment stats
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)

    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "目前尚未設定任何設備。" in args[0].messages[0].text


@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_equipment_status_db_error(mock_db, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "設備狀態"
    mock_event.source.user_id = "test_user_db_error"
    mock_event.reply_token = "test_reply_token_db_error"

    # Mock the db connection to raise pyodbc.Error
    # This will be raised when `mock_db._get_connection()` is called
    mock_db._get_connection.side_effect = pyodbc.Error("Simulated DB Connection Error")

    linebot_connect.handle_message(mock_event)

    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    reply_request = args[0]
    assert "資料庫查詢失敗，無法取得設備狀態。" in reply_request.messages[0].text


# --- Tests for "設備詳情" (Equipment Details) command ---
@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_equipment_details_success(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "設備詳情 BonderTest01"
    mock_event.source.user_id = "test_user_eq_detail"
    mock_event.reply_token = "test_reply_token_eq_detail"

    mock_cursor = MagicMock()
    # Simulate DB responses for equipment details, metrics, alerts, operation
    mock_cursor.fetchone.side_effect = [
        ("EQ001", "BonderTest01", "die_bonder", "normal", "Location A", "2023-01-01 12:00:00"), # Equipment details
        ("processing_lot", "2023-01-01 13:00:00", "Lot123", "ProductX") # Current operation
    ]
    mock_cursor.fetchall.side_effect = [
        [("Temperature", 45.5, "C", "2023-01-01 13:30:00"), ("Pressure", 1.2, "Pa", "2023-01-01 13:30:00")], # Metrics
        [("Low Pressure", "warning", "2023-01-01 09:00:00")] # Active alerts
    ]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)

    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    reply_text = args[0].messages[0].text

    assert "設備詳情：" in reply_text
    assert "名稱: BonderTest01" in reply_text
    assert "類型: 黏晶機" in reply_text # Translated from die_bonder
    assert "狀態: ✅" in reply_text # normal
    assert "📊 最新監測值：" in reply_text
    assert "Temperature: 45.5 C" in reply_text
    assert "⚠️ 未解決的警告：" in reply_text
    assert "⚠️ Low Pressure 於 2023-01-01 09:00:00" in reply_text
    assert "🔄 目前運行中的作業：" in reply_text
    assert "作業類型: processing_lot" in reply_text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_equipment_details_not_found(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "設備詳情 NonExistent"
    mock_event.source.user_id = "test_user_eq_detail_nf"
    mock_event.reply_token = "test_reply_token_eq_detail_nf"

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None # Equipment not found
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "查無設備資料。" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
def test_handle_message_equipment_details_no_name(mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "設備詳情" # No name provided
    # user_id and reply_token are not strictly needed if it returns early
    mock_event.source.user_id = "test_user_eq_detail_no_name"
    mock_event.reply_token = "test_reply_token_eq_detail_no_name"

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "請指定設備名稱" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_equipment_details_db_error(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "設備詳情 SomeDevice"
    mock_event.source.user_id = "test_user_eq_detail_dberr"
    mock_event.reply_token = "test_reply_token_eq_detail_dberr"

    mock_db_module._get_connection.side_effect = pyodbc.Error("Simulated DB Error on connect")

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "資料庫查詢失敗，無法取得設備詳情。" in args[0].messages[0].text


# --- Tests for "訂閱設備" (Subscribe Equipment) command ---
@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_subscribe_list_available(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "訂閱設備" # No argument, should list
    mock_event.source.user_id = "test_user_sub_list"
    mock_event.reply_token = "test_reply_token_sub_list"

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("DB001", "Die Bonder A1", "die_bonder", "Fab1"),
        ("WB002", "Wire Bonder B2", "wire_bonder", "Fab2")
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    reply_text = args[0].messages[0].text
    assert "可訂閱的設備清單：" in reply_text
    assert "DB001 - Die Bonder A1 (Fab1)" in reply_text
    assert "WB002 - Wire Bonder B2 (Fab2)" in reply_text
    assert "使用方式: 訂閱設備 [設備ID]" in reply_text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_subscribe_success(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "訂閱設備 DB001"
    mock_event.source.user_id = "test_user_sub_success"
    mock_event.reply_token = "test_reply_token_sub_success"

    mock_cursor = MagicMock()
    # Side effect for multiple execute/fetchone calls:
    # 1. Check if equipment exists (fetchone returns equipment name)
    # 2. Check if already subscribed (fetchone returns None - not subscribed)
    # 3. Insert subscription (no return needed from execute for this mock)
    mock_cursor.fetchone.side_effect = [("Die Bonder A1",), None]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)

    assert mock_cursor.execute.call_count == 3 # equipment check, subscription check, insert
    mock_conn.commit.assert_called_once() # Ensure commit was called
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "訂閱成功！" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_subscribe_already_subscribed(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "訂閱設備 DB001"
    mock_event.source.user_id = "test_user_sub_already"
    mock_event.reply_token = "test_reply_token_sub_already"

    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [("Die Bonder A1",), (1,)] # Equipment exists, Is already subscribed (returns a row)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_conn.commit.assert_not_called() # No commit if already subscribed
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "您已訂閱該設備。" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_subscribe_equipment_not_found(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "訂閱設備 NotFoundEQ"
    mock_event.source.user_id = "test_user_sub_eq_nf"
    mock_event.reply_token = "test_reply_token_sub_eq_nf"

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None # Equipment not found
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "查無此設備ID，請確認後再試。" in args[0].messages[0].text


@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_subscribe_db_error_on_insert(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "訂閱設備 DB001"
    mock_event.source.user_id = "test_user_sub_dberr"
    mock_event.reply_token = "test_reply_token_sub_dberr"

    mock_cursor = MagicMock()
    mock_cursor.fetchone.side_effect = [("Die Bonder A1",), None] # Equipment exists, not subscribed
    mock_cursor.execute.side_effect = [
        None, # Check equipment
        None, # Check subscription
        pyodbc.Error("Simulated DB error on INSERT") # Error on insert
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_conn.commit.assert_not_called() # Should not commit if insert fails
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    # The actual error message might be generic "訂閱設備時資料庫操作失敗" due to the broad except pyodbc.Error
    assert "訂閱設備時資料庫操作失敗" in args[0].messages[0].text


# --- Tests for "取消訂閱" (Unsubscribe Equipment) command ---
@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_unsubscribe_list_subscribed(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "取消訂閱" # No argument
    mock_event.source.user_id = "test_user_unsub_list"
    mock_event.reply_token = "test_reply_token_unsub_list"

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("DB001", "Die Bonder A1", "die_bonder", "Fab1"),
        ("WB002", "Wire Bonder B2", "wire_bonder", "Fab2")
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    reply_text = args[0].messages[0].text
    assert "您已訂閱的設備：" in reply_text
    assert "DB001 - Die Bonder A1 (黏晶機, Fab1)" in reply_text # Type is translated
    assert "使用方式: 取消訂閱 [設備ID]" in reply_text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_unsubscribe_success(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "取消訂閱 DB001"
    mock_event.source.user_id = "test_user_unsub_success"
    mock_event.reply_token = "test_reply_token_unsub_success"

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("Die Bonder A1",) # Equipment found
    mock_cursor.rowcount = 1 # Simulate successful delete
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)

    assert mock_cursor.execute.call_count == 2 # find equipment, then delete
    mock_conn.commit.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "取消訂閱成功！" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_unsubscribe_not_subscribed(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "取消訂閱 DB001"
    mock_event.source.user_id = "test_user_unsub_not"
    mock_event.reply_token = "test_reply_token_unsub_not"

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = ("Die Bonder A1",) # Equipment found
    mock_cursor.rowcount = 0 # Simulate delete affecting 0 rows (not subscribed)
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_conn.commit.assert_not_called() # No commit if nothing deleted
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "您並未訂閱該設備。" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_unsubscribe_equipment_not_found(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "取消訂閱 NotFoundEQ"
    mock_event.source.user_id = "test_user_unsub_eq_nf"
    mock_event.reply_token = "test_reply_token_unsub_eq_nf"

    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None # Equipment not found
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "查無此設備ID，無法取消訂閱。" in args[0].messages[0].text


# --- Tests for "我的訂閱" (My Subscriptions) command ---
@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_my_subscriptions_has_subs(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "我的訂閱"
    mock_event.source.user_id = "test_user_my_subs"
    mock_event.reply_token = "test_reply_token_my_subs"

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("DB001", "Die Bonder A1", "die_bonder", "Fab1", "normal"),
        ("WB002", "Wire Bonder B2", "wire_bonder", "Fab2", "warning")
    ]
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    reply_text = args[0].messages[0].text
    assert "您已訂閱的設備：" in reply_text
    assert "DB001 - Die Bonder A1 (黏晶機, Fab1) 狀態: ✅" in reply_text
    assert "WB002 - Wire Bonder B2 (打線機, Fab2) 狀態: ⚠️" in reply_text
    assert "管理訂閱:" in reply_text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_my_subscriptions_no_subs(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "我的訂閱"
    mock_event.source.user_id = "test_user_no_subs"
    mock_event.reply_token = "test_reply_token_no_subs"

    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [] # No subscriptions
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_db_module._get_connection.return_value = mock_conn

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "您目前沒有訂閱任何設備。" in args[0].messages[0].text

@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.db")
def test_handle_message_my_subscriptions_db_error(mock_db_module, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "我的訂閱"
    mock_event.source.user_id = "test_user_my_subs_dberr"
    mock_event.reply_token = "test_reply_token_my_subs_dberr"

    mock_db_module._get_connection.side_effect = pyodbc.Error("Simulated DB Error")

    linebot_connect.handle_message(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "資料庫查詢失敗，無法獲取您的訂閱清單。" in args[0].messages[0].text


# --- Test for Default (OpenAI) case ---
# Patching 'src.main.reply_message' which is used by linebot_connect's handle_message
@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.reply_message") # Mock the imported reply_message
def test_handle_message_default_openai_call(mock_reply_message, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "Tell me a joke"
    mock_event.source.user_id = "test_user_openai"
    mock_event.reply_token = "test_reply_token_openai"

    mock_reply_message.return_value = "This is a joke from OpenAI."

    linebot_connect.handle_message(mock_event)

    mock_reply_message.assert_called_once_with(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert args[0].messages[0].text == "This is a joke from OpenAI."


@patch("src.linebot_connect.line_bot_api")
@patch("src.linebot_connect.reply_message")
def test_handle_message_default_openai_exception(mock_reply_message, mock_line_api):
    mock_event = MagicMock()
    mock_event.message.text = "Something to OpenAI"
    mock_event.source.user_id = "test_user_openai_exc"
    mock_event.reply_token = "test_reply_token_openai_exc"

    mock_reply_message.side_effect = Exception("Simulated OpenAI Service Error")

    linebot_connect.handle_message(mock_event)

    mock_reply_message.assert_called_once_with(mock_event)
    mock_line_api.reply_message_with_http_info.assert_called_once()
    args, _ = mock_line_api.reply_message_with_http_info.call_args
    assert "抱歉，處理您的請求時發生錯誤，請稍後再試。" in args[0].messages[0].text


# TODO: Add more tests for:
# - admin_login success and failure (mocking ADMIN_USERNAME/PASSWORD)
# - admin_view_conversation (user_id validation, db interaction)
# - handle_message:
#   - "設備詳情" (name parsing, db interaction, not found)
#   - "訂閱設備" (ID parsing, db interaction for exists/new subscription)
#   - "取消訂閱" (ID parsing, db interaction)
#   - "我的訂閱" (db interaction)
#   - Default case (OpenAI call - mock OpenAIService)
# - Test rate_limit_check (mock time.time)
# - Test other utility functions if any

# Cleanup data directory if it was created by tests and is empty
def teardown_module(module):
    datadir = "data"
    if os.path.exists(datadir) and not os.listdir(datadir):
        try:
            os.rmdir(datadir)
            print(f"Cleaned up empty directory: {datadir}")
        except OSError as e:
            print(f"Error removing directory {datadir}: {e}")
    elif os.path.exists(datadir) and os.listdir(datadir):
         # Clean up specific test files if they are known and were created
        test_files = [
            "data/test_secret_key.txt",
            "data/test_secret_key_exists.txt",
            "data/test_secret_key_empty.txt"
        ]
        for tf in test_files:
            if os.path.exists(tf):
                os.remove(tf)
        if not os.listdir(datadir): # Re-check if empty after removing specific files
             os.rmdir(datadir)
