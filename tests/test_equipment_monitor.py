import os
import sys
import pytest
import pyodbc
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta

# Ensure src is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.equipment_monitor import EquipmentMonitor
# Assuming database.py contains a db instance. We'll mock its methods.
from src.database import Database


@pytest.fixture
def monitor():
    """Fixture to create an EquipmentMonitor instance with a mocked db."""
    # Mock the global 'db' instance from the database module if EquipmentMonitor uses it directly
    # Or, if EquipmentMonitor takes a db instance in __init__, mock that.
    # Current EquipmentMonitor uses self.db = db (global instance)

    # We need to mock the methods of the actual 'db' instance that EquipmentMonitor uses.
    # Let's create a MagicMock that can be assigned to equipment_monitor.db
    mock_db_instance = MagicMock(spec=Database)

    # If _get_connection is called on the mock_db_instance
    mock_conn = MagicMock(spec=pyodbc.Connection)
    mock_cursor = MagicMock(spec=pyodbc.Cursor)
    mock_conn.cursor.return_value = mock_cursor
    # Simulate context manager for 'with self.db._get_connection() as conn:'
    mock_db_instance._get_connection.return_value.__enter__.return_value = mock_conn

    with patch('src.equipment_monitor.db', mock_db_instance):
        monitor_instance = EquipmentMonitor()
        # Attach mocks for easier access in tests if needed, e.g. monitor_instance.mock_cursor = mock_cursor
        monitor_instance._db_mock = mock_db_instance # For tests to access the main mock
        monitor_instance._conn_mock = mock_conn    # For tests to access the connection mock
        monitor_instance._cursor_mock = mock_cursor # For tests to access the cursor mock
        yield monitor_instance

# --- Tests for _determine_severity ---
@pytest.mark.parametrize("metric_type,value,min_val,max_val,expected_severity", [
    ("溫度", 100, 20, 80, EquipmentMonitor.SEVERITY_CRITICAL), # 100 > 80*1.1 (88) -> Critical (assuming 1.2 for emergency)
    ("溫度", 100, 20, 70, EquipmentMonitor.SEVERITY_EMERGENCY),# 100 > 70*1.2 (84) -> Emergency
    ("溫度", 85, 20, 80, EquipmentMonitor.SEVERITY_WARNING),  # 85 > 80 -> Warning
    ("溫度", 10, 5, 80, EquipmentMonitor.SEVERITY_WARNING),   # 10 < 5 (error in logic, should be 10 > 5, this implies min is lower bound)
                                                              # Corrected: value < min_val
    ("壓力", 50, 60, 100, EquipmentMonitor.SEVERITY_WARNING), # Value is 50, min is 60. Below min.
    ("良率", 0.7, 0.9, 1.0, EquipmentMonitor.SEVERITY_CRITICAL), # 0.7 < 0.9*0.8 (0.72) -> Critical
    ("良率", 0.85, 0.9, 1.0, EquipmentMonitor.SEVERITY_WARNING),# 0.85 < 0.9 -> Warning
    ("未知指標", 100, 0, 50, EquipmentMonitor.SEVERITY_WARNING), # Unknown metric type
    ("溫度", None, 0, 50, EquipmentMonitor.SEVERITY_WARNING), # Value is None
    ("溫度", 50, None, None, EquipmentMonitor.SEVERITY_WARNING), # No thresholds
])
def test_determine_severity(monitor, metric_type, value, min_val, max_val, expected_severity):
    # Note: The original _determine_severity had some specific logic for value < min if max was not exceeded.
    # The refactored version in my thought process was more structured. Testing against the provided code structure.
    # The current _determine_severity in problem description doesn't handle value < min_val for temp/pressure/rpm directly.
    # It seems to only trigger warning for those if value > max_val but not critically/emergently so.
    # This test will reflect the logic in the *refactored* version from the previous steps.

    # Adapting test based on the provided _determine_severity logic (which might be the one to be refactored)
    # For "溫度", "壓力", "轉速":
    #   - Emergency if value >= max * 1.2
    #   - Critical if value >= max * 1.1
    #   - Warning if value > max (but less than 1.1*max) OR value < min (this part was missing in original, added in refactor)
    # For "良率", etc.:
    #   - Critical if value <= min * 0.8
    #   - Warning if value < min (but greater than 0.8*min) OR value > max

    # Re-evaluating expectations based on the *refactored* _determine_severity from previous step's thought process
    if metric_type == "溫度":
        if value is not None and max_val is not None and value >= max_val * 1.2:
            expected_severity = EquipmentMonitor.SEVERITY_EMERGENCY
        elif value is not None and max_val is not None and value >= max_val * 1.1:
            expected_severity = EquipmentMonitor.SEVERITY_CRITICAL
        elif (value is not None and max_val is not None and value > max_val) or \
             (value is not None and min_val is not None and value < min_val):
            expected_severity = EquipmentMonitor.SEVERITY_WARNING
        elif value is None or (min_val is None and max_val is None): # No thresholds or None value
             expected_severity = EquipmentMonitor.SEVERITY_WARNING


    elif metric_type == "良率":
        if value is not None and min_val is not None and value <= min_val * 0.8:
            expected_severity = EquipmentMonitor.SEVERITY_CRITICAL
        elif (value is not None and min_val is not None and value < min_val) or \
             (value is not None and max_val is not None and value > max_val):
            expected_severity = EquipmentMonitor.SEVERITY_WARNING
        elif value is None or (min_val is None and max_val is None):
            expected_severity = EquipmentMonitor.SEVERITY_WARNING

    assert monitor._determine_severity(metric_type, value, min_val, max_val) == expected_severity

def test_determine_severity_invalid_value_type(monitor, caplog):
    assert monitor._determine_severity("溫度", "not-a-number", 0, 100) == EquipmentMonitor.SEVERITY_WARNING
    assert "無效的 'value' (not-a-number)" in caplog.text
    caplog.clear()
    assert monitor._determine_severity("溫度", 50, "invalid_min", 100) == EquipmentMonitor.SEVERITY_WARNING
    assert "無效的 'threshold_min' (invalid_min)" in caplog.text
    caplog.clear()
    assert monitor._determine_severity("溫度", 50, 0, "invalid_max") == EquipmentMonitor.SEVERITY_WARNING
    assert "無效的 'threshold_max' (invalid_max)" in caplog.text


# --- Tests for _get_equipment_data ---
def test_get_equipment_data_found(monitor):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchone.return_value = ("Bonder01", "die_bonder", "Fab A")

    data = monitor._get_equipment_data(monitor._conn_mock, "EQ001")

    mock_cursor.execute.assert_called_once_with(
        "\n            SELECT name, type, location\n            FROM equipment\n            WHERE equipment_id = ?\n            ",
        ("EQ001",)
    )
    assert data["name"] == "Bonder01"
    assert data["type"] == "die_bonder"
    assert data["type_name"] == "黏晶機" # From monitor.equipment_type_names
    assert data["location"] == "Fab A"

def test_get_equipment_data_not_found(monitor, caplog):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchone.return_value = None # Simulate not found

    data = monitor._get_equipment_data(monitor._conn_mock, "EQ_NOT_FOUND")

    assert "找不到設備 ID: EQ_NOT_FOUND" in caplog.text
    assert data["name"] == "未知設備 (EQ_NOT_FOUND)"
    assert data["type_name"] == "未知設備"
    assert data["error_reason"] == "Not found in DB"

def test_get_equipment_data_db_error(monitor, caplog):
    # Test the wrapper _get_equipment_data directly
    monitor._db_mock._get_connection.return_value.__enter__.side_effect = pyodbc.Error("Simulated DB connection error in _get_equipment_data")

    data = monitor._get_equipment_data(None, "EQ_DB_ERROR") # Pass None for conn to trigger new connection attempt

    assert "取得設備資料時無法建立新的資料庫連線" in caplog.text # From the fallback in _get_equipment_data
    assert data["name"] == "未知設備 (EQ_DB_ERROR)"
    assert "Fallback connection error" in data["error_reason"]

# --- Tests for _check_equipment_metrics (Example Structure) ---
@patch("src.equipment_monitor.EquipmentMonitor._send_alert_notification")
@patch("src.equipment_monitor.EquipmentMonitor._generate_ai_recommendation")
def test_check_equipment_metrics_normal(mock_ai_rec, mock_send_alert, monitor):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchall.return_value = [
        ("Temperature", 75.0, 20.0, 80.0, "C"), # Normal
        ("Pressure", 1.0, 0.5, 1.5, "atm")      # Normal
    ]

    monitor._check_equipment_metrics(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder")

    mock_send_alert.assert_not_called()
    # Check that no alert was inserted and status not updated
    # This requires checking that execute was not called with INSERT or UPDATE
    insert_update_calls = [
        c for c in mock_cursor.execute.call_args_list
        if "INSERT INTO alert_history" in c.args[0] or "UPDATE equipment" in c.args[0]
    ]
    assert not insert_update_calls
    monitor._conn_mock.commit.assert_not_called()


@patch("src.equipment_monitor.EquipmentMonitor._send_alert_notification")
@patch("src.equipment_monitor.EquipmentMonitor._generate_ai_recommendation")
def test_check_equipment_metrics_abnormal_critical(mock_ai_rec, mock_send_alert, monitor, caplog):
    mock_cursor = monitor._cursor_mock
    # Temperature significantly above max threshold -> critical
    mock_cursor.fetchall.return_value = [
        ("Temperature", 95.0, 20.0, 80.0, "C"), # 95 > 80 * 1.1 (88) but < 80 * 1.2 (96) -> Critical
    ]
    mock_ai_rec.return_value = "AI suggestion for critical temp"
    # Mock _get_equipment_data called by _generate_ai_recommendation
    with patch.object(monitor, '_get_equipment_data', return_value={"name": "Bonder01", "type": "die_bonder", "location": "Fab A"}):
        monitor._check_equipment_metrics(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder")

    mock_send_alert.assert_called_once()
    # Check severity passed to send_alert
    assert mock_send_alert.call_args[0][2] == EquipmentMonitor.SEVERITY_CRITICAL
    # Check message content for AI rec
    assert "AI建議: AI suggestion for critical temp" in mock_send_alert.call_args[0][1]

    # Check DB operations
    assert any("INSERT INTO alert_history" in c.args[0] for c in mock_cursor.execute.call_args_list)
    assert any("UPDATE equipment" in c.args[0] and "status = ?" in c.args[0] and c.args[1][0] == "critical" for c in mock_cursor.execute.call_args_list)
    monitor._conn_mock.commit.assert_called_once()

@patch("src.equipment_monitor.EquipmentMonitor._send_alert_notification")
@patch("src.equipment_monitor.EquipmentMonitor._generate_ai_recommendation")
def test_check_equipment_metrics_missing_value(mock_ai_rec, mock_send_alert, monitor, caplog):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchall.return_value = [
        ("Temperature", None, 20.0, 80.0, "C"), # Value is None
    ]

    monitor._check_equipment_metrics(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder")

    assert "缺少 'value'，跳過此指標。" in caplog.text
    mock_send_alert.assert_not_called()
    assert not any("INSERT INTO alert_history" in c.args[0] for c in mock_cursor.execute.call_args_list)
    monitor._conn_mock.commit.assert_not_called()


@patch("src.equipment_monitor.EquipmentMonitor._send_alert_notification")
@patch("src.equipment_monitor.EquipmentMonitor._generate_ai_recommendation")
def test_check_equipment_metrics_invalid_threshold_type(mock_ai_rec, mock_send_alert, monitor, caplog):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchall.return_value = [
        ("Pressure", 1.0, "invalid_min", 1.5, "atm"), # Min threshold is not a number
    ]

    monitor._check_equipment_metrics(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder")

    assert "'min' 閾值 (invalid_min) 不是有效數字" in caplog.text
    # Metric should still be evaluated (Pressure 1.0 is between None and 1.5 atm), so no alert
    mock_send_alert.assert_not_called()
    assert not any("INSERT INTO alert_history" in c.args[0] for c in mock_cursor.execute.call_args_list)


# --- Tests for _check_operation_status ---
@patch("src.equipment_monitor.EquipmentMonitor._send_alert_notification")
def test_check_operation_status_normal(mock_send_alert, monitor):
    mock_cursor = monitor._cursor_mock
    start_time = datetime.now() - timedelta(hours=1)
    mock_cursor.fetchall.return_value = [
        (1, "processing_lot", start_time.isoformat(), "Lot123", "ProdX")
    ]

    monitor._check_operation_status(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder") # Max duration 6 hours for die_bonder

    mock_send_alert.assert_not_called()
    assert not any("INSERT INTO alert_history" in c.args[0] for c in mock_cursor.execute.call_args_list)

@patch("src.equipment_monitor.EquipmentMonitor._send_alert_notification")
def test_check_operation_status_long_running(mock_send_alert, monitor):
    mock_cursor = monitor._cursor_mock
    start_time = datetime.now() - timedelta(hours=7) # Longer than 6 hours for die_bonder
    mock_cursor.fetchall.return_value = [
        (1, "processing_lot", start_time.isoformat(), "Lot123", "ProdX")
    ]

    monitor._check_operation_status(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder")

    mock_send_alert.assert_called_once()
    alert_message = mock_send_alert.call_args[0][1] # message is the second arg
    assert "運行已超預期時間" in alert_message
    assert "die_bonder" not in alert_message # Equipment type not in user message
    assert "Bonder01 (EQ001) 的作業 processing_lot (ID: 1)" in alert_message # Check for specific details

    assert any("INSERT INTO alert_history" in c.args[0] for c in mock_cursor.execute.call_args_list)
    monitor._conn_mock.commit.assert_called_once()


def test_check_operation_status_invalid_start_time(monitor, caplog):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchall.return_value = [
        (1, "processing_lot", "not-a-valid-iso-datetime", "Lot123", "ProdX")
    ]

    monitor._check_operation_status(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder")

    assert "start_time (not-a-valid-iso-datetime) 格式無效" in caplog.text
    assert not any("INSERT INTO alert_history" in c.args[0] for c in mock_cursor.execute.call_args_list)

def test_check_operation_status_unknown_equipment_type(monitor, caplog):
    mock_cursor = monitor._cursor_mock
    start_time = datetime.now() - timedelta(hours=9) # Longer than default 8 hours
    mock_cursor.fetchall.return_value = [
        (1, "processing_lot", start_time.isoformat(), "Lot123", "ProdX")
    ]

    monitor._check_operation_status(monitor._conn_mock, "EQ001", "Bonder01", "unknown_type")

    assert "類型 unknown_type 未知，無法確定最大運行時間。將使用預設值 8 小時。" in caplog.text
    # Should still alert because 9 > 8 (default)
    assert any("INSERT INTO alert_history" in c.args[0] for c in mock_cursor.execute.call_args_list)


# --- Tests for _send_alert_notification ---
@patch("src.equipment_monitor.send_notification") # Mock the actual send_notification function
def test_send_alert_notification_warning_level(mock_send_notification_func, monitor):
    mock_cursor = monitor._cursor_mock
    # Setup side effects for multiple execute calls within _send_alert_notification
    # 1. Get subscribed users (filter by notification_level='all' for warning)
    # 2. Get equipment type
    # 3. Get responsible_area users and admins
    mock_cursor.fetchall.side_effect = [
        [("user_subscribed_all_level",)], # Subscribed with 'all'
        [("user_admin",)],                 # Admins (also fetched if type-responsible query runs)
    ]
    mock_cursor.fetchone.side_effect = [ # For equipment type
        ("die_bonder",)
    ]

    # Simulate that execute for responsible_area returns no one to simplify, admins picked up later
    # The execute calls are: subscribed users, equipment type, responsible_area users (+admins), then potentially just admins if no one else
    def execute_side_effect_warning(query, params=None):
        if "user_equipment_subscriptions" in query and "notification_level = 'all'" in query:
            mock_cursor.description = [("user_id",)] # Describe columns for fetchall
            return mock_cursor
        elif "SELECT e.type FROM equipment e WHERE e.equipment_id = ?" in query:
            mock_cursor.description = [("type",)]
            return mock_cursor
        elif "responsible_area = ? OR is_admin = 1" in query : # responsible + admins
            mock_cursor.description = [("user_id",)]
            # Return only admin to test distinct user logic, assuming admin also in general admin query
            return [("user_admin",)]
        elif "is_admin = 1" in query: # Fallback admin query
             mock_cursor.description = [("user_id",)]
             return [("user_admin",)]
        return mock_cursor # Default

    mock_cursor.execute.side_effect = execute_side_effect_warning

    monitor._send_alert_notification("EQ001", "Test Warning Message", EquipmentMonitor.SEVERITY_WARNING)

    # Expected: user_subscribed_all_level and user_admin (if distinct)
    # Based on current logic, responsible_area query also gets admins, so it might be just one call for them
    # Let's check calls to send_notification
    expected_calls = [
        call("user_subscribed_all_level", "⚠️ Test Warning Message"),
        call("user_admin", "⚠️ Test Warning Message")
    ]
    # Use set for call_args_list because order might not be guaranteed due to set usage for unique_users
    actual_calls = set(mock_send_notification_func.call_args_list)
    # Check if each expected call is in the actual calls
    for expected_call in expected_calls:
        assert expected_call in actual_calls
    assert mock_send_notification_func.call_count == len(expected_calls)


@patch("src.equipment_monitor.send_notification")
def test_send_alert_notification_critical_level(mock_send_notification_func, monitor):
    mock_cursor = monitor._cursor_mock
    # 1. Get subscribed users (no level filter for critical)
    # 2. Get equipment type
    # 3. Get responsible_area users and admins
    mock_cursor.fetchall.side_effect = [
        [("user_sub1",), ("user_sub2_critical_only",)], # All subscribers for this EQ
        [("user_responsible_type",), ("user_admin",)]  # Type responsible and admin
    ]
    mock_cursor.fetchone.return_value = ("die_bonder",) # Equipment type

    def execute_side_effect_critical(query, params=None):
        mock_cursor.description = [("user_id",)]
        if "user_equipment_subscriptions" in query and "notification_level" not in query: # Critical - no level filter
            return mock_cursor
        elif "SELECT e.type FROM equipment e WHERE e.equipment_id = ?" in query:
            mock_cursor.description = [("type",)]
            return mock_cursor
        elif "responsible_area = ? OR is_admin = 1" in query :
             return mock_cursor
        return mock_cursor
    mock_cursor.execute.side_effect = execute_side_effect_critical

    monitor._send_alert_notification("EQ002", "Test Critical Message", EquipmentMonitor.SEVERITY_CRITICAL)

    expected_users_notified = {"user_sub1", "user_sub2_critical_only", "user_responsible_type", "user_admin"}
    actual_users_notified = {c.args[0] for c in mock_send_notification_func.call_args_list}

    assert actual_users_notified == expected_users_notified
    for c_args in mock_send_notification_func.call_args_list:
        assert c_args[0][1] == "🔴 Test Critical Message" # Check message and emoji
    assert mock_send_notification_func.call_count == len(expected_users_notified)


@patch("src.equipment_monitor.send_notification")
def test_send_alert_notification_no_subscribers_only_admins(mock_send_notification_func, monitor):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchall.side_effect = [
        [], # No direct subscribers
        [("admin1",), ("admin2",)] # Admins (fetched via responsible_area OR is_admin = 1)
                                   # Or if type not found, then via direct admin query
    ]
    mock_cursor.fetchone.return_value = ("unknown_type",) # Equipment type

    # Simplified side effect for execute
    def execute_side_effect_admins(query, params=None):
        mock_cursor.description = [("user_id",)]
        if "user_equipment_subscriptions" in query: return mock_cursor # Returns empty from fetchall
        if "SELECT e.type FROM equipment" in query:
            mock_cursor.description = [("type",)]
            return mock_cursor
        if "responsible_area = ? OR is_admin = 1" in query: return mock_cursor # Returns admins from fetchall
        if "is_admin = 1" in query and "responsible_area" not in query: return mock_cursor # Fallback admin query
        return mock_cursor
    mock_cursor.execute.side_effect = execute_side_effect_admins

    monitor._send_alert_notification("EQ003", "Test Admin Only Message", EquipmentMonitor.SEVERITY_EMERGENCY)

    expected_users_notified = {"admin1", "admin2"}
    actual_users_notified = {c.args[0] for c in mock_send_notification_func.call_args_list}
    assert actual_users_notified == expected_users_notified
    assert mock_send_notification_func.call_count == len(expected_users_notified)
    assert mock_send_notification_func.call_args_list[0][0][1] == "🚨 Test Admin Only Message"


@patch("src.equipment_monitor.send_notification")
def test_send_alert_notification_no_users_found(mock_send_notification_func, monitor, caplog):
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchall.return_value = [] # No users from any query
    mock_cursor.fetchone.return_value = (None,) # No equipment type found

    monitor._send_alert_notification("EQ004", "Test No User Message", EquipmentMonitor.SEVERITY_WARNING)

    mock_send_notification_func.assert_not_called()
    assert "沒有找到任何使用者來接收設備 EQ004 (嚴重性: warning) 的警報。" in caplog.text


@patch("src.equipment_monitor.send_notification", side_effect=ImportError("Simulated ImportError"))
def test_send_alert_notification_import_error(mock_send_notification_import_error, monitor, caplog):
    # Need to ensure some users would be found to attempt calling send_notification
    mock_cursor = monitor._cursor_mock
    mock_cursor.fetchall.return_value = [("admin1",)]
    mock_cursor.fetchone.return_value = (None,)

    monitor._send_alert_notification("EQ005", "Test Import Error Message", EquipmentMonitor.SEVERITY_CRITICAL)
    assert "無法導入 send_notification 模組。通知發送失敗。" in caplog.text


# --- Test for check_all_equipment ---
@patch.object(EquipmentMonitor, "_check_equipment_metrics")
@patch.object(EquipmentMonitor, "_check_operation_status")
def test_check_all_equipment_success_flow(mock_check_op_status, mock_check_metrics, monitor):
    mock_cursor = monitor._cursor_mock
    # Simulate fetching two pieces of equipment
    mock_cursor.fetchall.return_value = [
        ("EQ001", "Bonder01", "die_bonder"),
        ("EQ002", "Dicer01", "dicer")
    ]

    monitor.check_all_equipment()

    # Check if _get_connection was called to start the process
    monitor._db_mock._get_connection.assert_called_once() # From the with statement in check_all_equipment

    # Check calls to helper methods
    expected_metrics_calls = [
        call(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder"),
        call(monitor._conn_mock, "EQ002", "Dicer01", "dicer")
    ]
    # Ensure correct arguments were passed to the mocked connection object
    mock_check_metrics.assert_has_calls(expected_metrics_calls, any_order=False)

    expected_op_status_calls = [
        call(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder"),
        call(monitor._conn_mock, "EQ002", "Dicer01", "dicer")
    ]
    mock_check_op_status.assert_has_calls(expected_op_status_calls, any_order=False)


@patch.object(EquipmentMonitor, "_check_equipment_metrics")
@patch.object(EquipmentMonitor, "_check_operation_status")
def test_check_all_equipment_db_connection_error_initial_fetch(mock_check_op_status, mock_check_metrics, monitor, caplog):
    # Simulate error when initially fetching list of equipment
    monitor._db_mock._get_connection.return_value.__enter__.side_effect = pyodbc.Error("Simulated DB Connection Error on initial fetch")

    monitor.check_all_equipment()

    assert "檢查所有設備時發生資料庫錯誤" in caplog.text
    mock_check_metrics.assert_not_called()
    mock_check_op_status.assert_not_called()

@patch.object(EquipmentMonitor, "_check_equipment_metrics", side_effect=Exception("Simulated error in metrics check"))
@patch.object(EquipmentMonitor, "_check_operation_status") # Keep this mocked
def test_check_all_equipment_error_during_metric_check(mock_check_op_status, mock_check_metrics_with_error, monitor, caplog):
    mock_cursor = monitor._cursor_mock
    # Simulate fetching one piece of equipment to enter the loop
    mock_cursor.fetchall.return_value = [("EQ001", "Bonder01", "die_bonder")]

    monitor.check_all_equipment()

    # The generic exception in check_all_equipment should catch this
    assert "檢查所有設備時發生非預期錯誤" in caplog.text
    assert "Simulated error in metrics check" in caplog.text # The specific error message
    mock_check_metrics_with_error.assert_called_once_with(monitor._conn_mock, "EQ001", "Bonder01", "die_bonder")
    # _check_operation_status for this item will be skipped because the error happens before it in the loop
    # and the main try-except block in check_all_equipment will catch it.
    mock_check_op_status.assert_not_called()
