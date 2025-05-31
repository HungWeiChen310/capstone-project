import logging
from datetime import datetime, timedelta
import pyodbc  # Added to resolve F821 and for type hinting if used
from database import db

logger = logging.getLogger(__name__)


class EquipmentMonitor:
    """半導體設備監控與異常偵測器"""

    # 設備類型常數
    DIE_BONDER = "die_bonder"  # 黏晶機
    WIRE_BONDER = "wire_bonder"  # 打線機
    DICER = "dicer"  # 切割機

    # 嚴重程度常數
    SEVERITY_WARNING = "warning"    # 警告
    SEVERITY_CRITICAL = "critical"  # 嚴重
    SEVERITY_EMERGENCY = "emergency"  # 緊急

    def __init__(self):
        self.db = db
        # 設備類型的中文名稱對應
        self.equipment_type_names = {
            self.DIE_BONDER: "黏晶機",
            self.WIRE_BONDER: "打線機",
            self.DICER: "切割機",
        }
        # 設備類型的關鍵指標對應
        self.equipment_metrics = {
            self.DIE_BONDER: ["溫度", "壓力", "Pick準確率", "良率", "運轉時間"],
            self.WIRE_BONDER: ["溫度", "壓力", "金絲張力", "良率", "運轉時間"],
            self.DICER: ["溫度", "轉速", "冷卻水溫", "切割精度", "良率", "運轉時間"],
        }

    def check_all_equipment(self):
        """檢查所有設備是否有異常"""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                # 取得所有活動中的設備
                cursor.execute(
                    "SELECT equipment_id, name, type FROM equipment WHERE status <> 'offline'"
                )
                equipments = cursor.fetchall()
                for equipment_id, name, equipment_type in equipments:
                    self._check_equipment_metrics(conn, equipment_id, name, equipment_type)
                    self._check_operation_status(conn, equipment_id, name, equipment_type)
            logger.info("所有設備檢查完成。")
        except pyodbc.Error as db_err:
            logger.exception(f"檢查所有設備時發生資料庫錯誤: {db_err}")
        except Exception as e:
            logger.exception(f"檢查所有設備時發生非預期錯誤: {e}")

    def _check_equipment_metrics(self, conn, equipment_id, name, equipment_type):
        """檢查設備的指標是否異常"""
        try:
            cursor = conn.cursor()
            # 取得該設備最新的監測指標
            cursor.execute(
                """
                SELECT metric_type, value, threshold_min, threshold_max, unit
                FROM equipment_metrics
                WHERE equipment_id = ?
                  AND timestamp > DATEADD(minute, -30, GETDATE())
                ORDER BY timestamp DESC
                """,
                (equipment_id,),
            )
            metrics = cursor.fetchall()

            latest_metrics = {}
            for metric_type, value, threshold_min, threshold_max, unit in metrics:
                if metric_type not in latest_metrics:
                    # Basic validation for value
                    if value is None:
                        logger.warning(
                            f"設備 {equipment_id} 的指標 {metric_type} 缺少 'value'，跳過此指標。"
                        )
                        continue
                    try:
                        num_value = float(value) # Ensure value is a number
                    except (ValueError, TypeError):
                        logger.warning(
                            f"設備 {equipment_id} 的指標 {metric_type} 的 'value' ({value}) 不是有效數字，跳過此指標。"
                        )
                        continue

                    latest_metrics[metric_type] = {
                        "value": num_value,
                        "min": threshold_min, # Will be checked for None later
                        "max": threshold_max, # Will be checked for None later
                        "unit": unit,
                    }

            anomalies = []
            for metric_type, data in latest_metrics.items():
                # data["value"] is already validated to be a float
                # Validate min/max if they are not None
                is_below_min = False
                if data["min"] is not None:
                    try:
                        min_val = float(data["min"])
                        if data["value"] < min_val:
                            is_below_min = True
                    except (ValueError, TypeError):
                        logger.warning(
                            f"設備 {equipment_id} 的指標 {metric_type} 的 'min' 閾值 ({data['min']}) 不是有效數字，此閾值將被忽略。"
                        )
                        data["min"] = None # Ignore invalid threshold

                is_above_max = False
                if data["max"] is not None:
                    try:
                        max_val = float(data["max"])
                        if data["value"] > max_val:
                            is_above_max = True
                    except (ValueError, TypeError):
                        logger.warning(
                            f"設備 {equipment_id} 的指標 {metric_type} 的 'max' 閾值 ({data['max']}) 不是有效數字，此閾值將被忽略。"
                        )
                        data["max"] = None # Ignore invalid threshold

                if is_below_min or is_above_max:
                    severity = self._determine_severity(
                        metric_type, data["value"],
                        float(data["min"]) if data["min"] is not None else None,
                        float(data["max"]) if data["max"] is not None else None
                    )
                    anomalies.append(
                        {
                            "metric": metric_type,
                            "value": data["value"],
                            "min": data["min"], # Already float or None
                            "max": data["max"], # Already float or None
                            "unit": data["unit"],
                            "severity": severity,
                        }
                    )

            if anomalies:
                highest_severity = max(
                    [a["severity"] for a in anomalies], key=self._severity_level, default=self.SEVERITY_WARNING
                )
                message = f"設備 {name} ({equipment_id}) 異常提醒:\n"
                for anomaly in anomalies:
                    if anomaly["min"] is not None and anomaly["value"] < float(anomaly["min"]):
                        message += (f"- {anomaly['metric']} 值 {anomaly['value']}"
                                    f" 低於最小閾值 {anomaly['min']} {anomaly['unit'] or ''} (嚴重性: {anomaly['severity']})\n")
                    elif anomaly["max"] is not None and anomaly["value"] > float(anomaly['max']):
                        message += (f"- {anomaly['metric']} 值 {anomaly['value']}"
                                    f" 超出最大閾值 {anomaly['max']} {anomaly['unit'] or ''} (嚴重性: {anomaly['severity']})\n")

                try:
                    equipment_data = self._get_equipment_data(conn, equipment_id) # Pass conn
                    ai_recommendation = self._generate_ai_recommendation(anomalies, equipment_data)
                    if ai_recommendation:
                        message += f"\nAI建議: {ai_recommendation}"
                except Exception as ai_err:
                    logger.error(f"為設備 {equipment_id} 產生 AI 建議時失敗: {ai_err}")

                try:
                    for anomaly in anomalies: # Assuming message is a summary for all anomalies
                        cursor.execute(
                            """
                            INSERT INTO alert_history (equipment_id, alert_type, severity, message)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                equipment_id,
                                f"metric_alert: {anomaly['metric']}", # More specific alert type
                                anomaly["severity"],
                                message, # Full message with all anomalies
                            ),
                        )

                    new_status = self.SEVERITY_WARNING # Default
                    if highest_severity == self.SEVERITY_CRITICAL:
                        new_status = "critical"
                    elif highest_severity == self.SEVERITY_EMERGENCY:
                        new_status = "emergency"

                    cursor.execute(
                        """
                        UPDATE equipment
                        SET status = ?, last_updated = GETDATE()
                        WHERE equipment_id = ? AND status <> ?
                        """, # Only update if status changed or to refresh timestamp for same warning
                        (new_status, equipment_id, new_status),
                    )
                    conn.commit()
                    self._send_alert_notification(equipment_id, message, highest_severity) # Uses its own db connection
                    logger.info(f"設備 {name} ({equipment_id}) 異常已記錄及通知。最高嚴重性: {highest_severity}")

                except pyodbc.Error as db_op_err:
                    conn.rollback() # Rollback on error during DB operations
                    logger.error(
                        f"記錄設備 {equipment_id} 的指標異常或更新狀態時發生資料庫錯誤: {db_op_err}"
                    )

        except pyodbc.Error as db_fetch_err:
            logger.error(
                f"檢查設備 {equipment_id} ({name}) 的指標時獲取數據庫連接或執行查詢失敗: {db_fetch_err}"
            )
        except Exception as e:
            logger.exception(
                f"檢查設備 {equipment_id} ({name}) 的指標時發生非預期錯誤: {e}"
            )


    def _check_operation_status(self, conn, equipment_id, name, equipment_type):
        """檢查設備運行狀態，包括長時間運行、異常停機等"""
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, operation_type, start_time, lot_id, product_id
                FROM equipment_operation_logs
                WHERE equipment_id = ? AND end_time IS NULL
                ORDER BY start_time ASC
                """,
                (equipment_id,),
            )
            operations = cursor.fetchall()

            if not operations:
                return

            for op_id, op_type, start_time_str, lot_id, product_id in operations:
                try:
                    # Ensure start_time_str is a string before replace
                    if not isinstance(start_time_str, str):
                         # Handle cases where start_time might be datetime obj already from some DB drivers
                        if isinstance(start_time_str, datetime):
                            start_datetime = start_time_str
                        else:
                            raise TypeError("start_time 不是有效的字串或日期時間物件。")
                    else:
                        start_datetime = datetime.fromisoformat(
                            start_time_str.replace("Z", "+00:00") if "Z" in start_time_str else start_time_str
                        )
                except (ValueError, TypeError) as time_err:
                    logger.error(
                        f"設備 {equipment_id} 的作業 {op_id} 的 start_time ({start_time_str}) 格式無效: {time_err}。跳過此作業檢查。"
                    )
                    continue

                current_time = datetime.now()
                operation_duration = current_time - start_datetime

                if equipment_type not in self.equipment_metrics: # Using self.equipment_metrics as a proxy for known types
                    logger.warning(
                        f"設備 {equipment_id} 的類型 {equipment_type} 未知，無法確定最大運行時間。將使用預設值 8 小時。"
                    )
                max_duration_hours = {
                    self.DIE_BONDER: 6,
                    self.WIRE_BONDER: 8,
                    self.DICER: 4,
                }.get(equipment_type, 8) # Default 8 hours if type unknown

                if operation_duration > timedelta(hours=max_duration_hours):
                    severity = self.SEVERITY_WARNING
                    message = (f"設備 {name} ({equipment_id}) 的作業 {op_type} (ID: {op_id}) "
                               f"已運行 {operation_duration}，超過預期的 {max_duration_hours} 小時，請注意檢查。")
                    try:
                        cursor.execute(
                            """
                            INSERT INTO alert_history (equipment_id, alert_type, severity, message)
                            VALUES (?, ?, ?, ?)
                            """,
                            (equipment_id, "operation_long_running", severity, message),
                        )
                        conn.commit()
                        self._send_alert_notification(equipment_id, message, severity) # Uses its own connection
                        logger.info(
                            f"設備 {name} ({equipment_id}) 長時間運行異常已通知 (作業ID: {op_id})。"
                        )
                        # return # Original code returns after first long op, consider if all should be checked
                    except pyodbc.Error as db_alert_err:
                        conn.rollback()
                        logger.error(
                            f"記錄設備 {equipment_id} 的長時間運行警報時發生資料庫錯誤: {db_alert_err}"
                        )
                    except Exception as e_alert: # Catch error during notification
                        logger.error(
                            f"發送設備 {equipment_id} 長時間運行警報通知時發生錯誤: {e_alert}"
                        )
                    return # Assuming we alert for the first long-running op and then stop for this equipment_id

        except pyodbc.Error as db_op_err:
            logger.error(
                f"檢查設備 {equipment_id} ({name}) 的運行狀態時獲取數據庫連接或執行查詢失敗: {db_op_err}"
            )
        except Exception as e:
            logger.exception(
                f"檢查設備 {equipment_id} ({name}) 的運行狀態時發生非預期錯誤: {e}"
            )


    def _determine_severity(self, metric_type, value, threshold_min, threshold_max):
        # Input validation
        if not isinstance(value, (int, float)):
            logger.warning(
                f"嚴重性判斷收到無效的 'value' ({value}) (指標: {metric_type})。預設為 'warning'。"
            )
            return self.SEVERITY_WARNING
        if threshold_min is not None and not isinstance(threshold_min, (int, float)):
            logger.warning(
                f"嚴重性判斷收到無效的 'threshold_min' ({threshold_min}) (指標: {metric_type})。此閾值將被忽略。"
            )
            threshold_min = None
        if threshold_max is not None and not isinstance(threshold_max, (int, float)):
            logger.warning(
                f"嚴重性判斷收到無效的 'threshold_max' ({threshold_max}) (指標: {metric_type})。此閾值將被忽略。"
            )
            threshold_max = None

        try:
            if metric_type in ["溫度", "壓力", "轉速"]:
                if threshold_max is not None and value >= threshold_max * 1.2:
                    return self.SEVERITY_EMERGENCY
                elif threshold_max is not None and value >= threshold_max * 1.1:
                    return self.SEVERITY_CRITICAL
                # Check for below min if max is not defined or not exceeded
                elif threshold_min is not None and value < threshold_min:
                     return self.SEVERITY_WARNING # Or critical depending on policy
                elif threshold_max is not None and value > threshold_max: # Exceeded max but not by 1.1x
                    return self.SEVERITY_WARNING
                elif threshold_min is None and threshold_max is None: # No thresholds defined
                    return self.SEVERITY_WARNING # Or log as info/debug
                else: # Within normal range or only one threshold defined and not breached
                    return self.SEVERITY_WARNING # Default for any deviation if not emergency/critical
            elif metric_type in ["良率", "Pick準確率", "切割精度"]:
                if threshold_min is not None and value <= threshold_min * 0.8: # Significantly below min
                    return self.SEVERITY_CRITICAL
                elif threshold_min is not None and value < threshold_min: # Below min but not significantly
                    return self.SEVERITY_WARNING
                elif threshold_max is not None and value > threshold_max: # Exceeded max if defined
                    return self.SEVERITY_WARNING
                elif threshold_min is None and threshold_max is None:
                    return self.SEVERITY_WARNING
                else: # Within normal range
                    return self.SEVERITY_WARNING # Default for any deviation
            else: # Unknown metric type
                logger.info(f"指標 {metric_type} 收到未知類型，預設嚴重性為 'warning'。")
                return self.SEVERITY_WARNING
        except Exception as e:
            logger.error(f"計算嚴重性時發生錯誤 (指標: {metric_type}, 值: {value}): {e}")
            return self.SEVERITY_WARNING


    def _severity_level(self, severity):
        levels = {
            self.SEVERITY_WARNING: 1,
            self.SEVERITY_CRITICAL: 2,
            self.SEVERITY_EMERGENCY: 3,
        }
        if severity not in levels:
            logger.warning(f"收到未知的嚴重性級別: {severity}。預設為 0。")
        return levels.get(severity, 0)

    def _severity_emoji(self, severity):
        emojis = {
            self.SEVERITY_WARNING: "⚠️",
            self.SEVERITY_CRITICAL: "🔴",
            self.SEVERITY_EMERGENCY: "🚨",
        }
        emojis = {
            self.SEVERITY_WARNING: "⚠️",
            self.SEVERITY_CRITICAL: "🔴",
            self.SEVERITY_EMERGENCY: "🚨",
        }
        if severity not in emojis:
            logger.warning(f"收到未知的嚴重性表情符號請求: {severity}。預設為 '⚠️'。")
        return emojis.get(severity, "⚠️")

    def _get_equipment_data(self, conn, equipment_id):
        """Helper to get equipment data using a passed connection."""
        # conn is expected to be an active pyodbc connection
        if conn is None:
            logger.error(f"為設備 {equipment_id} 取得設備資料時收到空的資料庫連線。")
            # Fallback to creating a new connection if conn is None, though this is not ideal design
            try:
                with self.db._get_connection() as new_conn:
                    return self._execute_get_equipment_data(new_conn, equipment_id)
            except pyodbc.Error as db_err:
                logger.error(f"為設備 {equipment_id} 取得設備資料時無法建立新的資料庫連線: {db_err}")
                return self._default_equipment_data(equipment_id, "Fallback connection error")
        try:
            return self._execute_get_equipment_data(conn, equipment_id)
        except pyodbc.Error as db_err:
            logger.error(
                f"為設備 {equipment_id} 取得設備資料時發生資料庫錯誤: {db_err}"
            )
            return self._default_equipment_data(equipment_id, str(db_err))
        except Exception as e:
            logger.exception(
                f"為設備 {equipment_id} 取得設備資料時發生非預期錯誤: {e}"
            )
            return self._default_equipment_data(equipment_id, str(e))

    def _execute_get_equipment_data(self, conn_to_use, equipment_id):
        """Executes the database query for _get_equipment_data."""
        cursor = conn_to_use.cursor()
        cursor.execute(
            """
            SELECT name, type, location
            FROM equipment
            WHERE equipment_id = ?
            """,
            (equipment_id,),
        )
        result = cursor.fetchone()
        if result:
            return {
                "name": result[0],
                "type": result[1],
                "type_name": self.equipment_type_names.get(result[1], result[1]), # Use stored mapping
                "location": result[2] or "未指定", # Handle None location
            }
        else:
            logger.warning(f"資料庫中找不到設備 ID: {equipment_id}。回傳預設資料。")
            return self._default_equipment_data(equipment_id, "Not found in DB")

    def _default_equipment_data(self, equipment_id, reason=""):
        return {
            "name": f"未知設備 ({equipment_id})",
            "type": "未知類型",
            "type_name": "未知設備",
            "location": "未知地點",
            "error_reason": reason
        }

    def _generate_ai_recommendation(self, anomalies, equipment_data):
        """產生 AI 增強的異常描述和建議（使用現有的 OpenAI 服務）"""
        if not anomalies: # No anomalies, no recommendation needed
            return None
        try:
            from src.main import OpenAIService # Local import to avoid circular dependency issues at module level

            # Construct a more detailed context
            context = "偵測到的設備異常狀況:\n"
            eq_name = equipment_data.get('name', '未知設備')
            eq_type = equipment_data.get('type_name', '未知類型')
            eq_loc = equipment_data.get('location', '未知地點')
            context += f"設備: {eq_name} ({eq_type}) 位於 {eq_loc}\n"

            for anomaly in anomalies:
                metric_desc = (f"- 指標 '{anomaly['metric']}': 目前值 {anomaly['value']:.2f} "
                               f"{anomaly['unit'] or ''}. ")
                if anomaly["min"] is not None and anomaly["max"] is not None:
                    metric_desc += f"正常範圍: {anomaly['min']} - {anomaly['max']}. "
                elif anomaly["min"] is not None:
                    metric_desc += f"應高於 {anomaly['min']}. "
                elif anomaly["max"] is not None:
                    metric_desc += f"應低於 {anomaly['max']}. "
                metric_desc += f"嚴重性: {anomaly['severity']}.\n"
                context += metric_desc

            prompt = (f"作為一個半導體設備維護專家，請根據以下監測到的異常狀況，為設備 {eq_name} 提供簡潔的初步分析和處理建議。"
                      f"請著重於最可能的根本原因和應立即採取的檢查步驟。\n\n{context}")

            # Assuming OpenAIService is set up correctly
            service = OpenAIService(message=prompt, user_id=f"system-monitor-{equipment_data.get('equipment_id', 'unknown')}")
            response = service.get_response()

            if response and "無法提供建議" not in response and "不相關" not in response:
                return response.strip()
            else:
                logger.info(f"AI 未能為設備 {equipment_data.get('equipment_id')} 的異常提供有效建議。")
                return None

        except ImportError: # Handled in main.py, but good for direct test
            logger.error("無法導入 OpenAIService，無法產生 AI 建議。")
            return None
        except Exception as e:
            logger.error(f"為設備 {equipment_data.get('equipment_id')} 產生 AI 建議時發生錯誤: {e}")
            return None


    def _send_alert_notification(self, equipment_id, message, severity):
        """發送通知給負責該設備的使用者"""
        if not equipment_id or not message:
            logger.error("發送通知時缺少 equipment_id 或 message。")
            return

        try:
            from src.linebot_connect import send_notification # Local import

            users_to_notify = set()
            with self.db._get_connection() as conn:
                cursor = conn.cursor()

                # Get users subscribed to this equipment
                query_subscribed = """
                    SELECT user_id FROM user_equipment_subscriptions
                    WHERE equipment_id = ?
                """
                params_subscribed = [equipment_id]
                if severity == self.SEVERITY_WARNING: # For warnings, only 'all' level
                    query_subscribed += " AND notification_level = 'all'"

                try:
                    cursor.execute(query_subscribed, tuple(params_subscribed))
                    subscribed_users = cursor.fetchall()
                    for (user_id,) in subscribed_users:
                        if user_id and isinstance(user_id, str): # Validate user_id
                            users_to_notify.add(user_id)
                        else:
                            logger.warning(f"從訂閱中獲取到無效的 user_id: {user_id} (設備: {equipment_id})")
                except pyodbc.Error as db_err_sub:
                    logger.error(f"查詢設備 {equipment_id} 的訂閱用戶時發生資料庫錯誤: {db_err_sub}")

                # Get users responsible for this equipment type or admins
                try:
                    cursor.execute("SELECT type FROM equipment WHERE equipment_id = ?", (equipment_id,))
                    eq_type_result = cursor.fetchone()
                    if eq_type_result and eq_type_result[0]:
                        equipment_type_str = eq_type_result[0]
                        cursor.execute(
                            """
                            SELECT user_id FROM user_preferences
                            WHERE responsible_area = ? OR is_admin = 1
                            """,
                            (equipment_type_str,),
                        )
                        responsible_and_admins = cursor.fetchall()
                        for (user_id,) in responsible_and_admins:
                            if user_id and isinstance(user_id, str): # Validate user_id
                                users_to_notify.add(user_id)
                            else:
                                logger.warning(f"從負責區域/管理員中獲取到無效的 user_id: {user_id} (設備類型: {equipment_type_str})")
                    else:
                        logger.warning(f"無法獲取設備 {equipment_id} 的類型，無法通知負責區域用戶。")
                        # Fallback to notify all admins if equipment type not found
                        cursor.execute("SELECT user_id FROM user_preferences WHERE is_admin = 1")
                        admin_users = cursor.fetchall()
                        for (user_id,) in admin_users:
                             if user_id and isinstance(user_id, str): users_to_notify.add(user_id)

                except pyodbc.Error as db_err_resp:
                    logger.error(f"查詢設備 {equipment_id} 的負責用戶或管理員時發生資料庫錯誤: {db_err_resp}")
                    # Attempt to notify at least admins if other queries fail
                    if not users_to_notify:
                         try:
                            cursor.execute("SELECT user_id FROM user_preferences WHERE is_admin = 1")
                            admin_users = cursor.fetchall()
                            for (user_id,) in admin_users:
                                if user_id and isinstance(user_id, str): users_to_notify.add(user_id)
                         except pyodbc.Error as db_err_admin:
                             logger.error(f"查詢管理員用戶以進行回退通知時發生資料庫錯誤: {db_err_admin}")


            if not users_to_notify:
                logger.warning(f"沒有找到任何使用者來接收設備 {equipment_id} (嚴重性: {severity}) 的警報。")
                return

            full_message = f"{self._severity_emoji(severity)} {message}"
            for user_id_to_notify in users_to_notify:
                try:
                    if not send_notification(user_id_to_notify, full_message):
                         logger.error(f"向使用者 {user_id_to_notify} 發送設備 {equipment_id} 的通知失敗。")
                    else:
                        logger.info(f"設備 {equipment_id} 的警報通知已成功發送給使用者: {user_id_to_notify}。")
                except Exception as notify_err: # Catch errors from send_notification call itself
                    logger.error(f"向使用者 {user_id_to_notify} 發送設備 {equipment_id} 通知時發生例外: {notify_err}")

        except ImportError:
            logger.error("無法導入 send_notification 模組。通知發送失敗。")
        except pyodbc.Error as db_err_main: # Catch error if _get_connection itself fails
             logger.critical(f"發送設備 {equipment_id} 通知時無法連接資料庫: {db_err_main}")
        except Exception as e:
            logger.exception(f"發送設備 {equipment_id} 的通知時發生未預期的嚴重錯誤: {e}")
