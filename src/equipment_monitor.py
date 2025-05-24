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
        # 按指標類型分組，只取每種類型的最新值
        latest_metrics = {}
        for metric_type, value, threshold_min, threshold_max, unit in metrics:
            if metric_type not in latest_metrics:
                latest_metrics[metric_type] = {
                    "value": value,
                    "min": threshold_min,
                    "max": threshold_max,
                    "unit": unit,
                }
        # 檢查是否有異常
        anomalies = []
        for metric_type, data in latest_metrics.items():
            if (data["min"] is not None and data["value"] < data["min"]) or (
                data["max"] is not None and data["value"] > data["max"]
            ):
                # 根據指標決定嚴重程度
                severity = self._determine_severity(
                    metric_type, data["value"], data["min"], data["max"]
                )
                anomalies.append(
                    {
                        "metric": metric_type,
                        "value": data["value"],
                        "min": data["min"],
                        "max": data["max"],
                        "unit": data["unit"],
                        "severity": severity,
                    }
                )
        if anomalies:
            highest_severity = max(
                [a["severity"] for a in anomalies], key=self._severity_level
            )
            message = f"設備 {name} 異常提醒:\n"
            for anomaly in anomalies:
                if anomaly["min"] is not None and anomaly["value"] < anomaly["min"]:
                    message += (f"{anomaly['metric']} 值 {anomaly['value']}"
                                f" 低於最小閾值 {anomaly['min']} {anomaly['unit'] or ''}\n")
                elif anomaly["max"] is not None and anomaly["value"] > anomaly["max"]:
                    message += (f"{anomaly['metric']} 值 {anomaly['value']}"
                                f" 超出最大閾值 {anomaly['max']} {anomaly['unit'] or ''}\n")
            # 生成 AI 分析建議（選用）
            if hasattr(self, "_generate_ai_recommendation"):
                equipment_data = self._get_equipment_data(conn, equipment_id)
                ai_recommendation = self._generate_ai_recommendation(anomalies, equipment_data)
                if ai_recommendation:
                    message += f"\n建議: {ai_recommendation}"
            # 記錄此警告
            for anomaly in anomalies:
                cursor.execute(
                    """
                    INSERT INTO alert_history (equipment_id, alert_type, severity, message)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        equipment_id,
                        "metric_alert",
                        anomaly["severity"],
                        message,
                    ),
                )
            # 更新設備狀態
            new_status = "warning"
            if highest_severity == self.SEVERITY_CRITICAL:
                new_status = "critical"
            elif highest_severity == self.SEVERITY_EMERGENCY:
                new_status = "emergency"
            cursor.execute(
                """
                UPDATE equipment
                SET status = ?, last_updated = GETDATE()
                WHERE equipment_id = ?
                """,
                (new_status, equipment_id),
            )
            conn.commit()
            # 發送 LINE 通知給相關使用者
            self._send_alert_notification(equipment_id, message, highest_severity)
            logger.info("設備 %s 異常已記錄及通知。", name)

    def _check_operation_status(self, conn, equipment_id, name, equipment_type):
        """檢查設備運行狀態，包括長時間運行、異常停機等"""
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
            return  # 無運行中的作業
        for op_id, op_type, start_time, lot_id, product_id in operations:
            start_datetime = datetime.fromisoformat(
                start_time.replace("Z", "+00:00") if "Z" in start_time else start_time
            )
            current_time = datetime.now()
            operation_duration = current_time - start_datetime
            # 根據設備類型決定的操作最大運行時間 (以小時為單位)
            max_duration_hours = {
                self.DIE_BONDER: 6,
                self.WIRE_BONDER: 8,
                self.DICER: 4,
            }.get(equipment_type, 8)
            if operation_duration > timedelta(hours=max_duration_hours):
                severity = self.SEVERITY_WARNING
                message = (f"設備 {name} 運行已超預期時間 {operation_duration}，"
                           f"請注意檢查。")
                cursor.execute(
                    """
                    INSERT INTO alert_history (equipment_id, alert_type, severity, message)
                    VALUES (?, ?, ?, ?)
                    """,
                    (equipment_id, "operation_long_running", severity, message),
                )
                conn.commit()
                self._send_alert_notification(equipment_id, message, severity)
                logger.info("設備 %s 長時間運行異常已通知。", name)
                return

    def _determine_severity(self, metric_type, value, threshold_min, threshold_max):
        if metric_type in ["溫度", "壓力", "轉速"]:
            if threshold_max and value >= threshold_max * 1.2:
                return self.SEVERITY_EMERGENCY
            elif threshold_max and value >= threshold_max * 1.1:
                return self.SEVERITY_CRITICAL
            else:
                return self.SEVERITY_WARNING
        elif metric_type in ["良率", "Pick準確率", "切割精度"]:
            if threshold_min and value <= threshold_min * 0.8:
                return self.SEVERITY_CRITICAL
            else:
                return self.SEVERITY_WARNING
        else:
            return self.SEVERITY_WARNING

    def _severity_level(self, severity):
        levels = {
            self.SEVERITY_WARNING: 1,
            self.SEVERITY_CRITICAL: 2,
            self.SEVERITY_EMERGENCY: 3,
        }
        return levels.get(severity, 0)

    def _severity_emoji(self, severity):
        emojis = {
            self.SEVERITY_WARNING: "⚠️",
            self.SEVERITY_CRITICAL: "🔴",
            self.SEVERITY_EMERGENCY: "🚨",
        }
        return emojis.get(severity, "⚠️")

    def _get_equipment_data(self, conn, equipment_id):
        cursor = conn.cursor()
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
                "type_name": self.equipment_type_names.get(result[1], result[1]),
                "location": result[2],
            }
        return {
            "name": "未知",
            "type": "未知",
            "type_name": "未知設備",
            "location": "未知",
        }

    def _generate_ai_recommendation(self, anomalies, equipment_data):
        """產生 AI 增強的異常描述和建議（使用現有的 OpenAI 服務）"""
        try:
            from src.main import OpenAIService
            context = "偵測到的異常狀況:\n"
            for anomaly in anomalies:
                context += (f"{anomaly['metric']}: {anomaly['value']} "
                            f"(閾值: {anomaly['min']} - {anomaly['max']}, "
                            f"單位: {anomaly['unit'] or '無'})\n")
            prompt = (f"請根據以下資料提供建議：\n設備資料：{equipment_data}\n"
                      f"異常狀況：\n{context}")
            service = OpenAIService(message=prompt, user_id="system")
            response = service.get_response()
            return response
        except ImportError as imp_err:
            logger.error(f"無法導入 OpenAIService: {imp_err}")
            return None
        except Exception as e:  # Keep a general Exception for OpenAI client errors for now
            logger.exception(f"產生 AI 建議時發生錯誤: {e}")
            return None

    def _send_alert_notification(self, equipment_id, message, severity):
        """發送通知給負責該設備的使用者"""
        try:
            from src.linebot_connect import send_notification
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                if severity == self.SEVERITY_WARNING:
                    cursor.execute(
                        """
                        SELECT user_id FROM user_equipment_subscriptions
                        WHERE equipment_id = ? AND notification_level = 'all'
                        """,
                        (equipment_id,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT user_id FROM user_equipment_subscriptions
                        WHERE equipment_id = ?
                        """,
                        (equipment_id,),
                    )
                users = cursor.fetchall()
                cursor.execute(
                    """
                    SELECT e.type FROM equipment e WHERE e.equipment_id = ?
                    """,
                    (equipment_id,),
                )
                equipment_type = cursor.fetchone()
                if equipment_type:
                    cursor.execute(
                        """
                        SELECT user_id FROM user_preferences
                        WHERE responsible_area = ? OR is_admin = 1
                        """,
                        (equipment_type[0],),
                    )
                    responsible_users = cursor.fetchall()
                    users.extend(responsible_users)
                unique_users = set(user_id for (user_id,) in users)
                if not unique_users:
                    cursor.execute(
                        """
                        SELECT user_id FROM user_preferences
                        WHERE is_admin = 1
                        """
                    )
                    admin_users = cursor.fetchall()
                    unique_users = set(user_id for (user_id,) in admin_users)
                for user_id in unique_users:
                    send_notification(user_id, message)
                    logger.info("通知已發送給使用者: %s", user_id)
        except pyodbc.Error as db_err:
            logger.exception(f"發送設備 {equipment_id} 的通知時發生資料庫錯誤: {db_err}")
        except ImportError as imp_err:
            logger.error(f"無法導入 send_notification: {imp_err}")
        except Exception as e:
            logger.exception(f"發送設備 {equipment_id} 的通知時發生非預期錯誤: {e}")
