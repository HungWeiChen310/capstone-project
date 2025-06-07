import logging
import pyodbc
from database import db

logger = logging.getLogger(__name__)


class EquipmentMonitor:
    """半導體設備監控與異常偵測器 (僅限切割機)"""

    # 設備類型常數 (只保留切割機)
    DICER = "dicer"  # 切割機

    # 嚴重程度常數
    SEVERITY_WARNING = "warning"  # 警告
    SEVERITY_CRITICAL = "critical"  # 嚴重
    SEVERITY_EMERGENCY = "emergency"  # 緊急

    def __init__(self):
        self.db = db
        self.equipment_type_names = {
            self.DICER: "切割機",
        }
        # 這些指標現在會從資料庫的 equipment_metric_thresholds 表中獲取標準
        self.equipment_metrics = {
            self.DICER: ["變形量(mm)", "轉速", "刀具裂痕"],  # 增加刀具裂痕
        }
        # 用於儲存從資料庫載入的閾值
        self.metric_thresholds_data = {}
        self._load_metric_thresholds_from_db()  # 初始化時從資料庫載入標準

    def _load_metric_thresholds_from_db(self):
        """從資料庫的 equipment_metric_thresholds 表中載入所有指標的閾值。"""
        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        metric_type, normal_value,
                        warning_min, warning_max,
                        critical_min, critical_max,
                        emergency_min, emergency_max, emergency_op
                    FROM equipment_metric_thresholds;
                """)
                rows = cursor.fetchall()
                if not rows:
                    logger.warning("資料庫中 equipment_metric_thresholds 表無閾值數據。")

                for row in rows:
                    (metric_type, normal_value,
                     w_min, w_max,
                     c_min, c_max,
                     e_min, e_max, e_op) = row

                    self.metric_thresholds_data[metric_type] = {
                        "normal_value": normal_value,
                        "warning": {"min": w_min, "max": w_max},
                        "critical": {"min": c_min, "max": c_max},
                        "emergency": {"min": e_min, "max": e_max, "op": e_op}
                    }
                logger.info(f"成功從資料庫載入 {len(self.metric_thresholds_data)} 個指標的閾值。")
        except pyodbc.Error as db_err:
            logger.exception(f"從資料庫載入閾值時發生錯誤: {db_err}")
            self.metric_thresholds_data = {}  # 清空，避免使用不完整的數據
        except Exception as e:
            logger.exception(f"載入閾值時發生非預期錯誤: {e}")
            self.metric_thresholds_data = {}

    def check_all_equipment(self):
        """檢查所有切割機設備是否有異常"""
        # 在每次檢查前重新載入閾值，以確保是最新的（如果資料庫有更新）
        self._load_metric_thresholds_from_db()

        try:
            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT equipment_id, name, eq_type FROM equipment "
                    "WHERE status <> 'offline' AND eq_type = ?;",
                    (self.DICER,)
                )
                equipments = cursor.fetchall()
                for equipment_id, name, eq_type in equipments:
                    self._check_equipment_metrics(
                        conn, equipment_id, name, eq_type
                    )
                logger.info("所有切割機設備檢查完成。")
        except pyodbc.Error as db_err:
            logger.exception(f"檢查所有切割機設備時發生資料庫錯誤: {db_err}")
        except Exception as e:
            logger.exception(f"檢查所有切割機設備時發生非預期錯誤: {e}")

    def _check_equipment_metrics(self, conn, equipment_id, name, eq_type):
        """檢查設備的指標是否異常"""
        try:
            cursor = conn.cursor()
            # SQL 查詢修改：只選擇實際會用到的欄位，移除 threshold_min/max 因為它們從另一張表獲取
            sql_get_metrics = """
                WITH RankedMetrics AS (
                    SELECT
                        id, equipment_id, metric_type, status, value, unit, timestamp,
                        ROW_NUMBER() OVER(
                            PARTITION BY equipment_id, metric_type
                            ORDER BY timestamp DESC
                        ) as rn
                    FROM equipment_metrics
                    WHERE equipment_id = ? AND timestamp > DATEADD(minute, -30, GETDATE())
                )
                SELECT id, equipment_id, metric_type, status, value, unit, timestamp
                FROM RankedMetrics
                WHERE rn = 1;
            """
            cursor.execute(sql_get_metrics, (equipment_id,))

            latest_metrics = {}
            for metric_row in cursor.fetchall():
                # 解包需要匹配新的 SELECT 順序
                _id, _eq_id, metric_type, status, value, unit, ts = metric_row
                # 這裡的 latest_metrics key 使用 metric_type
                latest_metrics[metric_type] = {
                    "value": float(value) if value is not None else None,
                    "unit": unit,
                    "timestamp": ts,
                    "status_from_metric": status  # 儲存指標自身的狀態
                }

            anomalies = []
            if not latest_metrics:
                logger.debug(
                    f"設備 {name} ({equipment_id}) 在過去30分鐘內沒有新的監測指標。"
                )
                # 可考慮長時間無數據回報的處理邏輯
                return

            for metric_type, data in latest_metrics.items():
                # 只處理 self.equipment_metrics 中為 DICER 定義的指標
                if metric_type in self.equipment_metrics.get(self.DICER, []) and data["value"] is not None:
                    # 使用從資料庫載入的閾值數據進行判斷
                    severity = self._determine_severity(
                        metric_type, data["value"]
                    )

                    # 只有在判斷出非 None 的嚴重程度時才加入 anomalies
                    if severity:
                        anomalies.append({
                            "metric": metric_type,
                            "value": data["value"],
                            "unit": data["unit"],
                            "severity": severity,
                            "timestamp": data["timestamp"]
                        })

            if anomalies:
                highest_severity = max(
                    (a["severity"] for a in anomalies),
                    key=self._severity_level,
                    default=self.SEVERITY_WARNING
                )
                anomaly_messages = []
                for anomaly in anomalies:
                    ts_str = (
                        anomaly['timestamp'].strftime('%H:%M:%S')
                        if anomaly.get('timestamp') else 'N/A'
                    )
                    msg = ""
                    # 從 self.metric_thresholds_data 獲取正常值以供顯示
                    normal_val_display = self.metric_thresholds_data.get(
                        anomaly["metric"], {}
                    ).get("normal_value")

                    if anomaly["metric"] == "轉速":
                        msg = (
                            f"指標 {anomaly['metric']} 值 {anomaly['value']:.0f} RPM "
                            f"(正常應為 {normal_val_display:.0f} RPM 左右)"
                            if normal_val_display is not None else ""
                        ) + (
                            f"。偵測為 {anomaly['severity'].upper()} 等級異常 (於 {ts_str})"
                        )
                    elif anomaly["metric"] == "變形量(mm)":
                        msg = (
                            f"指標 {anomaly['metric']} 值 {anomaly['value']:.3f} mm"
                            f"(正常應為 {normal_val_display:.3f} mm 以下)"
                            if normal_val_display is not None else ""
                        ) + (
                            f"。偵測為 {anomaly['severity'].upper()} 等級異常 (於 {ts_str})"
                        )
                    elif anomaly["metric"] == "刀具裂痕":  # 新增刀具裂痕的訊息格式
                        msg = (
                            f"指標 {anomaly['metric']} 值 {anomaly['value']:.3f} mm"
                            f"(正常應為 {normal_val_display:.3f} mm 以下)"
                            if normal_val_display is not None else ""
                        ) + (
                            f"。偵測為 {anomaly['severity'].upper()} 等級異常 (於 {ts_str})"
                        )
                    else:
                        msg = (
                            f"指標 {anomaly['metric']} 值 {anomaly['value']:.2f} {anomaly['unit'] or ''}。"
                            f"偵測為 {anomaly['severity'].upper()} 等級異常 (於 {ts_str})"
                        )
                    anomaly_messages.append(msg)

                full_message = (
                    f"設備 {name} ({equipment_id}) 異常提醒 "
                    f"({self._severity_emoji(highest_severity)} {highest_severity.upper()}):\n"
                    + "\n".join(anomaly_messages)
                )

                for anomaly in anomalies:
                    alert_msg_for_db = (
                        f"{anomaly['metric']} 值 {anomaly['value']:.2f} {anomaly['unit'] or ''} "
                        f"(嚴重程度: {anomaly['severity'].upper()})"
                    )
                    cursor.execute(
                        """
                        INSERT INTO alert_history (equipment_id, alert_type, severity, message, created_at)
                        VALUES (?, ?, ?, ?, GETDATE()); -- created_at 自動填寫
                        """,
                        (
                            equipment_id,
                            f"{anomaly['metric']}_alert",
                            anomaly["severity"],
                            alert_msg_for_db
                        )
                    )

                self._update_equipment_status(
                    conn, equipment_id, highest_severity, full_message
                )
                conn.commit()
                self._send_alert_notification(equipment_id, full_message, highest_severity)
                logger.info(
                    f"設備 {name} ({equipment_id}) 異常已記錄及通知 ({highest_severity})。"
                )
            else:
                cursor.execute(
                    "SELECT status FROM equipment WHERE equipment_id = ?;", (equipment_id,)
                )
                current_status_row = cursor.fetchone()
                if current_status_row and current_status_row[0] not in ['normal', 'offline']:
                    logger.info(
                        f"設備 {name} ({equipment_id}) 指標已恢復正常，"
                        f"先前狀態為 {current_status_row[0]}。"
                    )
                    self._update_equipment_status(conn, equipment_id, "normal", "指標已恢復正常")
                    conn.commit()

        except pyodbc.Error as db_err:
            logger.error(
                f"檢查設備 {name} ({equipment_id}) 指標時發生資料庫錯誤: {db_err}"
            )
        except Exception as e:
            logger.error(
                f"檢查設備 {name} ({equipment_id}) 指標時發生未知錯誤: {e}"
            )

    def _update_equipment_status(
        self, conn, equipment_id, new_status_key, alert_message_for_log="狀態更新"
    ):
        """輔助函數：更新設備狀態並記錄到 alert_history (如果狀態改變)"""
        status_map = {
            self.SEVERITY_WARNING: "warning",
            self.SEVERITY_CRITICAL: "critical",
            self.SEVERITY_EMERGENCY: "emergency",
            "normal": "normal",
            "offline": "offline",
            "stale_data": "warning"
        }
        db_status = status_map.get(new_status_key, "warning")

        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM equipment WHERE equipment_id = ?;", (equipment_id,)
        )
        current_status_row = cursor.fetchone()

        if current_status_row and current_status_row[0] != db_status:
            cursor.execute(
                "UPDATE equipment SET status = ?, last_updated = GETDATE() "
                "WHERE equipment_id = ?;",
                (db_status, equipment_id)
            )
            if new_status_key == "normal" or db_status != current_status_row[0]:
                alert_type = (
                    "status_change" if new_status_key != "normal" else "recovery"
                )
                severity_for_log = (
                    new_status_key if new_status_key != "normal" else "info"
                )
                is_resolved_log = 1 if new_status_key == "normal" else 0
                cursor.execute(
                    """
                    INSERT INTO alert_history
                        (equipment_id, alert_type, severity, message, is_resolved, created_at)
                    VALUES (?, ?, ?, ?, ?, GETDATE());
                    """,
                    (
                        equipment_id,
                        alert_type,
                        severity_for_log,
                        alert_message_for_log,
                        is_resolved_log
                    )
                )
            logger.info(
                f"設備 {equipment_id} 狀態從 {current_status_row[0]} 更新為 {db_status}。"
            )
            return True
        return False

    def _check_operation_status(self, conn, equipment_id, name, eq_type):
        """檢查設備運行狀態，包括長時間運行、異常停機等 (已停用)"""
        logger.debug(f"設備 {name} ({equipment_id}) 的運行狀態監控已停用。")
        return

    def _determine_severity(self, metric_type, value):
        """
        根據載入的閾值數據，判斷指標的嚴重程度。
        """
        val = float(value)
        thresholds = self.metric_thresholds_data.get(metric_type)

        if not thresholds:
            logger.warning(f"未找到指標 '{metric_type}' 的閾值數據。無法判斷嚴重程度。")
            return None

        # 優先判斷重度異常
        emergency_thresh = thresholds.get("emergency")
        if emergency_thresh:
            e_min = emergency_thresh.get("min")
            e_max = emergency_thresh.get("max")
            e_op = emergency_thresh.get("op")

            if e_op == '>':
                if e_min is not None and val > e_min:
                    return self.SEVERITY_EMERGENCY
            elif e_op == '<':
                if e_max is not None and val < e_max:
                    return self.SEVERITY_EMERGENCY
            # 如果沒有操作符，則預設為區間 [min, max]
            elif e_min is not None and e_max is not None and e_min <= val <= e_max:
                return self.SEVERITY_EMERGENCY

        # 判斷中度異常 (臨界)
        critical_thresh = thresholds.get("critical")
        if critical_thresh:
            c_min = critical_thresh.get("min")
            c_max = critical_thresh.get("max")
            # 區間判斷 (min, max]
            if c_min is not None and c_max is not None and c_min < val <= c_max:
                return self.SEVERITY_CRITICAL

        # 判斷輕度異常 (警告)
        warning_thresh = thresholds.get("warning")
        if warning_thresh:
            w_min = warning_thresh.get("min")
            w_max = warning_thresh.get("max")
            # 區間判斷 (min, max]
            if w_min is not None and w_max is not None and w_min < val <= w_max:
                return self.SEVERITY_WARNING

        return None  # 如果不在任何異常區間內，則視為正常

    def _severity_level(self, severity):
        levels = {
            self.SEVERITY_WARNING: 1,
            self.SEVERITY_CRITICAL: 2,
            self.SEVERITY_EMERGENCY: 3,
            "info": 0,
            "normal_recovery": 0
        }
        return levels.get(severity, 0)

    def _severity_emoji(self, severity):
        emojis = {
            self.SEVERITY_WARNING: "⚠️",
            self.SEVERITY_CRITICAL: "🔴",
            self.SEVERITY_EMERGENCY: "🚨",
            "info": "ℹ️",
            "normal_recovery": "✅"
        }
        return emojis.get(severity, "⚠️")

    def _get_equipment_data(self, conn_unused, equipment_id):
        """從資料庫獲取指定設備的名稱、類型和位置資訊"""
        try:
            with self.db._get_connection() as new_conn:
                cursor = new_conn.cursor()
                cursor.execute(
                    "SELECT name, eq_type, location FROM equipment WHERE equipment_id = ?;",
                    (equipment_id,),
                )
                result = cursor.fetchone()
                if result:
                    return {
                        "name": result[0], "type": result[1],
                        "type_name": self.equipment_type_names.get(result[1], result[1]),
                        "location": result[2]
                    }
        except pyodbc.Error as db_err:
            logger.error(
                f"從 _get_equipment_data 獲取設備 {equipment_id} 資料失敗: {db_err}"
            )
        return {
            "name": "未知", "type": "未知",
            "type_name": "未知設備", "location": "未知"
        }

    def _generate_ai_recommendation(self, anomalies, equipment_data):
        """產生 AI 增強的異常描述和建議（使用現有的 OpenAI 服務）"""
        try:
            from src.main import OpenAIService

            context_parts = []
            for anomaly in anomalies:
                ts_str = (
                    anomaly['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    if anomaly.get('timestamp') else 'N/A'
                )
                value_str = f"{anomaly['value']:.2f}" if anomaly['value'] is not None else "N/A"

                normal_val_display = self.metric_thresholds_data.get(
                    anomaly["metric"], {}
                ).get("normal_value")

                if anomaly["metric"] == "轉速":
                    metric_detail = (f"轉速: {int(anomaly['value'])} RPM "
                                     f"(正常約 {int(normal_val_display)} RPM)"
                                     if normal_val_display is not None else "RPM")
                elif anomaly["metric"] in ["變形量(mm)", "刀具裂痕"]:  # 統一處理這兩種
                    metric_detail = (f"{anomaly['metric']}: {anomaly['value']:.3f} mm "
                                     f"(正常約 {normal_val_display:.3f} mm 以下)"
                                     if normal_val_display is not None else "mm")
                else:
                    metric_detail = (
                        f"指標 '{anomaly['metric']}': 目前值 {value_str} "
                        f"(單位: {anomaly['unit'] or ''})"
                    )

                context_parts.append(f"- {metric_detail}, 記錄時間: {ts_str}, 異常等級: {anomaly['severity'].upper()}")
            context = "偵測到的異常狀況:\n" + "\n".join(context_parts)

            prompt = (
                "作為一個半導體設備維護專家，請分析以下設備的異常狀況並提供具體的初步排查建議和可能的解決方案。\n"
                f"設備資料：名稱 {equipment_data.get('name')}, "
                f"型號 {equipment_data.get('type_name')}, "
                f"位置 {equipment_data.get('location')}\n"
                f"異常詳情：\n{context}\n"
                "請以簡潔、條列式的方式提供建議，重點放在操作員或初級維護人員可以執行的檢查步驟。"
            )

            system_ai_user_id = "SYSTEM_AI_HELPER_EQUIPMENT"
            db.set_user_preference(system_ai_user_id, language="zh-Hant")

            service = OpenAIService(message=prompt, user_id=system_ai_user_id)
            response = service.get_response()
            return response
        except ImportError as imp_err:
            logger.error(f"無法導入 OpenAIService: {imp_err}")
            return "無法獲取 AI 建議 (模組導入錯誤)。"
        except Exception as e:
            logger.exception(f"產生 AI 建議時發生錯誤: {e}")
            return "無法獲取 AI 建議 (系統錯誤)。"

    def _send_alert_notification(self, equipment_id, message, severity):
        """發送通知給訂閱該設備的使用者及相關負責人"""
        try:
            from src.linebot_connect import send_notification

            user_ids_to_notify = set()

            with self.db._get_connection() as conn:
                cursor = conn.cursor()
                level_filter_tuple = ()
                if severity == self.SEVERITY_EMERGENCY:
                    level_filter_tuple = ('all', 'critical', 'emergency')
                elif severity == self.SEVERITY_CRITICAL:
                    level_filter_tuple = ('all', 'critical')
                elif severity == self.SEVERITY_WARNING:
                    level_filter_tuple = ('all',)
                else:
                    level_filter_tuple = ('all',)

                if level_filter_tuple:
                    placeholders = ', '.join(['?'] * len(level_filter_tuple))
                    sql_subscriptions = (
                        f"SELECT user_id FROM user_equipment_subscriptions "
                        f"WHERE equipment_id = ? AND notification_level IN ({placeholders});"
                    )
                    params = [equipment_id] + list(level_filter_tuple)
                    cursor.execute(sql_subscriptions, params)
                    for row in cursor.fetchall():
                        user_ids_to_notify.add(row[0])

                cursor.execute(
                    "SELECT eq_type FROM equipment WHERE equipment_id = ?;", (equipment_id,)
                )
                equipment_info = cursor.fetchone()
                if equipment_info:
                    equipment_type = equipment_info[0]
                    cursor.execute(
                        "SELECT user_id FROM user_preferences "
                        "WHERE responsible_area = ? OR is_admin = 1;",
                        (equipment_type,)
                    )
                    for row in cursor.fetchall():
                        user_ids_to_notify.add(row[0])

            if not user_ids_to_notify:
                logger.warning(
                    f"設備 {equipment_id} 發生警報，但找不到任何符合條件的通知對象。"
                )
                return  # Added return here if no users to notify

            final_message = (
                f"{self._severity_emoji(severity)} "
                f"設備警報 ({equipment_id}):\n{message}"
            )

            for user_id_val in user_ids_to_notify:
                if send_notification(user_id_val, final_message):
                    logger.info(
                        f"警報通知已發送給使用者: {user_id_val} 針對設備 {equipment_id}"
                    )
                else:
                    logger.error(f"發送警報通知給使用者: {user_id_val} 失敗")

        except pyodbc.Error as db_err:
            logger.exception(
                f"發送設備 {equipment_id} 的通知時發生資料庫錯誤: {db_err}"
            )
        except ImportError:  # send_notification 導入失敗
            logger.error("無法導入 send_notification 函數。警報無法發送。")
        except Exception as e:  # Renamed 'e' from previous 'e' in _check_equipment_metrics
            logger.exception(
                f"發送設備 {equipment_id} 的通知時發生非預期錯誤: {e}"
            )
