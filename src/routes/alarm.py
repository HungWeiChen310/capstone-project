# src/routes/alarm.py
from flask import request, jsonify
from . import alarm_bp
from ..database import db
from ..services.line_service import send_notification, send_multicast_notification
import logging
import datetime

logger = logging.getLogger(__name__)


@alarm_bp.route("/alarms", methods=["POST"])
def alarms():
    """接收警報訊息"""
    data = request.get_json(force=True, silent=True)
    key = ("equipment_id", "detected_anomaly_type", "severity_level")
    if data and all(k in data for k in key):
        data["created_time"] = str(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.insert_alert_history(log_data=data)
        equipment_id = data["equipment_id"]
        subscribers = db.get_subscribed_users(equipment_id)
        if subscribers:
            message_text = (
                f"設備 {equipment_id} 在 {data['created_time']} 時發生 {data['detected_anomaly_type']} 警報，"
                f"嚴重程度 {data['severity_level']}。"
            )
            # 使用 Multicast 一次發送給所有訂閱者
            send_multicast_notification(subscribers, message_text)
        else:
            logger.info(f"No subscribers found for equipment {equipment_id}")

    logger.info("Received JSON from client:", data)
    return jsonify({"status": "success"}), 200


@alarm_bp.route("/resolvealarms", methods=["POST"])
def resolve_alarms():
    """接收警報解決訊息"""
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received."}), 400

    required_keys = ("error_id", "detected_anomaly_type", "equipment_id", "resolved_by")
    if not all(k in data for k in required_keys):
        return jsonify({
            "status": "error",
            "message": (
                "Missing required keys: error_id, detected_anomaly_type, "
                "equipment_id, resolved_by."
            ),
        }), 400

    try:
        db_result = db.resolve_alert_history(log_data=data)

        if db_result is None:
            logger.warning(
                f"嘗試解決警報失敗，找不到 error_id: {data['error_id']} / "
                f"detected_anomaly_type: {data['detected_anomaly_type']} / "
                f"equipment_id: {data['equipment_id']}。"
            )
            return jsonify({
                "status": "error",
                "message": (
                    f"Alarm with error_id {data['error_id']} "
                    f"and detected_anomaly_type {data['detected_anomaly_type']} "
                    f"and equipment_id {data['equipment_id']} not found."
                )
            }), 404
        elif isinstance(db_result, tuple):
            logger.info(
                f"警報 {data['error_id']} / detected_anomaly_type: {data['detected_anomaly_type']} / "
                f"equipment_id: {data['equipment_id']} 先前已被解決，不發送通知。")
            return jsonify({
                "status": "success", "message": "Alarm was already resolved. No notification sent."
            }), 200
        else:
            equipment_id = data['equipment_id']
            detected_anomaly_type = data['detected_anomaly_type']
            resolved_time_from_db = db_result

            subscribers = db.get_subscribed_users(equipment_id)
            if subscribers:
                message_text = (
                    f"設備 {equipment_id} 發生 {detected_anomaly_type} 警報，"
                    f"在 {resolved_time_from_db.strftime('%Y-%m-%d %H:%M:%S')} 由 {data['resolved_by']} 解決。"
                    f"解決說明: {data.get('resolution_notes') or '無'}"
                )
                # 使用 Multicast 一次發送給所有訂閱者
                send_multicast_notification(subscribers, message_text)
            else:
                logger.info(f"No subscribers found for equipment {equipment_id}")

            return jsonify({"status": "success", "message": "Alarm resolved and notification sent."}), 200

    except Exception as e:
        logger.error(f"處理警報解決請求時發生錯誤: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred."}), 500
