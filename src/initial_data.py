# src/initial_data.py
import logging
import sqlite3
from database import db
logger = logging.getLogger(__name__)


def initialize_equipment_data():
    """初始化設備資料"""
    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.cursor()
            # 檢查是否已有設備資料
            cursor.execute("SELECT COUNT(*) FROM equipment")
            if cursor.fetchone()[0] > 0:
                logger.info("設備資料已存在，略過初始化")
                return
            # 插入黏晶機
            die_bonders = [
                ("DB001", "黏晶機A1", "die_bonder", "生產線A"),
                ("DB002", "黏晶機A2", "die_bonder", "生產線A"),
                ("DB003", "黏晶機B1", "die_bonder", "生產線B"),
            ]
            # 插入打線機
            wire_bonders = [
                ("WB001", "打線機A1", "wire_bonder", "生產線A"),
                ("WB002", "打線機A2", "wire_bonder", "生產線A"),
                ("WB003", "打線機B1", "wire_bonder", "生產線B"),
                ("WB004", "打線機B2", "wire_bonder", "生產線B"),
            ]
            # 插入切割機
            dicers = [
                ("DC001", "切割機A1", "dicer", "生產線A"),
                ("DC002", "切割機B1", "dicer", "生產線B"),
            ]
            # 合併所有設備
            equipments = die_bonders + wire_bonders + dicers
            # 插入設備資料
            for equipment_id, name, equipment_type, location in equipments:
                cursor.execute(
                    """
                    INSERT INTO equipment (equipment_id, name, type, location, status)
                    VALUES (?, ?, ?, ?, 'normal')
                """,
                    (equipment_id, name, equipment_type, location),
                )
            # 插入黏晶機閾值
            die_bonder_metrics = [
                # 設備ID, 指標類型, 數值, 最小閾值, 最大閾值, 單位
                ("DB001", "溫度", 23.5, 18.0, 28.0, "°C"),
                ("DB001", "壓力", 1.5, 1.0, 2.0, "MPa"),
                ("DB001", "Pick準確率", 99.2, 98.0, None, "%"),
                ("DB001", "良率", 99.5, 98.0, None, "%"),
                ("DB001", "運轉時間", 120, None, None, "分鐘"),
                ("DB002", "溫度", 24.2, 18.0, 28.0, "°C"),
                ("DB002", "壓力", 1.6, 1.0, 2.0, "MPa"),
                ("DB002", "Pick準確率", 98.7, 98.0, None, "%"),
                ("DB002", "良率", 99.1, 98.0, None, "%"),
                ("DB002", "運轉時間", 45, None, None, "分鐘"),
                ("DB003", "溫度", 22.8, 18.0, 28.0, "°C"),
                ("DB003", "壓力", 1.4, 1.0, 2.0, "MPa"),
                ("DB003", "Pick準確率", 99.4, 98.0, None, "%"),
                ("DB003", "良率", 99.7, 98.0, None, "%"),
                ("DB003", "運轉時間", 210, None, None, "分鐘"),
            ]
            # 插入打線機閾值
            wire_bonder_metrics = [
                # 設備ID, 指標類型, 數值, 最小閾值, 最大閾值, 單位
                ("WB001", "溫度", 26.2, 20.0, 30.0, "°C"),
                ("WB001", "壓力", 0.8, 0.5, 1.2, "MPa"),
                ("WB001", "金絲張力", 18.5, 15.0, 22.0, "cN"),
                ("WB001", "良率", 99.3, 98.0, None, "%"),
                ("WB001", "運轉時間", 180, None, None, "分鐘"),
                ("WB002", "溫度", 25.8, 20.0, 30.0, "°C"),
                ("WB002", "壓力", 0.9, 0.5, 1.2, "MPa"),
                ("WB002", "金絲張力", 17.2, 15.0, 22.0, "cN"),
                ("WB002", "良率", 98.9, 98.0, None, "%"),
                ("WB002", "運轉時間", 60, None, None, "分鐘"),
            ]
            # 插入切割機閾值
            dicer_metrics = [
                # 設備ID, 指標類型, 數值, 最小閾值, 最大閾值, 單位
                ("DC001", "溫度", 24.7, 20.0, 28.0, "°C"),
                ("DC001", "轉速", 30000, 25000, 35000, "RPM"),
                ("DC001", "冷卻水溫", 18.5, 16.0, 22.0, "°C"),
                ("DC001", "切割精度", 99.1, 98.5, None, "%"),
                ("DC001", "良率", 99.4, 98.0, None, "%"),
                ("DC001", "運轉時間", 90, None, None, "分鐘"),
                ("DC002", "溫度", 25.1, 20.0, 28.0, "°C"),
                ("DC002", "轉速", 29500, 25000, 35000, "RPM"),
                ("DC002", "冷卻水溫", 19.2, 16.0, 22.0, "°C"),
                ("DC002", "切割精度", 98.9, 98.5, None, "%"),
                ("DC002", "良率", 99.2, 98.0, None, "%"),
                ("DC002", "運轉時間", 150, None, None, "分鐘"),
            ]
            # 合併所有監測指標
            all_metrics = die_bonder_metrics + wire_bonder_metrics + dicer_metrics
            # 插入監測指標資料
            for (
                equipment_id,
                metric_type,
                value,
                threshold_min,
                threshold_max,
                unit,
            ) in all_metrics:
                cursor.execute(
                    """
                    INSERT INTO equipment_metrics
                    (equipment_id, metric_type, value, threshold_min, threshold_max, unit)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        equipment_id,
                        metric_type,
                        value,
                        threshold_min,
                        threshold_max,
                        unit,
                    ),
                )
            # 建立使用者訂閱（以管理員為例）
            cursor.execute(
                "SELECT user_id FROM user_preferences WHERE is_admin = 1 LIMIT 1"
            )
            admin_user = cursor.fetchone()
            if admin_user:
                admin_id = admin_user[0]
                # 訂閱所有設備
                for equipment_id, _, _, _ in equipments:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO user_equipment_subscriptions
                        (user_id, equipment_id, notification_level)
                        VALUES (?, ?, 'all')
                    """,
                        (admin_id, equipment_id),
                    )
            # 模擬一些運行中的作業
            operations = [
                ("DB001", "常規生產", "LOT-2023-11-001", "PROD-A123"),
                ("WB001", "常規生產", "LOT-2023-11-002", "PROD-B456"),
                ("DC001", "特殊切割", "LOT-2023-11-003", "PROD-C789"),
            ]
            for equipment_id, op_type, lot_id, product_id in operations:
                cursor.execute(
                    """
                    INSERT INTO equipment_operation_logs
                    (equipment_id, operation_type, start_time, lot_id, product_id)
                    VALUES (?, ?, datetime('now', '-2 hours'), ?, ?)
                """,
                    (equipment_id, op_type, lot_id, product_id),
                )
            conn.commit()
            logger.info("設備資料初始化完成")
    except Exception as e:
        logger.error(f"初始化設備資料失敗: {e}")
