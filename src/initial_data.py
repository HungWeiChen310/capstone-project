import logging
import pyodbc  # 為了捕獲 pyodbc.Error，需要匯入
from database import db  # db 物件現在是 MS SQL Server 的接口

logger = logging.getLogger(__name__)


def initialize_equipment_data():
    """初始化設備資料 (使用 MS SQL Server)"""
    try:
        # 使用 db._get_connection() 來獲取 MS SQL Server 連線
        with db._get_connection() as conn:
            cursor = conn.cursor()
            # 檢查是否已有設備資料
            cursor.execute("SELECT COUNT(*) FROM equipment;")
            if cursor.fetchone()[0] > 0:
                logger.info("設備資料已存在，略過初始化 (MS SQL Server)")
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
            equipments = die_bonders + wire_bonders + dicers

            for equipment_id, name, equipment_type, location in equipments:
                cursor.execute(
                    """
                    INSERT INTO equipment (equipment_id, name, type, location, status)
                    VALUES (?, ?, ?, ?, 'normal');
                    """,
                    (equipment_id, name, equipment_type, location),
                )

            # 設備指標 (範例，您可以擴展)
            # 注意：這裡的 SQL Server 的 GETDATE() 用於 timestamp，
            # 而 equipment_metrics 表的定義是 timestamp DATETIME2 DEFAULT GETDATE()
            # 所以插入時不需要特別指定 timestamp，除非要指定特定時間
            equipment_metrics_data = [
                # 黏晶機指標
                ("DB001", "溫度", 23.5, 18.0, 28.0, "°C"),
                ("DB001", "壓力", 1.5, 1.0, 2.0, "MPa"),
                ("DB001", "Pick準確率", 99.2, 98.0, None, "%"),
                ("DB001", "良率", 99.5, 98.0, None, "%"),
                ("DB001", "運轉時間", 120.0, None, None, "分鐘"),  # 確保 value 是 FLOAT
                ("DB002", "溫度", 24.2, 18.0, 28.0, "°C"),
                # ... (其他 DB002, DB003 的指標) ...

                # 打線機指標
                ("WB001", "溫度", 26.2, 20.0, 30.0, "°C"),
                ("WB001", "金絲張力", 18.5, 15.0, 22.0, "cN"),
                # ... (其他 WB001, WB002, WB003, WB004 的指標) ...

                # 切割機指標
                ("DC001", "溫度", 24.7, 20.0, 28.0, "°C"),
                ("DC001", "轉速", 30000.0, 25000.0, 35000.0, "RPM"),  # 確保 value 是 FLOAT
                # ... (其他 DC001, DC002 的指標) ...
            ]
            # 插入前檢查 equipment_metrics 是否為空，避免 cursor.executemany 出錯
            if equipment_metrics_data:
                cursor.executemany(
                    """
                    INSERT INTO equipment_metrics
                    (equipment_id, metric_type, value, threshold_min, threshold_max, unit)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    equipment_metrics_data,
                )

            # 模擬一些運行中的作業 (使用 MS SQL Server 的語法)
            operations = [
                ("DB001", "常規生產", "LOT-2023-11-001", "PROD-A123"),
                ("WB001", "常規生產", "LOT-2023-11-002", "PROD-B456"),
                ("DC001", "特殊切割", "LOT-2023-11-003", "PROD-C789"),
            ]
            for eq_id, op_type, lot_id, prod_id in operations:
                cursor.execute(
                    """
                    INSERT INTO equipment_operation_logs
                    (equipment_id, operation_type, start_time, lot_id, product_id)
                    VALUES (?, ?, DATEADD(hour, -2, GETDATE()), ?, ?);
                    """,
                    (eq_id, op_type, lot_id, prod_id),
                )

            conn.commit()
            logger.info("設備資料初始化完成 (MS SQL Server)")
    except pyodbc.Error as e:  # 捕獲 pyodbc 的錯誤
        logger.error(f"初始化設備資料失敗 (MS SQL Server): {e}")
    except Exception as e:
        logger.error(f"初始化設備資料時發生未知錯誤 (MS SQL Server): {e}")


if __name__ == '__main__':
    # 為了能夠獨立執行此腳本進行測試或初始化
    # 需要確保 database.py 中的 db 物件能正確連接
    # 通常這意味著環境變數 (如 DB_SERVER, DB_NAME) 需要被設定
    # 或者 Config 類能提供有效的預設值
    initialize_equipment_data()
