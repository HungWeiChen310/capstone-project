import re
import logging
import pandas as pd
import pyodbc
from database import db


# --- 1. 設定日誌記錄 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- 2. 設定 Excel 檔案路徑 ---
# 請確保此路徑對於您的執行環境是正確的
EXCEL_FILE_PATH = r'data\simulated_data (1).xlsx'


# --- 3. 輔助函數 (無需修改) ---
def parse_threshold_string(threshold_str):
    if pd.isna(threshold_str) or threshold_str is None:
        return None, None, None
    s = str(threshold_str).strip()
    match = re.match(r'([><=~]?)\s*([0-9.]+)\s*([~-]\s*([0-9.]+))?', s)
    if not match:
        return None, None, None
    op, val1, _, val2 = match.groups()
    try:
        val1 = float(val1) if val1 else None
        val2 = float(val2) if val2 else None
    except ValueError:
        return None, None, None
    if op == '>':
        return val1, None, '>'
    elif op == '<':
        return None, val1, '<'
    elif op == '~' or _ == '-':
        return val1, val2, None
    else:
        return val1, val1, None


def parse_and_transform_threshold_row(row):
    metric_type = row.get("異常類型")
    normal_value, _, _ = parse_threshold_string(row.get("正常值"))
    warning_min, warning_max, _ = parse_threshold_string(row.get("輕度異常"))
    critical_min, critical_max, _ = parse_threshold_string(row.get("中度異常"))
    emergency_min, emergency_max, emergency_op = parse_threshold_string(
        row.get("重度異常")
    )
    return (metric_type, normal_value, warning_min, warning_max,
            critical_min, critical_max, emergency_min, emergency_max,
            emergency_op)


# --- 4. 完整的表格匯入設定 (包含所有統計表) ---
TABLE_CONFIGS = [
    {
        "excel_sheet_name": "equipment",
        "sql_table_name": "equipment",
        "sql_columns": ["equipment_id", "name", "eq_type", "location",
                        "status", "last_updated"],
        "transform_row_data": lambda row: (
            row.get('equipment_id'), row.get('name'), row.get('eq_type'),
            row.get('location'), row.get('status'),
            pd.to_datetime(row.get('last_updated'))
            if pd.notna(row.get('last_updated')) else None
        )
    },
    {
        "excel_sheet_name": "alert_history",
        "sql_table_name": "alert_history",
        "sql_columns": ["id", "equipment_id", "alert_type", "severity",
                        "message", "is_resolved", "created_at",
                        "resolved_at", "resolved_by", "resolution_notes"],
        "transform_row_data": lambda row: (
            row.get('ID'), row.get('equipment_id'), row.get('alert_type'),
            row.get('severity'),
            str(row.get('訊息')) if pd.notna(row.get('訊息')) else None,
            row.get('is_resolved'),
            pd.to_datetime(row.get('created_at'))
            if pd.notna(row.get('created_at')) else None,
            pd.to_datetime(row.get('resolved_at'))
            if pd.notna(row.get('resolved_at')) else None,
            str(row.get('resolved_by'))
            if pd.notna(row.get('resolved_by')) else None,
            str(row.get('resolution_notes'))
            if pd.notna(row.get('resolution_notes')) else None
        )
    },
    {
        "excel_sheet_name": "設備監測數據",
        "sql_table_name": "equipment_metrics",
        "sql_columns": ["id", "equipment_id", "metric_type", "status",
                        "value", "unit", "timestamp"],
        "transform_row_data": lambda row: (
            row.get('id'), row.get('equipment_id'), row.get('metric_type'),
            str(row.get('狀態')) if pd.notna(row.get('狀態')) else None,
            row.get('value'),
            str(row.get('unit')) if pd.notna(row.get('unit')) else None,
            pd.to_datetime(row.get('timestamp'))
            if pd.notna(row.get('timestamp')) else None
        )
    },
    {
        "excel_sheet_name": "切割機標準值",
        "sql_table_name": "equipment_metric_thresholds",
        "sql_columns": ["metric_type", "normal_value", "warning_min",
                        "warning_max", "critical_min", "critical_max",
                        "emergency_min", "emergency_max", "emergency_op"],
        "transform_row_data": parse_and_transform_threshold_row
    },
    {
        "excel_sheet_name": "異常紀錄error_log",
        "sql_table_name": "error_logs",
        "sql_columns": ["log_date", "error_id", "equipment_id",
                        "deformation_mm", "rpm", "event_time",
                        "detected_anomaly_type", "downtime_duration",
                        "resolved_at", "resolution_notes"],
        "transform_row_data": lambda row: (
            pd.to_datetime(str(row.get('日期')))
            if pd.notna(row.get('日期')) else None,
            str(row.get('error_id')), str(row.get('equipment_id')),
            row.get('變形量(mm)'), row.get('轉速'),
            pd.to_datetime(str(row.get('時間')))
            if pd.notna(row.get('時間')) else None,
            str(row.get('偵測異常類型')), str(row.get('停機時長')),
            pd.to_datetime(str(row.get('回復時間')))
            if pd.notna(row.get('回復時間')) else None,
            str(row.get('備註')) if pd.notna(row.get('備註')) else None
        )
    },
    {
        "excel_sheet_name": "運作統計(月)",
        "sql_table_name": "stats_operational_monthly",
        "sql_columns": ["equipment_id", "year", "month",
                        "total_operation_duration", "total_downtime_duration",
                        "downtime_rate_percent", "description"],
        "transform_row_data": lambda row: (
            str(row.get('equipment_id')), row.get('年'), row.get('月'),
            str(row.get('總運作時長')),
            str(row.get('停機總時長')), str(row.get('停機率(%)')),
            str(row.get('說明'))
        )
    },
    {
        "excel_sheet_name": "運作統計(季)",
        "sql_table_name": "stats_operational_quarterly",
        "sql_columns": ["equipment_id", "year", "quarter",
                        "total_operation_duration", "total_downtime_duration",
                        "downtime_rate_percent", "description"],
        "transform_row_data": lambda row: (
            str(row.get('equipment_id')), row.get('年'), row.get('季度'),
            str(row.get('總運作時長')),
            str(row.get('停機總時長')), str(row.get('停機率(%)')),
            str(row.get('說明'))
        )
    },
    {
        "excel_sheet_name": "運作統計(年)",
        "sql_table_name": "stats_operational_yearly",
        "sql_columns": ["equipment_id", "year", "total_operation_duration",
                        "total_downtime_duration", "downtime_rate_percent",
                        "description"],
        "transform_row_data": lambda row: (
            str(row.get('equipment_id')), row.get('年'),
            str(row.get('總運作時長')),
            str(row.get('停機總時長')), str(row.get('停機率(%)')),
            str(row.get('說明'))
        )
    },
    {
        "excel_sheet_name": "各異常統計(月)",
        "sql_table_name": "stats_abnormal_monthly",
        "sql_columns": ["equipment_id", "year", "month",
                        "detected_anomaly_type", "downtime_duration",
                        "downtime_rate_percent", "description"],
        "transform_row_data": lambda row: (
            str(row.get('equipment_id')), row.get('年'), row.get('月'),
            str(row.get('偵測異常類型')),
            str(row.get('停機時長')), str(row.get('停機率(%)')),
            str(row.get('說明'))
        )
    },
    {
        "excel_sheet_name": "各異常統計(季)",
        "sql_table_name": "stats_abnormal_quarterly",
        "sql_columns": ["equipment_id", "year", "quarter",
                        "detected_anomaly_type", "downtime_duration",
                        "downtime_rate_percent", "description"],
        "transform_row_data": lambda row: (
            str(row.get('equipment_id')), row.get('年'), row.get('季度'),
            str(row.get('偵測異常類型')),
            str(row.get('停機時長')), str(row.get('停機率(%)')),
            str(row.get('說明'))
        )
    },
    {
        "excel_sheet_name": "各異常統計(年)",
        "sql_table_name": "stats_abnormal_yearly",
        "sql_columns": ["equipment_id", "year", "detected_anomaly_type",
                        "downtime_duration", "downtime_rate_percent",
                        "description"],
        "transform_row_data": lambda row: (
            str(row.get('equipment_id')), row.get('年'),
            str(row.get('偵測異常類型')),
            str(row.get('停機時長')), str(row.get('停機率(%)')),
            str(row.get('說明'))
        )
    }
]


# --- 5. 修改後的匯入主程式 ---
def import_data_from_excel():
    """從指定的 Excel 檔案讀取數據，並使用高效能的批次插入將其匯入到資料庫中。"""
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.fast_executemany = True
            logger.info("成功連接到 MS SQL 資料庫，已啟用 fast_executemany。")

            for config in TABLE_CONFIGS:
                sheet_name = config["excel_sheet_name"]
                sql_table_name = config["sql_table_name"]
                sql_columns = config["sql_columns"]
                transform_row_data = config["transform_row_data"]

                logger.info(
                    f"--- 開始處理資料表: {sql_table_name} (來源: {sheet_name}) ---"
                )

                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {sql_table_name}")
                    if cursor.fetchone()[0] > 0:
                        logger.info(f"資料表 '{sql_table_name}' 已存在資料，跳過匯入。")
                        continue

                    data_frame = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name)
                    data_frame = data_frame.where(pd.notna(data_frame), None)

                    if data_frame.empty:
                        logger.warning(f"工作表 '{sheet_name}' 為空，跳過。")
                        continue

                    logger.info(
                        f"成功讀取 Excel 工作表 '{sheet_name}'，共 {len(data_frame)} 行。"
                    )

                    sql_columns_str = ', '.join([f"[{col}]" for col in sql_columns])
                    placeholders_str = ', '.join(['?' for _ in sql_columns])
                    insert_sql = (
                        f"INSERT INTO [{sql_table_name}] ({sql_columns_str}) "
                        f"VALUES ({placeholders_str})"
                    )

                    data_to_insert = []
                    for index, row in data_frame.iterrows():
                        # --- 修改部分 START ---
                        # 採納建議，對單行資料轉換進行更精細的錯誤捕捉。
                        # 這可以幫助我們快速定位是數值問題、類型問題還是其他未知問題。
                        try:
                            transformed_tuple = transform_row_data(row)
                            data_to_insert.append(transformed_tuple)
                        except ValueError as ve:
                            # 捕獲數值轉換錯誤，例如 float('無效字串')
                            logger.error(
                                f"轉換第 {index + 2} 行資料時數值轉換失敗: {ve}。"
                                f"資料: {row.to_dict()}"
                            )
                        except TypeError as te:
                            # 捕獲類型錯誤，例如對 NoneType 進行了不支援的操作
                            logger.error(
                                f"轉換第 {index + 2} 行資料時類型錯誤: {te}。"
                                f"資料: {row.to_dict()}"
                            )
                        except Exception as e:
                            # 捕獲所有其他未預期的錯誤
                            logger.error(
                                f"轉換第 {index + 2} 行資料時發生未知失敗: {e}。"
                                f"資料: {row.to_dict()}"
                            )
                        # --- 修改部分 END ---

                    if data_to_insert:
                        logger.info(
                            f"準備將 {len(data_to_insert)} 行資料批次插入到 "
                            f"'{sql_table_name}'..."
                        )
                        try:
                            cursor.executemany(insert_sql, data_to_insert)
                            conn.commit()
                            logger.info(f"'{sql_table_name}' 資料匯入完成。")
                        except pyodbc.Error as e:
                            logger.error(
                                f"批次插入到 '{sql_table_name}' 時發生資料庫錯誤，"
                                f"正在回滾: {e}"
                            )
                            conn.rollback()

                except pd.errors.ParserError as e:
                    logger.error(f"解析 Excel 工作表 '{sheet_name}' 失敗: {e}")
                except Exception as e:
                    logger.error(
                        f"處理工作表 '{sheet_name}' 時發生未預期錯誤: {e}"
                    )
                    continue

    except FileNotFoundError:
        logger.error(f"錯誤：找不到 Excel 檔案 '{EXCEL_FILE_PATH}'。請檢查路徑。")
    except pyodbc.Error as e:
        logger.error(f"資料庫連線或操作失敗: {e}")
    except Exception as e:
        logger.error(f"執行 Excel 匯入腳本時發生未知錯誤: {e}")


if __name__ == '__main__':
    logger.info("腳本啟動：開始匯入初始資料到資料庫。")
    import_data_from_excel()
    logger.info("所有資料匯入任務完成。")
