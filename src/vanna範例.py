from vanna.ollama import Ollama
from vanna.chromadb import ChromaDB_VectorStore
from config import Config


class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)


vn = MyVanna(config={'model': 'gpt-oss:20b'})
resolved_server = Config.DB_SERVER
resolved_database = Config.DB_NAME
connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={resolved_server};"
    f"DATABASE={resolved_database};"
    "Trusted_Connection=yes;"
)

vn.connect_to_mssql(odbc_conn_str=connection_string)  # You can use the ODBC connection string here
# The information schema query may need some tweaking depending on your database. This is a good starting point.
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
plan

# If you like the plan, then uncomment this and run it to train
vn.train(plan=plan)

# The following are methods for adding training data. Make sure you modify the examples to match your database.

# DDL statements are powerful because they specify table names, colume names, types, and potentially relationships

# --- 情境 1：查詢查詢EQ003在2025年整年三種異常的各類型時數 ---
vn.train(ddl="""CREATE TABLE stats_abnormal_yearly (
        [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
        [year] INT NOT NULL,
        [detected_anomaly_type] NVARCHAR(255) NOT NULL,
        [downtime_sec] INT NULL,""")
vn.train(documentation="如果要查詢EQ003設備的資料，equipment_id 欄位必須為 'EQ003'。")
vn.train(documentation=" 'downtime_sec' 欄位是「故障時數」或「停機時間」，單位是「秒」。")
vn.train(documentation="要計算「加總時數」或「總時長」，必須使用 SUM(downtime_sec)。")
vn.train(documentation="detected_anomaly_type 是用來識別異常類型的欄位。")
vn.train(documentation="detected_anomaly_type 有三個異常情況，分別是：轉速太低、刀具裂痕、刀具變形。")
vn.train(documentation="若是查詢整年的資料，year欄位要等於2025。")
vn.train(sql="""SELECT detected_anomaly_type,
               SUM(downtime_sec) AS total_downtime_sec
        FROM stats_abnormal_yearly
        WHERE equipment_id = 'EQ003'
          AND year = 2025
          AND detected_anomaly_type IN (N'轉速太低', N'刀具裂痕', N'刀具變形')
        GROUP BY detected_anomaly_type
        ORDER BY detected_anomaly_type
        """)

# --- 情境 2：查詢EQ003在2025年1~3月(第一季)三種異常的各類型時數 ---
vn.train(ddl="""CREATE TABLE stats_abnormal_quarterly (
        [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
        [year] INT NOT NULL,
        [quarter] INT NOT NULL,
        [detected_anomaly_type] NVARCHAR(255) NOT NULL,
        [downtime_sec] INT NULL,
    )""")
vn.train(documentation="如果要查詢EQ003設備的資料，equipment_id 欄位必須為 'EQ003'。")
vn.train(documentation=" 'downtime_sec' 欄位是「故障時數」或「停機時間」，單位是「秒」。")
vn.train(documentation="要計算「加總時數」或「總時長」，必須使用 SUM(downtime_sec)。")
vn.train(documentation="detected_anomaly_type 是用來識別異常類型的欄位。")
vn.train(documentation="detected_anomaly_type 有三個異常情況，分別是：轉速太低、刀具裂痕、刀具變形。")
vn.train(documentation="要查詢2025年第一季的資料，year欄位要等於2025，quarter欄位要等於1。")
vn.train(documentation="要查詢2025年1至3月的資料，year欄位要等於2025，quarter欄位要等於1。")
vn.train(documentation="要查詢2025年4至6月的資料，year欄位要等於2025，quarter欄位要等於2。")
vn.train(documentation="要查詢2025年7至9月的資料，year欄位要等於2025，quarter欄位要等於3。")
vn.train(documentation="要查詢2025年10至12月的資料，year欄位要等於2025，quarter欄位要等於4。")
vn.train(sql="""SELECT detected_anomaly_type,
               SUM(downtime_sec) AS total_downtime_sec
        FROM stats_abnormal_quarterly
        WHERE equipment_id = 'EQ003'
          AND year = 2025 AND quarter = 1
          AND detected_anomaly_type IN (N'轉速太低', N'刀具裂痕', N'刀具變形')
        GROUP BY detected_anomaly_type
        ORDER BY detected_anomaly_type
        """)

# --- 情境 3：查詢 EQ002 在 2025 年 10 月三種異常的各類型時數 ---
vn.train(ddl="""CREATE TABLE stats_abnormal_monthly (
        [equipment_id] NVARCHAR(255) NOT NULL FOREIGN KEY REFERENCES equipment(equipment_id),
        [year] INT NOT NULL,
        [month] INT NOT NULL,
        [detected_anomaly_type] NVARCHAR(255) NOT NULL,
        [total_operation_hrs] INT NULL,
        [downtime_sec] INT NULL,
    )""")
vn.train(documentation="如果要查詢EQ002設備的資料，equipment_id 欄位必須為 'EQ002'。")
vn.train(documentation=" 'downtime_sec' 欄位是「故障時數」或「停機時間」，單位是「秒」。")
vn.train(documentation="要計算「加總時數」或「總時長」，必須使用 SUM(downtime_sec)。")
vn.train(documentation="detected_anomaly_type 是用來識別異常類型的欄位。")
vn.train(documentation="detected_anomaly_type 有三個異常情況，分別是：轉速太低、刀具裂痕、刀具變形。")
vn.train(documentation="要查詢2025年10月的資料，year欄位要等於2025，month欄位要等於10。")

vn.train(
    sql="""
        SELECT detected_anomaly_type,
               SUM(downtime_sec) AS total_downtime_sec
        FROM stats_abnormal_monthly
        WHERE equipment_id = 'EQ002'
          AND year = 2025 AND month = 10
          AND detected_anomaly_type IN (N'轉速太低', N'刀具裂痕', N'刀具變形')
        GROUP BY detected_anomaly_type
    """
)

# At any time you can inspect what training data the package is able to reference
training_data = vn.get_training_data()
training_data

"""## Asking the AI
Whenever you ask a new question, it will find the 10 most relevant pieces of training data
and use it as part of the LLM prompt to generate the SQL.
python"""

results = vn.ask(
    question=(
        "EQ004 在 2025 年 10到12月 三種異常的各類型的時數，用分鐘+秒為單位。"
        "如果該異常沒有時數，也必須出現在圖表上。"
    )
)
print(results)
print(type(results))
