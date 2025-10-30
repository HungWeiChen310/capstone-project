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

vn.connect_to_mssql(odbc_conn_str=connection_string) # You can use the ODBC connection string here
# The information schema query may need some tweaking depending on your database. This is a good starting point.
df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
plan

# If you like the plan, then uncomment this and run it to train
vn.train(plan=plan)

# The following are methods for adding training data. Make sure you modify the examples to match your database.

# DDL statements are powerful because they specify table names, colume names, types, and potentially relationships

#--- 情境 1：查詢 EQ001 的近五個異常通知 ---
vn.train(ddl="""CREATE TABLE alert_history(
        [error_id] INT PRIMARY KEY, 
        [equipment_id] NVARCHAR(255), 
        [detected_anomaly_type] NVARCHAR(255), 
        [created_time] datetime2(2))""")
vn.train(documentation="要查詢「最近」通知，請使用 'created_time' 欄位並以 DESC 排序。")
vn.train(documentation="查詢前五筆通知，在 SQL 中可使用 'TOP 5'")
vn.train(documentation="若要篩選特定設備，請使用 WHERE 條件過濾 'equipment_id'。")
vn.train(sql="""SELECT TOP 5 error_id, equipment_id,
        detected_anomaly_type,
        created_time FROM alert_history
        WHERE equipment_id = 'EQ001'
        ORDER BY created_time DESC, error_id DESC
        """)

# --- 情境 2：查詢所有「故障中」的機器 ---
vn.train(ddl="""CREATE TABLE equipment (
        [equipment_id] NVARCHAR(255) NOT NULL PRIMARY KEY,
        [name] NVARCHAR(255) NOT NULL,
        [status] NVARCHAR(255) NOT NULL
        )""")
vn.train(documentation="如果 status 不是 'normal'，代表設備異常或故障。")
vn.train(documentation="查詢異常中的設備可使用 WHERE 過濾 status 欄位。")
vn.train(sql="""SELECT name, equipment_id 
        FROM equipment
        WHERE status != 'normal'
        """)

# --- 情境 3：查詢 EQ002 在 2025 年 10 月三種異常的各類型時數 ---
vn.train(ddl="""CREATE TABLE error_logs (
        [log_date] DATE NOT NULL,
        [equipment_id] NVARCHAR(255) NOT NULL,
        [detected_anomaly_type] NVARCHAR(MAX) NOT NULL,
        [downtime_sec] INT NULL
    )""")
vn.train(documentation=" 'downtime_sec' 欄位是「故障時數」或「停機時間」，單位是「秒」。")
vn.train(documentation="要計算「加總時數」或「總時長」，必須使用 SUM(downtime_sec)。")
vn.train(documentation="要查詢「各種類型」的時數，必須 GROUP BY detected_anomaly_type。")
vn.train(documentation="要查詢2025年10月的資料，log_date 欄位需要在 '2025-10-01' 和 '2025-11-01' 之間。")

vn.train(
    sql="""
        SELECT detected_anomaly_type,
               SUM(ISNULL(downtime_sec, 0)) AS total_downtime_sec
        FROM error_logs
        WHERE equipment_id = 'EQ002'
          AND log_date >= '2025-10-01' AND log_date < '2025-11-01'
          AND detected_anomaly_type IN (N'轉速太低', N'刀具裂痕', N'刀具變形')
        GROUP BY detected_anomaly_type
        ORDER BY detected_anomaly_type
    """
)

# At any time you can inspect what training data the package is able to reference
training_data = vn.get_training_data()
training_data

"""## Asking the AI
Whenever you ask a new question, it will find the 10 most relevant pieces of training data and use it as part of the LLM prompt to generate the SQL.
python"""

results = vn.ask(question="EQ001 在 2025 年 10 月三種異常的各類型的時間，用分鐘+秒為單位")
print(results)
