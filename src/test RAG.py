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
# vn.train(plan=plan)

# The following are methods for adding training data. Make sure you modify the examples to match your database.
"""
# DDL statements are powerful because they specify table names, colume names, types, and potentially relationships
vn.train(ddl=" ""
    CREATE TABLE IF NOT EXISTS my-table (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        age INT
    )
" "")

# Sometimes you may want to add documentation about your business terminology or definitions.
vn.train(documentation="Our business defines OTIF score as the percentage of orders that are delivered on time and in full")

# You can also add SQL queries to your training data. This is useful if you have some queries already laying around. You can just copy and paste those from your editor to begin generating new SQL.
vn.train(sql="SELECT * FROM my-table WHERE name = 'John Doe'")

# At any time you can inspect what training data the package is able to reference
training_data = vn.get_training_data()
training_data

# You can remove training data if there's obsolete/incorrect information. 
vn.remove_training_data(id='1-ddl')

" ""## Asking the AI
Whenever you ask a new question, it will find the 10 most relevant pieces of training data and use it as part of the LLM prompt to generate the SQL.
python"" "
vn.ask(question=...)
"""