# LINE Bot + 多模 LLM 智能助理系統

整合 LINE Messaging API、雲端 OpenAI 服務與本地 Ollama 推論能力，並搭配 SQL Server 資料庫及輕量化事件系統，打造具知識庫增強功能的企業級智能助理。系統除能回應工程技術問題，也提供半導體設備監控與管理後台功能。

## 主要功能

- **LINE Bot 智能對話**：接收使用者訊息並透過多種 LLM 生成專業、具實踐性的回應。
- **SQL RAG 知識庫增強**：從 SQL Server 中的知識庫檢索相關資訊，於回答前自動補強上下文。
- **半導體設備監控**：監控黏晶機、打線機、切割機等設備狀態，依異常程度推播警報。
- **多語言支援**：支援繁體中文、簡體中文、英文、日文與韓文等多種語言。
- **事件系統**：提供輕量級事件發布/訂閱機制，解耦模組依賴。
- **資料分析與儲存**：使用 SQL Server 儲存對話歷史、使用者偏好、設備指標與知識文件。
- **管理後台**：提供設備狀態、對話記錄、統計分析等可視化管理功能。

## 技術棧

- **Python 3.11**：主要開發語言。
- **Flask**：輕量級網頁框架，整合 LINE Webhook 與管理後台。
- **LINE Bot SDK v3.x**：處理 LINE 訊息互動。
- **OpenAI API / Ollama**：支援雲端與本地 LLM 模型。
- **SQL Server**：儲存對話、設備與知識庫資料。
- **Flask-Talisman**：提供 CSP、強制 HTTPS 等安全防護。
- **Docker**：容器化部署。
- **GitHub Actions**：CI/CD 自動化工作流程。

## 專案架構

```
.
├── src/                          # 主要源碼
│   ├── __init__.py               # Python 包初始化
│   ├── analytics.py              # 數據分析模組
│   ├── app.py                    # Flask 應用程式創建與配置
│   ├── config.py                 # 集中式配置管理模組
│   ├── database.py               # 資料庫操作與知識庫搜尋
│   ├── event_system.py           # 事件發布/訂閱系統
│   ├── initial_data.py           # 初始資料匯入工具
│   ├── linebot_connect.py        # LINE Bot 事件與後台路由
│   ├── llm_client.py             # OpenAI / Ollama LLM 客戶端抽象
│   ├── main.py                   # 對話流程與 RAG 結合
│   ├── rag/                      # SQL RAG 管線與檢索器
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── sql_retriever.py
│   └── reply.py                  # 常用指令與回覆模組
├── templates/                    # HTML 模板
├── static/                       # 靜態資源
├── .github/workflows/            # GitHub Actions 設定
├── Dockerfile                    # Docker 配置
├── requirements.txt              # 相依套件列表
└── Documentary.md                # 詳細專案文件
```

## 環境設定

### 必要環境變數

請建立 `.env` 檔案並設定以下環境變數：

```
# Flask 與服務基本設定
FLASK_DEBUG=False
HOST=127.0.0.1
PORT=5000
SECRET_KEY=your_secret_key               # 若未設定將改寫 SECRET_KEY_FILE
SECRET_KEY_FILE=data/secret_key.txt

# LINE Bot API
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
LINE_CHANNEL_SECRET=your_line_channel_secret

# LLM 供應商設定
LLM_PROVIDER=openai                      # 可選 openai / ollama
OPENAI_API_KEY=your_openai_api_key       # LLM_PROVIDER=openai 時必填
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=500
OPENAI_TIMEOUT=10
OLLAMA_BASE_URL=http://localhost:11434   # LLM_PROVIDER=ollama 時使用
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=30

# SQL Server 連線設定
DB_SERVER=your_sqlserver_host
DB_NAME=Project
DB_USER=                                 # 選填，Trusted Connection 時可留空
DB_PASSWORD=

# 管理後台
ADMIN_USERNAME=admin_username
ADMIN_PASSWORD=admin_password

# 驗證模式
VALIDATION_MODE=strict                   # loose 可在缺少變數時繼續啟動

# SQL RAG 設定 (選填)
ENABLE_SQL_RAG=true
SQL_RAG_TABLE=knowledge_documents
SQL_RAG_SEARCH_FIELDS=title,content,tags
SQL_RAG_MAX_DOCS=3
SQL_RAG_MAX_CHARS_PER_DOC=480
SQL_RAG_CONTEXT_HEADER=                 # 自訂知識庫前導文字，可留空
```

### 安裝相依套件

```bash
pip install -r requirements.txt
```

### 資料庫與知識庫初始化

1. 確保 SQL Server 已建立 `DB_NAME` 指定的資料庫。
2. 首次啟動應用程式時會自動建立對話、設備、知識庫等所需資料表。
3. 若需預載設備或知識文件，可於 `data/` 目錄放置 Excel，再透過 `initial_data.py` 匯入。

## 執行方式

### 本地開發

```bash
python -m src.app
```

預設會讀取 `certs/` 目錄下的自簽憑證並啟動 HTTPS。若需改用 HTTP，可在測試環境設定 `FLASK_DEBUG=True` 並於呼叫 `run_app()` 時提供 `ssl_context=None`。

### Docker 部署

```bash
docker build -t line-bot-project .
docker run -p 5000:5000 --env-file .env line-bot-project
```

Docker 映像檔採用官方 Python 3.11-slim 為基礎，並實作以下安全最佳實踐：
- 使用非 root 用戶運行應用程式。
- 移除不必要的套件以減少攻擊面。
- 適當設定檔案權限。

## 系統功能說明

### LINE Bot 指令

- **知識庫查詢**：任何技術問題皆會搭配 SQL RAG 檢索結果。
- **一般對話**：直接輸入問題，AI 將生成回應。
- **設備狀態**：輸入「設備狀態」或「機台狀態」查看所有設備概況。
- **設備詳情**：輸入「設備詳情 [設備名稱]」查看特定設備的詳細資訊。
- **訂閱設備**：輸入「訂閱設備」查看可用設備列表並進行訂閱。
- **我的訂閱**：輸入「我的訂閱」查看已訂閱的設備。
- **語言設定**：輸入「language:語言代碼」更改語言 (例如「language:en」切換至英文)。
- **幫助選單**：輸入「help」或「幫助」查看功能選單。

### 管理後台

1. 訪問 `/admin/login` 進行登入。
2. 查看系統統計資料、近期對話與設備狀態。
3. 可檢視個別使用者的完整對話歷史。

### 設備監控功能

- 監控黏晶機、打線機與切割機等半導體設備。
- 針對溫度、壓力、良率等指標進行異常偵測。
- 依嚴重程度自動發送 LINE 通知。
- 提供設備詳情查詢與訂閱功能。

### SQL RAG 知識庫

- 預設啟用 `ENABLE_SQL_RAG=true` 時會檢索 `knowledge_documents` 表格。
- 可透過 `SQL_RAG_SEARCH_FIELDS` 指定檢索欄位，並使用 `SQL_RAG_MAX_DOCS` / `SQL_RAG_MAX_CHARS_PER_DOC` 控制上下文長度。
- 使用 `SQL_RAG_CONTEXT_HEADER` 自訂回覆前綴文字，協助使用者辨識引用的知識來源。
- 若資料庫尚未準備好，可暫時設定 `ENABLE_SQL_RAG=false` 關閉此功能。

### 事件系統

- 使用發布/訂閱模式解耦各模組間的直接依賴。
- 允許不同模組對相同事件進行響應。
- 通過 `event_system.subscribe()` 註冊事件處理函數。
- 使用 `event_system.publish()` 發布事件到系統。

## CI/CD 工作流程

本專案使用 GitHub Actions 自動化下列流程：

1. **安全性掃描**：使用 Bandit 與 Safety 檢查程式碼與相依套件漏洞。
2. **測試**：執行單元測試與程式碼品質檢查。
3. **建置與部署**：建置 Docker 映像檔並發布。
4. **部署後安全掃描**：使用 Trivy 掃描容器映像檔。

## 安全考量

本專案實作多層次安全防護：

- API 金鑰與敏感資訊透過環境變數管理。
- 使用 Flask-Talisman 實作內容安全政策 (CSP)。
- 輸入驗證與清理，防止潛在攻擊。
- 非 root 使用者運行 Docker 容器。
- 實作請求速率限制，防止暴力攻擊。

## 問題排解

- **LINE 簽名驗證失敗**：檢查 LINE_CHANNEL_SECRET 設定。
- **OpenAI API 回應異常**：確認 API 金鑰與使用額度。
- **設備監控問題**：確認資料庫初始化與排程器狀態。
- **知識庫查詢沒有結果**：確認 `knowledge_documents` 表格是否有資料並檢查 `SQL_RAG_SEARCH_FIELDS`。

## 相關資源

- [LINE 開發者平台](https://developers.line.biz/zh-hant/)
- [OpenAI API 文件](https://platform.openai.com/docs/)
- [Flask 文件](https://flask.palletsprojects.com/)
