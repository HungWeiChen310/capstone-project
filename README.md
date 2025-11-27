# LINE Bot + Ollama (Llama 3) 整合系統

## 主要功能

- **LINE Bot 智能對話**：接收使用者訊息，利用 Ollama (Llama 3 / gpt-oss:20b) 生成專業、具實踐性的回應
- **文件檢索增強 (RAG)**：整合本地專案文件與程式碼，透過 ChromaDB 與 SentenceTransformers 進行檢索增強生成，提供更準確的答案
- **半導體設備監控**：即時監控切割機設備的運作狀態，自動偵測異常並發送警報
- **多語言支援**：支援繁體中文、簡體中文、英文、日文與韓文等多種語言
- **事件系統**：實作輕量級事件發布/訂閱系統，解耦模組間的依賴
- **資料分析與儲存**：使用 SQL Server 資料庫儲存對話歷史、使用者偏好與設備監控資料
- **管理後台**：提供系統監控與管理功能，包括使用統計、對話記錄查詢等

## 技術棧

- **Python 3.11**：主要開發語言
- **Flask**：輕量級網頁框架
- **LINE Bot SDK v3.x**：處理 LINE 訊息互動
- **Ollama**：本地運行的大型語言模型 (Llama 3 / gpt-oss:20b)
- **SQL Server**：關聯式資料庫管理系統
- **ChromaDB**：向量資料庫，用於 RAG 知識庫儲存
- **SentenceTransformers**：用於生成語意向量 (Embedding)
- **Flask-Talisman**：網頁安全增強
- **Schedule**：設備監控排程
- **Docker**：容器化部署
- **GitHub Actions**：CI/CD 自動化工作流程

## 專案架構

```
.
├── app.py                      # Flask 應用程式進入點
├── src/                          # 主要源碼
│   ├── __init__.py               # Python 包初始化
│   ├── routes/                   # 路由模組 (Blueprints)
│   ├── services/                 # 業務邏輯模組
│   ├── utils.py                  # 工具函數
│   ├── config.py                 # 集中式配置管理
│   ├── main.py                   # 核心邏輯與 Ollama 服務
│   ├── rag.py                    # RAG 檢索增強生成邏輯 (ChromaDB + SentenceTransformers)
│   ├── linebot_connect.py        # LINE Bot 事件處理
│   ├── database.py               # 資料庫操作
│   ├── analytics.py              # 數據分析模組
│   ├── equipment_monitor.py      # 設備監控與異常偵測
│   ├── equipment_scheduler.py    # 設備監控排程器
│   ├── event_system.py           # 事件發布/訂閱系統
│   └── initial_data.py           # 初始資料生成
├── rag_db/                       # ChromaDB 向量資料庫儲存目錄
├── templates/                    # HTML 模板
├── .github/workflows/            # GitHub Actions 設定
├── Dockerfile                    # Docker 配置
├── requirements.txt              # 相依套件列表
└── Documentary.md                # 詳細專案文件
```

## 環境設定

### 必要環境變數

請建立 `.env` 檔案並設定以下環境變數：

```
# 一般設定
FLASK_DEBUG=False
PORT=5000
# HOST=127.0.0.1 (非必要，視部署環境而定)
SECRET_KEY=your_secret_key
SECRET_KEY_FILE=data/secret_key.txt

# Ollama 設定 (本地 AI 模型)
OLLAMA_HOST=120.105.18.33
OLLAMA_PORT=11434
OLLAMA_MODEL=gpt-oss:20b
OLLAMA_TIMEOUT=30.0

# LINE Bot API
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
LINE_CHANNEL_SECRET=your_line_channel_secret

# 資料庫設定 (SQL Server)
DB_SERVER=localhost
DB_NAME=Project
DB_USER=your_db_user
DB_PASSWORD=your_db_password

# 管理後台
ADMIN_USERNAME=admin_username
ADMIN_PASSWORD=admin_password
```

### RAG 設定

系統使用 `ChromaDB` 與 `SentenceTransformers` (`paraphrase-multilingual-mpnet-base-v2`) 自動將專案中的文件與資料庫內容載入為知識庫。
可依需求透過以下環境變數調整行為：

- `ENABLE_RAG` / `ENABLE_RAG_DB`：是否啟用檢索增強（預設為 `true`）
- `RAG_SOURCE_PATHS`：自訂知識庫來源路徑，使用作業系統的路徑分隔符號分隔多個路徑
- `RAG_CHUNK_SIZE`：切分文件時的片段大小（預設 500）
- `RAG_CHUNK_OVERLAP`：片段重疊字元數（預設 100）
- `RAG_DB_TOP_K`：檢索時包含的資料庫記錄數量（預設 3）
- `RAG_TOP_K`：每次檢索返回的總片段數量
- `RAG_MIN_SCORE`：相似度門檻值

### 安裝相依套件

```bash
pip install -r requirements.txt
```

## 執行方式

### 本地開發

```bash
python app.py
```

### Docker 部署

```bash
docker build -t line-bot-project .
docker run -p 5000:5000 --env-file .env line-bot-project
```

Docker 映像檔採用官方 Python 3.11-slim 為基礎，並實作以下安全最佳實踐：
- 使用非 root 用戶運行應用程式
- 移除不必要的套件以減少攻擊面
- 適當設定檔案權限

## 系統功能說明

### LINE Bot 指令

- **一般對話**：直接輸入問題，AI 將生成回應
- **設備狀態**：輸入「設備狀態」或「機台狀態」查看所有設備概況
- **設備詳情**：輸入「設備詳情 [設備名稱]」查看特定設備的詳細資訊
- **訂閱設備**：輸入「訂閱設備」查看可用設備列表並進行訂閱
- **我的訂閱**：輸入「我的訂閱」查看已訂閱的設備
- **語言設定**：輸入「language:語言代碼」更改語言 (例如「language:en」切換至英文)
- **幫助選單**：輸入「help」或「幫助」查看功能選單

### 管理後台

1. 訪問 `/admin/login` 進行登入
2. 查看系統統計資料、近期對話與設備狀態
3. 可檢視個別使用者的完整對話歷史

### 設備監控功能

- 監控三種半導體設備：黏晶機、打線機與切割機
- 針對溫度、壓力、良率等指標進行異常偵測
- 依嚴重程度自動發送 LINE 通知
- 提供設備詳情查詢功能

### 事件系統

- 使用發布/訂閱模式解耦各模組間的直接依賴
- 允許不同模組對相同事件進行響應
- 通過 `event_system.subscribe()` 註冊事件處理函數
- 使用 `event_system.publish()` 發布事件到系統

## CI/CD 工作流程

本專案使用 GitHub Actions 自動化下列流程：

1. **安全性掃描**：使用 Bandit 與 Safety 檢查程式碼與相依套件漏洞
2. **測試**：執行單元測試與程式碼品質檢查
3. **建置與部署**：建置 Docker 映像檔並發布
4. **部署後安全掃描**：使用 Trivy 掃描容器映像檔

## 安全考量

本專案實作多層次安全防護：

- API 金鑰與敏感資訊透過環境變數管理
- 使用 Flask-Talisman 實作內容安全政策(CSP)
- 輸入驗證與清理，防止潛在攻擊
- 非 root 使用者運行 Docker 容器
- 實作請求速率限制，防止暴力攻擊

## 問題排解

- **LINE 簽名驗證失敗**：檢查 LINE_CHANNEL_SECRET 設定
- **Ollama 連線異常**：確認 OLLAMA_HOST 與 OLLAMA_PORT 設定，並確保 Ollama 服務已啟動且模型 (gpt-oss:20b) 已下載
- **設備監控問題**：確認資料庫初始化與排程器狀態

## 相關資源

- [LINE 開發者平台](https://developers.line.biz/zh-hant/)
- [Ollama 官方網站](https://ollama.com/)
- [Flask 文件](https://flask.palletsprojects.com/)

更多詳細資訊，請參閱專案中的 [Documentary.md](Documentary.md)
