# 專案文件

## 概述

本專案整合 [LINE Messaging API](https://developers.line.biz/zh-hant/)、[OpenAI ChatCompletion](https://platform.openai.com/docs/guides/chat)、本地 Ollama 推論服務，以及以 SQL Server 為基礎的檢索式增強生成（RAG）知識庫，打造多功能智能助理。系統可在 LINE 平台上即時提供工程技術支援與諮詢，並可選擇性地引用企業知識庫內容來補強回答。同時仍保留半導體設備監控、異常警報推播與管理後台功能，提供完整的營運支援能力。

## 技術棧

- **Python 3.11**：專案主要開發語言，基於最新穩定版本。
- **Flask**：輕量級後端 Web 框架，處理 LINE Webhook、管理後台與 REST API。
- **line-bot-sdk v3.x**：整合 LINE Bot API，處理訊息接收與回覆。
- **OpenAI API / Ollama**：支援雲端 ChatGPT 與本地 LLM 推論流程。
- **SQL Server**：關聯式資料庫，儲存對話歷史、使用者偏好、設備數據與知識文件。
- **SQL RAG 管線**：以 SQL 檢索輔助 LLM 回覆，提升回答準確性。
- **Schedule**：簡易任務排程，用於定期設備監控（可依需求啟用）。
- **Flask-Talisman**：實作內容安全政策(CSP)與其他安全防護。
- **Docker**：容器化部署，確保環境一致性。
- **GitHub Actions**：CI/CD 流程自動化與安全掃描。

## 主要功能

1. **LINE 訊息處理**
   - 利用 `/callback` Webhook 接收並驗證 LINE 傳來的事件。
   - 根據使用者訊息，呼叫已設定的 LLM 服務產生回應。
   - 實作輸入驗證與清理，防止潛在的 XSS 攻擊。
   - 支援快速回覆、按鈕模板等 LINE 互動元素。

2. **智能對話與多語言支援**
   - 根據預先定義的系統提示與使用者輸入產生上下文相關的回覆。
   - 維護對話歷史紀錄，結合 SQL Server 與記憶體快取最佳化效能。
   - 支援繁體中文、簡體中文、英文、日文、韓文等多語言回覆。

3. **SQL RAG 知識庫整合**
   - 透過 `RAGPipeline` 與 `SQLRetriever` 自動搜尋 `knowledge_documents` 資料表。
   - 依據環境變數設定控制檢索欄位、文件數量與擷取長度。
   - 支援自訂回覆前綴文字與多語言提示，讓使用者辨識引用資訊來源。
   - 可透過 `ENABLE_SQL_RAG` 變數快速啟用 / 停用知識庫整合。

4. **半導體設備監控**
   - 即時監控黏晶機、打線機、切割機等設備運作狀態與關鍵指標。
   - 自動偵測溫度、壓力、轉速、良率等異常並分類嚴重程度。
   - 透過 LINE 機器人即時發送警報通知給相關負責人員。
   - 支援排程功能，定期檢查設備狀態並產生彙整報表。
   - 提供設備狀態查詢、詳細資訊檢視與訂閱管理功能。

5. **資料儲存與分析**
   - 使用 SQL Server 儲存對話歷史、使用者偏好與設備監控數據。
   - 內建分析模組，追蹤用戶行為與系統使用狀況。
   - 生成使用趨勢數據，包括每日訊息量、活躍用戶數等統計。
   - 支援關鍵字追蹤與分析，了解使用者主要關注的話題。
   - 儲存設備運行指標與警報歷史，便於後續分析與改善。

6. **管理後台**
   - 提供安全的管理員登入系統。
   - 展示系統使用統計，包括總對話數、用戶數等關鍵指標。
   - 查看個別使用者的完整對話歷史與近期對話摘要。
   - 監控系統狀態，顯示 OpenAI / Ollama / PowerBI 等外部連線情況。
   - 檢視設備監控概況與異常警報記錄。

7. **事件系統**
   - 實作輕量級事件發布 / 訂閱系統，解耦模組間的依賴。
   - 允許不同模組響應系統事件，如設備警報、使用者行為等。
   - 錯誤隔離機制確保事件處理失敗不影響整體系統運行。

8. **集中式配置與生命週期管理**
   - 使用 `config.py` 統一管理環境變數並提供驗證機制。
   - `app.py` 作為統一應用程序創建與配置入口，協助測試與部署。
   - 內容安全政策(CSP)、強制 HTTPS 與 ProxyFix 提供安全與相容性。
   - 支援嚴格與寬鬆模式切換，方便在測試期間快速啟動應用。

## 專案架構

```
.
├── src/
│   ├── __init__.py               # Python 包初始化
│   ├── app.py                    # Flask 應用程式創建與配置
│   ├── analytics.py              # 數據分析與統計模組
│   ├── config.py                 # 集中式配置管理模組
│   ├── database.py               # 資料庫操作與知識庫存取
│   ├── event_system.py           # 事件發布/訂閱系統
│   ├── initial_data.py           # 初始資料匯入工具
│   ├── linebot_connect.py        # LINE Bot 事件與後台路由
│   ├── llm_client.py             # LLM 客戶端抽象 (OpenAI / Ollama)
│   ├── main.py                   # 對話流程、輸入清理與 RAG 整合
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py           # RAG 管線
│   │   └── sql_retriever.py      # SQL 知識庫檢索器
│   └── reply.py                  # LINE Bot 指令回覆模組
├── templates/                    # HTML 模板
├── static/                       # 靜態資源
├── tests/                        # 單元測試
├── Dockerfile                    # Docker 部署設定檔
├── requirements.txt              # Python 套件相依列表
├── README.md                     # 專案簡介與快速上手指南
└── Documentary.md                # 專案詳細文件（本文件）
```

## 環境設定

請依照以下步驟配置執行環境，確保各模組順利運作：

1. **環境變數設定** － 依照 `.env.example` 建立 `.env` 檔案，設定下列必要變數：
   - **Flask 與服務：**
     - `FLASK_DEBUG`：是否啟用 Flask 除錯模式（建議生產環境設為 False）。
     - `PORT` / `HOST`：服務監聽設定。
     - `SECRET_KEY` 或 `SECRET_KEY_FILE`：Session 加密密鑰來源。
     - `VALIDATION_MODE`：`strict` / `loose`，控制驗證失敗時的行為。
   - **LINE Bot：**
     - `LINE_CHANNEL_ACCESS_TOKEN`、`LINE_CHANNEL_SECRET`。
   - **LLM 設定：**
     - `LLM_PROVIDER`：`openai` 或 `ollama`。
     - 若使用 OpenAI：`OPENAI_API_KEY`、`OPENAI_MODEL`、`OPENAI_MAX_TOKENS`、`OPENAI_TIMEOUT`。
     - 若使用 Ollama：`OLLAMA_BASE_URL`、`OLLAMA_MODEL`、`OLLAMA_TIMEOUT`。
   - **SQL Server：**
     - `DB_SERVER`、`DB_NAME`，依需求可補充 `DB_USER`、`DB_PASSWORD`。
   - **管理後台：**
     - `ADMIN_USERNAME`、`ADMIN_PASSWORD`。
   - **SQL RAG 相關（選填）：**
     - `ENABLE_SQL_RAG`：`true` / `false`。
     - `SQL_RAG_TABLE`、`SQL_RAG_SEARCH_FIELDS`、`SQL_RAG_MAX_DOCS`、`SQL_RAG_MAX_CHARS_PER_DOC`。
     - `SQL_RAG_CONTEXT_HEADER`：自訂知識庫前導文字。

2. **安裝依賴套件**
   ```bash
   pip install -r requirements.txt
   ```

3. **資料庫初始化**
   - 確保 SQL Server 已建立 `DB_NAME` 指定的資料庫。
   - 首次啟動應用程式時，`database.py` 會自動建立必要的表格。
   - 如需預載設備與知識資料，可於 `data/` 目錄放置 Excel，並透過 `initial_data.py` 匯入。

## 知識庫（SQL RAG）設計

- 預設使用的資料表為 `knowledge_documents`，欄位包含：
  - `doc_id`（主鍵）、`title`、`content`、`tags`、`source`、`last_updated`。
- `database.Database.search_knowledge_documents()` 支援以多欄位 LIKE 檢索，並將結果回傳給 RAG 管線。
- `database.Database.upsert_knowledge_document()` 可新增或更新知識文件，方便建立管理介面或批次匯入工具。
- 可透過環境變數調整搜尋欄位與回傳文件數量，上下文會插入至 LLM 對話訊息中以補強回答。

## 執行應用

1. **本地執行**
   ```bash
   python -m src.app
   ```
   - 預設啟用 HTTPS，憑證位於 `certs/` 目錄，可透過環境變數覆蓋。
   - LINE Webhook 接收端點為 `/callback`。
   - 管理後台登入頁面：`https://localhost:5000/admin/login`。

2. **Docker 部署**
   ```bash
   docker build -t capstone-project .
   docker run -p 5000:5000 --env-file .env capstone-project
   ```
   - 以 Python 3.11-slim 為基礎映像檔，使用非 root 用戶執行。
   - 可透過掛載外部憑證與設定檔，自訂部署環境。

## CI/CD

1. **持續整合與部署** － GitHub Actions 工作流程包含：
   - **安全性掃描**：Bandit、Safety 分別檢查程式碼與套件漏洞。
   - **測試階段**：執行 `pytest` 與 flake8 等品質檢查，並上傳覆蓋率報告。
   - **建置與部署**：使用 Docker Buildx 建置映像檔並以 Trivy 掃描漏洞。
   - **部署後安全掃描**：再次執行 Trivy，將結果上傳至 GitHub Security。

## 使用指南

### LINE Bot 功能

- **一般對話 / 技術諮詢**：直接輸入問題，AI 會自動結合 RAG 給出建議。
- **設備狀態查詢**：輸入「設備狀態」或「機台狀態」查看各設備的運作概況。
- **設備詳情查詢**：輸入「設備詳情 [設備名稱]」取得特定設備的指標與警報紀錄。
- **訂閱設備**：輸入「訂閱設備」查看列表，或「訂閱設備 [設備ID]」訂閱特定設備。
- **取消訂閱**：輸入「取消訂閱 [設備ID]」。
- **我的訂閱**：輸入「我的訂閱」查看目前訂閱項目。
- **語言設定**：輸入「language:語言代碼」更改語言（如 `language:en`）。
- **幫助選單**：輸入「help」「幫助」或「使用說明」獲取操作指引。
- **關於系統**：輸入「關於」或「about」查看系統簡介與支援的 LLM 供應商。

### 管理後台

1. 訪問 `/admin/login` 並以環境變數設定的帳號密碼登入。
2. 儀表板顯示系統統計、設備監控概況與近期對話摘要。
3. 可檢視個別使用者對話、語言偏好與最後活動時間。
4. 支援 PowerBI 報表嵌入與外部服務狀態檢視。

## 半導體設備監控功能

1. **支援設備類型**
   - **黏晶機 (Die Bonder)**：監控溫度、壓力、Pick 準確率、良率等指標。
   - **打線機 (Wire Bonder)**：監控金絲張力、壓力、良率等指標。
   - **切割機 (Dicer)**：監控轉速、冷卻水溫、切割精度、良率等指標。

2. **監控機制**
   - 使用排程器定期檢查設備狀態（預設每 5 分鐘）。
   - 自動比較當前指標值與設定的閾值，並依偏差程度決定警報嚴重性。
   - 偵測長時間運行作業，避免設備過度使用。

3. **警報與通知**
   - 針對不同嚴重程度的異常使用視覺標識（⚠️ / 🔴 / 🚨）。
   - 自動發送 LINE 通知給設備負責人或區域管理員。
   - 記錄所有警報至資料庫，提供歷史追蹤與分析。
   - 可透過 OpenAI 提供異常原因與建議處理方式。

4. **使用者訂閱**
   - 使用者可訂閱特定設備的警報通知，並自訂通知層級。
   - 依據使用者負責區域自動分配通知對象。

5. **查詢指令**
   - 「設備狀態」：顯示所有設備的狀態摘要。
   - 「設備詳情 [設備名稱]」：顯示指定設備的詳細資訊、指標與警報。

6. **資料庫結構重點**
   - `equipment`：儲存設備基本資訊（ID、名稱、類型、位置等）。
   - `equipment_metrics`：儲存設備監測指標與閾值。
   - `equipment_operation_logs`：記錄設備運轉情形。
   - `alert_history`：記錄警報歷史、處理狀態與備註。
   - `user_equipment_subscriptions`：儲存使用者與設備的訂閱關係。

## 事件系統

- 透過 `event_system.subscribe()` 與 `event_system.publish()` 提供模組間鬆耦合的溝通方式。
- 支援任意數量的事件處理器，並在失敗時記錄錯誤而不影響其他處理器。
- 可用於擴充通知機制、紀錄使用者行為或觸發自訂自動化流程。

## 集中式應用程序管理

1. **統一入口點**：`app.py` 提供 `create_app()` 與 `run_app()`，方便測試與多實例部署。
2. **環境變數驗證**：啟動時執行 `Config.validate()`，可依 `VALIDATION_MODE` 決定是否在缺少設定時停止。
3. **安全性配置**：集中設定 Flask-Talisman、ProxyFix、Session Cookie 屬性與 HTTPS。
4. **資源釋放**：可在 `teardown_appcontext` 中註冊清理流程（如排程器停止）。

## 資料分析功能

- **事件追蹤**：記錄訊息發送、語言變更等行為，支援使用者 ID 關聯。
- **每日統計**：統計每日訊息數、活躍使用者數與平均回應時間。
- **關鍵字分析**：追蹤使用者訊息中的關鍵字，分析熱門議題。
- **使用趨勢**：生成趨勢數據，評估使用者活躍度與留存率。
- **語言偏好分析**：記錄與分析不同語言使用者的分佈情況。
- **設備監控分析**：追蹤設備異常發生頻率與模式，輔助預防性維護。

## 安全性考量

1. **API 與密鑰管理**
   - 所有敏感資訊透過環境變數管理，不直接寫入程式碼。
   - 實作 LINE 簽名驗證與 OpenAI/Ollama 連線錯誤處理。

2. **網頁安全**
   - 使用 Flask-Talisman 實作嚴格的內容安全政策 (CSP)。
   - 設定 HttpOnly、Secure 等 Cookie 屬性並強制 HTTPS。
   - 實作 IP 請求速率限制，降低暴力攻擊風險。

3. **輸入驗證**
   - `sanitize_input` 函數移除危險字元，避免指令注入或 XSS。
   - 對 API 參數與資料庫操作進行格式驗證。

4. **容器安全**
   - Docker 映像檔以非 root 使用者執行並移除不必要套件。
   - 使用 Trivy、Safety 等工具持續掃描漏洞。

5. **持續安全監控**
   - CI/CD 流程包含自動化安全檢查與報告。
   - 支援在寬鬆模式下啟動以利偵錯，同時透過日誌提示缺失設定。

## 常見問題與疑難排解

1. **簽名驗證失敗**
   - 確認 `LINE_CHANNEL_SECRET` 與 LINE 開發者控制台設定一致。
   - 如使用 ngrok 進行測試，確認 HTTPS 標頭未被移除。

2. **LLM 回覆異常**
   - 驗證 `OPENAI_API_KEY` 或本地 Ollama 服務是否可用。
   - 檢查 `LLM_PROVIDER`、模型名稱與逾時設定是否正確。
   - 查看應用程式日誌以取得詳細錯誤訊息。

3. **SQL RAG 沒有輸出**
   - 確認 `ENABLE_SQL_RAG` 為 `true` 且資料表存在資料。
   - 檢查 `SQL_RAG_SEARCH_FIELDS` 是否包含實際欄位。
   - 查看資料庫日誌是否有連線或權限問題。

4. **設備監控問題**
   - 確認資料表已初始化並有正確閾值設定。
   - 檢查排程器是否啟動，可透過日誌確認背景任務運行狀態。
   - 使用 `tests/test_equipment_alert.py` 模擬異常驗證流程。

5. **Docker 部署問題**
   - 確認 `.env` 中包含所有必要的環境變數。
   - 檢查容器日誌，確認外部 API（OpenAI、LINE、PowerBI）可連線。
   - 如需自訂憑證，請將檔案掛載到容器並更新相關環境變數。

6. **資料庫連線問題**
   - 確認 SQL Server 連線字串與權限設定正確。
   - 查看資料庫錯誤日誌，或使用 SQL Server Profiler 偵錯。
   - 如需重置資料庫，可重新建立資料表或回復備份。

7. **管理後台登入問題**
   - 確認 `ADMIN_USERNAME`、`ADMIN_PASSWORD` 與 `SECRET_KEY` 設定正確。
   - 檢查瀏覽器 Cookie 設定是否允許寫入。
   - 忘記密碼時可調整環境變數重新部署。

8. **事件系統問題**
   - 確認事件訂閱與發布參數一致。
   - 審查日誌中的事件處理錯誤訊息，避免循環依賴。

9. **應用啟動失敗**
   - 檢查 `PORT` 是否被其他應用程式佔用。
   - 如遇 SSL 錯誤，確認憑證路徑與權限。
   - 在 `VALIDATION_MODE=loose` 下啟動，可協助辨識缺少的環境變數。

