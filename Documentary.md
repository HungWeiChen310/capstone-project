# 專案文件

## 專案概述

本專案整合 [LINE Messaging API](https://developers.line.biz/zh-hant/) 與 [OpenAI ChatCompletion](https://platform.openai.com/docs/guides/chat)（ChatGPT）的功能，打造一個能夠在 LINE 平台上即時提供工程技術支援與諮詢的智能助理。同時，專案擴展為 PowerBI 報表嵌入功能，讓使用者可以在網頁上直觀地展示與分析數據。

## 技術棧

- **Python 3.11**：專案主要開發語言，基於最新穩定版本
- **Flask**：輕量級後端 Web 框架，處理 API 請求與網頁展示
- **line-bot-sdk v3.x**：整合 LINE Bot API，處理訊息接收與回覆
- **OpenAI API**：調用 ChatGPT 模型生成智能回覆
- **PowerBI API**：嵌入 PowerBI 報表以進行數據展示
- **Flask-Talisman**：實作內容安全政策(CSP)與其他安全防護
- **pytest**：進行單元測試與覆蓋率分析
- **Docker**：容器化部署，確保環境一致性
- **GitHub Actions**：CI/CD 流程自動化與安全掃描

## 主要功能

1. **LINE 訊息處理**  
   - 利用 `/callback` Webhook 接收並驗證 LINE 傳來的事件
   - 根據使用者訊息，調用 OpenAI ChatCompletion API 生成專業回覆，再透過 LINE Bot API 回傳結果
   - 實作訊息輸入驗證與清理，防止潛在的 XSS 攻擊

2. **智能對話生成**  
   - 根據預先定義的系統提示，結合使用者輸入生成邏輯嚴謹且具體建議的回應
   - 維護對話歷史紀錄，提供上下文相關的回覆
   - 透過 sanitize_input 函數進行輸入清理，增強安全性

3. **PowerBI 報表整合**  
   - 透過 OAuth2 客戶端憑證流程，取得 PowerBI API 存取權杖與嵌入 Token
   - 於 `/powerbi` 路由提供網頁介面展示 PowerBI 報表
   - 實作基於 IP 的請求限制，防止潛在的 DoS 攻擊

4. **網站安全性強化**
   - 使用 Flask-Talisman 實作內容安全政策(CSP)，限制資源載入來源
   - 實作安全的 Cookie 設定（HttpOnly, Secure flags）
   - 設置適當的特性政策(Feature-Policy)，限制敏感 API 的使用
   - 處理代理標頭，確保在代理伺服器後方運作正常

5. **集中式配置管理**
   - 使用 config.py 統一管理所有環境變數與設定
   - 實作環境變數驗證機制，確保必要的設定存在
   - 集中式日誌配置，提供一致的日誌格式與級別

## 專案架構

```
.
├── src/
│   ├── __init__.py             # Python 包初始化檔案
│   ├── config.py               # 集中式配置管理模組
│   ├── main.py                 # 核心業務邏輯與 OpenAI 服務封裝
│   ├── linebot_connect.py      # Flask 應用與 LINE Bot 事件處理
│   └── powerbi_integration.py  # PowerBI API 整合與報表嵌入模組
├── tests/
│   ├── __init__.py             # 測試包初始化檔案
│   ├── conftest.py             # pytest 配置與共用 fixtures
│   ├── test_main.py            # 測試 OpenAI 服務及回覆函式
│   ├── test_linebot_connect.py # 測試 LINE Bot 事件處理與簽名驗證
│   └── test_powerbi_integration.py # 測試 PowerBI 整合模組
├── templates/
│   ├── index.html              # 服務狀態頁面
│   └── powerbi.html            # PowerBI 報表展示頁面
├── .github/
│   └── workflows/
│       └── main.yml            # GitHub Actions CI/CD 設定檔
├── .env.example                # 環境變數範例檔案
├── requirements.txt            # Python 套件相依列表
├── Dockerfile                  # Docker 部署設定檔
├── README.md                   # 專案簡介與快速上手指南
├── Documentary.md              # 專案詳細文件（本文件）
├── pytest.py                   # pytest 設定檔
└── .gitignore                  # 忽略檔案清單
```

## 環境設定

請依照以下步驟配置執行環境，確保各模組順利運作：

1. **環境變數設定**  
   依照 `.env.example` 建立 `.env` 檔案，設定下列必要環境變數：
   - **OpenAI 相關：**
     - `OPENAI_API_KEY`：你的 OpenAI API 金鑰
   - **LINE Bot 相關：**
     - `LINE_CHANNEL_ACCESS_TOKEN`：LINE Bot 存取金鑰
     - `LINE_CHANNEL_SECRET`：LINE Bot 密鑰
   - **PowerBI 相關：**
     - `POWERBI_CLIENT_ID`：Azure AD 應用程式用戶端 ID
     - `POWERBI_CLIENT_SECRET`：Azure AD 應用程式用戶端密碼
     - `POWERBI_TENANT_ID`：Azure 租戶 ID
     - `POWERBI_WORKSPACE_ID`：PowerBI 工作區 ID
     - `POWERBI_REPORT_ID`：PowerBI 報表 ID
   - **應用程式設定：**
     - `FLASK_DEBUG`：是否啟用 Flask 除錯模式（建議生產環境設為 False）
     - `PORT`：應用程式監聽的埠號（預設 5000）

2. **安裝依賴套件**  
   執行以下指令安裝所需套件：
   ```bash
   pip install -r requirements.txt
   ```

## 執行應用

1. **本地執行**  
   啟動 Flask 應用：
   ```bash
   python src/linebot_connect.py
   ```
   - LINE Webhook 接收端點為 `/callback`
   - PowerBI 報表嵌入頁面：`http://localhost:5000/powerbi`
   - 服務狀態頁面：`http://localhost:5000/`

2. **Docker 部署**  
   使用 Docker 部署，確保環境一致性：
   ```bash
   docker build -t capstone-project .
   docker run -p 5000:5000 \
     -e OPENAI_API_KEY="your_openai_api_key" \
     -e LINE_CHANNEL_ACCESS_TOKEN="your_line_access_token" \
     -e LINE_CHANNEL_SECRET="your_line_channel_secret" \
     -e POWERBI_CLIENT_ID="your_powerbi_client_id" \
     -e POWERBI_CLIENT_SECRET="your_powerbi_client_secret" \
     -e POWERBI_TENANT_ID="your_powerbi_tenant_id" \
     -e POWERBI_WORKSPACE_ID="your_powerbi_workspace_id" \
     -e POWERBI_REPORT_ID="your_powerbi_report_id" \
     capstone-project
   ```

   Docker 映像檔採用官方 Python 3.11-slim 為基礎，並實作以下安全最佳實踐：
   - 使用非 root 用戶運行應用程式
   - 移除不必要的套件以減少攻擊面
   - 適當設定檔案權限

## 測試與 CI/CD

1. **單元測試**  
   使用 pytest 執行所有單元測試，確保各模組功能正常：
   ```bash
   pytest
   ```
   
   若要產生覆蓋率報告：
   ```bash
   pytest --cov=src --cov-report=xml
   ```

   測試涵蓋：
   - LINE Bot 事件處理與簽名驗證
   - OpenAI 回覆生成服務
   - PowerBI API 整合功能

2. **持續整合與部署**  
   GitHub Actions 自動化流程包含以下階段：

   a. **安全性掃描**
   - Bandit：掃描 Python 程式碼中的安全漏洞
   - Safety：檢查相依套件中的已知漏洞

   b. **測試階段**
   - 執行 flake8 程式碼品質檢查
   - 執行 pytest 單元測試與覆蓋率分析
   - 上傳測試覆蓋率報告至 Codecov

   c. **建置與部署**
   - 使用 Docker Buildx 建置最佳化的容器映像檔
   - 推送映像檔至 Docker Hub
   - 實作映像檔快取以加速建置流程

   d. **部署後安全掃描**
   - 使用 Trivy 掃描容器映像檔中的漏洞
   - 將掃描結果上傳至 GitHub Security

## 安全性考量

本專案實作多層次的安全防護機制：

1. **API 安全**
   - 所有 API 金鑰與敏感資訊均通過環境變數管理，不直接寫入程式碼
   - 實作 LINE 簽名驗證，確保請求來源
   - PowerBI 整合使用 OAuth2 客戶端憑證流程，不暴露敏感資訊

2. **網頁安全**
   - 使用 Flask-Talisman 實作嚴格的內容安全政策(CSP)
   - 設定安全的 Cookie 屬性（HttpOnly, Secure）
   - 強制使用 HTTPS 連線

3. **輸入驗證與清理**
   - 使用 sanitize_input 函數處理使用者輸入，防止潛在的注入攻擊
   - 實作請求速率限制，防止暴力攻擊

4. **容器安全**
   - 使用非 root 用戶運行容器化應用
   - 移除不必要的套件以減少攻擊面
   - 最小化容器映像檔大小

5. **持續安全監控**
   - 在 CI/CD 流程中包含自動化安全掃描
   - 使用 Trivy 檢測容器映像檔中的漏洞

## 常見問題與疑難排解

1. **簽名驗證失敗**  
   - 確認 `LINE_CHANNEL_SECRET` 與 LINE 開發者控制台設定一致
   - 檢查是否使用了 ngrok 等工具進行本地測試，可能影響 HTTPS 標頭

2. **API 回覆異常**  
   - 驗證 `OPENAI_API_KEY` 是否正確，並檢查 API 調用是否超出使用配額
   - 查看應用程式日誌中的詳細錯誤訊息
   - 確認網路連線是否穩定，特別是在容器化環境中

3. **PowerBI 嵌入報表失敗**  
   - 檢查所有 PowerBI 相關環境變數是否正確配置
   - 確認 Azure AD 應用程式已被授予適當的 PowerBI API 權限
   - 檢查 PowerBI 工作區與報表 ID 是否正確
   - 確認報表有公開存取權限或適當的權限設置

4. **Docker 部署問題**
   - 確認已正確設置所有必要的環境變數
   - 檢查容器日誌以獲取詳細的錯誤訊息
   - 確認 Docker 主機的網路設定允許容器連接外部 API（OpenAI、LINE、PowerBI）

5. **安全警告與 CSP 違規**
   - 如需調整內容安全政策，修改 `linebot_connect.py` 中的 `csp` 字典
   - 對於 PowerBI 嵌入頁面，可能需要允許來自 app.powerbi.com 和 cdn.powerbi.com 的資源

## 開發與擴展指南

1. **新增功能**
   - 遵循模組化設計，將新功能放置在適當的模組中
   - 為新功能編寫單元測試，確保代碼覆蓋率

2. **更新相依套件**
   - 定期更新 requirements.txt 中的套件版本
   - 使用 GitHub Actions 中的 Safety 檢查來監控相依套件的漏洞

3. **擴展 AI 功能**
   - 可在 main.py 中修改 OpenAI 的提示和參數，以優化回覆品質
   - 考慮實作頻率限制，避免過度使用 OpenAI API

4. **本地開發提示**
   - 設置 FLASK_DEBUG=True 以啟用熱重載與詳細錯誤訊息
   - 使用 ngrok 等工具為本地伺服器建立公開 URL，以便測試 LINE Webhook
