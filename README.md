# LINE Bot + OpenAI + PowerBI 整合專案

本專案整合了 LINE Messaging API 與 OpenAI ChatGPT 的功能，打造一個能夠在 LINE 平台上即時提供工程技術支援與諮詢的智能助理。同時，專案也擴展了 PowerBI 報表嵌入功能，讓使用者可以在網頁上直觀地展示與分析數據。

## 主要功能

- **LINE Bot 智能對話**  
  接收使用者訊息，利用 ChatGPT 生成專業、具實踐性的回應，並回傳給使用者。

- **PowerBI 報表嵌入**  
  整合 PowerBI API，使用 OAuth2 客戶端憑證流程取得存取權杖與嵌入 Token，並在網頁上展示 PowerBI 報表。

- **安全性與合規性**  
  實作內容安全政策、XSS 防護、安全 Cookie 處理以及適當的輸入過濾。

- **完整測試與 CI/CD**  
  使用 pytest 進行單元測試，並利用 GitHub Actions 自動化測試、Docker 建置與部署，確保專案品質與穩定性。

## 技術棧

- **後端框架**：Flask
- **LINE 整合**：LINE Bot SDK
- **AI 整合**：OpenAI API（GPT-3.5 Turbo）
- **數據視覺化**：PowerBI API
- **測試框架**：pytest
- **CI/CD 工具**：GitHub Actions
- **容器化**：Docker

## 專案架構

```
.
├── src/
│   ├── main.py                 # 核心業務邏輯與 OpenAI 服務封裝
│   ├── linebot_connect.py      # Flask 應用與 LINE Bot 事件處理
│   ├── powerbi_integration.py  # PowerBI API 整合與報表嵌入模組
│   └── config.py               # 集中式配置管理
├── tests/
│   ├── test_main.py            # 測試 OpenAI 服務及回覆函式
│   ├── test_linebot_connect.py # 測試 LINE Bot 事件處理與簽名驗證
│   └── test_powerbi_integration.py # 測試 PowerBI 整合模組
├── templates/
│   ├── index.html              # 服務狀態頁面
│   └── powerbi.html            # PowerBI 報表展示頁面
├── requirements.txt            # Python 套件相依列表
├── .github/workflows/main.yml  # GitHub Actions CI/CD 設定檔
├── Dockerfile                  # Docker 部署設定檔
├── README.md                   # 專案簡介與快速上手指南
├── Documentary.md              # 專案詳細文件
└── .gitignore                  # 忽略檔案清單
```

## 環境設定

請依照以下設定環境變數，確保所有模組均能正確運作：

- **OpenAI 相關：**
  - `OPENAI_API_KEY`：您的 OpenAI API 金鑰

- **LINE Bot 相關：**
  - `LINE_CHANNEL_ACCESS_TOKEN`：LINE Bot 存取金鑰
  - `LINE_CHANNEL_SECRET`：LINE Bot 密鑰

- **PowerBI 相關：**
  - `POWERBI_CLIENT_ID`：Azure AD 註冊應用程式的用戶端 ID
  - `POWERBI_CLIENT_SECRET`：Azure AD 註冊應用程式的用戶端密碼
  - `POWERBI_TENANT_ID`：您的 Azure 租戶 ID
  - `POWERBI_WORKSPACE_ID`：PowerBI 工作區（群組）ID
  - `POWERBI_REPORT_ID`：PowerBI 報表 ID

## 安裝步驟

1. **複製專案儲存庫**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **安裝相依套件**
   ```bash
   pip install -r requirements.txt
   ```

3. **設定環境變數**
   在專案根目錄建立 `.env` 檔案，填入上述所需的環境變數，或將它們設定為系統環境變數。

## 執行專案

### 本地開發
啟動 Flask 伺服器：
```bash
python src/linebot_connect.py
```
- LINE Webhook 接收端點：`/callback`
- PowerBI 報表嵌入頁面：`http://localhost:5000/powerbi`
- 服務狀態頁面：`http://localhost:5000/`

### Docker 部署
若使用 Docker 部署：
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

## 測試與 CI/CD

### 單元測試
使用 pytest 執行所有單元測試：
```bash
pytest
```

測試涵蓋：
- LINE Bot 事件處理與簽名驗證
- OpenAI 回覆生成
- PowerBI API 整合

### 持續整合
GitHub Actions 工作流程（`.github/workflows/main.yml`）自動執行：
1. **安全性掃描**：運行 Bandit 和 Safety 檢查安全漏洞
2. **測試**：安裝相依套件並執行 pytest 測試（含覆蓋率報告）
3. **建置與部署**：建立並推送 Docker 映像檔以便生產環境部署
4. **部署安全掃描**：使用 Trivy 檢查 Docker 映像檔的漏洞

## 疑難排解

- **LINE 簽名驗證失敗**
  - 確認 `LINE_CHANNEL_SECRET` 與 LINE 開發者控制台設定一致。

- **API 回覆異常**
  - 檢查 `OPENAI_API_KEY` 是否正確，並確認未超出使用配額。

- **PowerBI 報表嵌入失敗**
  - 確保所有 PowerBI 相關環境變數已正確配置。
  - 檢查 Azure AD 註冊與權限設定是否正確。

## 相關資源

- [LINE 官方帳號管理](https://manager.line.biz/)
- [LINE API 文件（中文）](https://developers.line.biz/zh-hant/)
- [OpenAI API 文件](https://platform.openai.com/docs/)
- [PowerBI API 文件](https://docs.microsoft.com/zh-tw/power-bi/developer/)

## 詳細文件

更多詳細技術資訊與整合步驟，請參閱:
- [Documentary.md](Documentary.md) – 專案詳細文件
