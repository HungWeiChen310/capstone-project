# LINE Bot + OpenAI 整合系統

## 主要功能

- **LINE Bot 智能對話**：接收使用者訊息，利用 ChatGPT 生成專業、具實踐性的回應
- **半導體設備監控**：即時監控黏晶機、打線機、切割機等設備的運作狀態，自動偵測異常並發送警報
- **多語言支援**：支援繁體中文、簡體中文、英文、日文與韓文等多種語言
- **事件系統**：實作輕量級事件發布/訂閱系統，解耦模組間的依賴
- **資料分析與儲存**：使用 SQL Server 資料庫儲存對話歷史、使用者偏好與設備監控資料
- **管理後台**：提供系統監控與管理功能，包括使用統計、對話記錄查詢等

## 技術棧

- **Python 3.11**：主要開發語言
- **Flask**：輕量級網頁框架
- **LINE Bot SDK v3.x**：處理 LINE 訊息互動
- **OpenAI API**：接入 ChatGPT 模型
- **SQL Server**：關聯式資料庫管理系統
- **Flask-Talisman**：網頁安全增強
- **Schedule**：設備監控排程
- **Docker**：容器化部署
- **GitHub Actions**：CI/CD 自動化工作流程

## 專案架構

```
.
├── src/                          # 主要源碼
│   ├── __init__.py               # Python 包初始化
│   ├── app.py                    # Flask 應用程式創建與配置
│   ├── config.py                 # 集中式配置管理模組
│   ├── main.py                   # 核心邏輯與 OpenAI 服務
│   ├── linebot_connect.py        # LINE Bot 事件處理
│   ├── database.py               # 資料庫操作
│   ├── analytics.py              # 數據分析模組
│   ├── equipment_monitor.py      # 設備監控與異常偵測
│   ├── equipment_scheduler.py    # 設備監控排程器
│   ├── event_system.py           # 事件發布/訂閱系統
│   └── initial_data.py           # 初始資料生成
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
SECRET_KEY=your_secret_key

# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# LINE Bot API
LINE_CHANNEL_ACCESS_TOKEN=your_line_channel_access_token
LINE_CHANNEL_SECRET=your_line_channel_secret

# 管理後台
ADMIN_USERNAME=admin_username
ADMIN_PASSWORD=admin_password
```

### 安裝相依套件

```bash
pip install -r requirements.txt
```

## 執行方式

### 本地開發

```bash
python -m src.app
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
- **OpenAI API 回應異常**：確認 API 金鑰與使用額度
- **設備監控問題**：確認資料庫初始化與排程器狀態

## 相關資源

- [LINE 開發者平台](https://developers.line.biz/zh-hant/)
- [OpenAI API 文件](https://platform.openai.com/docs/)
- [Flask 文件](https://flask.palletsprojects.com/)

更多詳細資訊，請參閱專案中的 [Documentary.md](Documentary.md)
