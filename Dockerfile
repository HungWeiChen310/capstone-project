# 使用官方 Python 3.11 作為基礎映像與 CI/CD 一致
FROM python:3.11-slim

# 建立非 root 用戶
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 設定工作目錄
WORKDIR /app

# 複製專案檔案到容器
COPY . .

# 安裝專案所需的 Python 套件
RUN pip install --no-cache-dir -r requirements.txt && \
    # 移除不必要的套件以減少攻擊面
    apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 設定環境變數 (不包含敏感資訊)
ENV PYTHONPATH=/app

# 確保檔案權限正確
RUN chown -R appuser:appuser /app

# 切換到非 root 用戶
USER appuser

# 指定容器啟動時執行的指令
CMD ["python", "-m", "src.linebot_connect"]