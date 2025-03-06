import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)

# 從環境變數讀取 PowerBI API 所需參數
POWERBI_CLIENT_ID = os.getenv("POWERBI_CLIENT_ID")
POWERBI_CLIENT_SECRET = os.getenv("POWERBI_CLIENT_SECRET")
POWERBI_TENANT_ID = os.getenv("POWERBI_TENANT_ID")
POWERBI_WORKSPACE_ID = os.getenv("POWERBI_WORKSPACE_ID")
POWERBI_REPORT_ID = os.getenv("POWERBI_REPORT_ID")

if not all([POWERBI_CLIENT_ID, POWERBI_CLIENT_SECRET, POWERBI_TENANT_ID, POWERBI_WORKSPACE_ID, POWERBI_REPORT_ID]):
    logger.warning("PowerBI 環境變數未完整設定，部分功能可能無法使用。")

# Token 快取與鎖定
_access_token = None
_access_token_expiry = 0
_token_lock = Lock()

def get_powerbi_access_token() -> str:
    """
    透過 OAuth2 客戶端憑證流程取得 PowerBI API 存取權杖
    使用記憶體快取，避免頻繁請求
    """
    global _access_token, _access_token_expiry
    
    # 使用鎖避免並發問題
    with _token_lock:
        current_time = time.time()
        
        # 檢查是否有有效的 token 快取
        if _access_token and _access_token_expiry > current_time + 30:  # 30 秒緩衝
            return _access_token
        
        url = f"https://login.microsoftonline.com/{POWERBI_TENANT_ID}/oauth2/v2.0/token"
        payload = {
            'grant_type': 'client_credentials',
            'client_id': POWERBI_CLIENT_ID,
            'client_secret': POWERBI_CLIENT_SECRET,
            'scope': 'https://analysis.windows.net/powerbi/api/.default'
        }
        
        logger.info("正在取得新的 PowerBI access token")
        response = requests.post(url, data=payload)
        
        if response.status_code != 200:
            logger.error(f"取得 access token 失敗: Status code {response.status_code}, Response: {response.text}")
            raise Exception("無法取得 PowerBI 存取權杖，請檢查憑證設定。")
            
        json_resp = response.json()
        _access_token = json_resp.get("access_token")
        expires_in = json_resp.get("expires_in", 3600)  # 默認為 1 小時
        
        if not _access_token:
            logger.error(f"回應中未包含 access_token: {json_resp}")
            raise Exception("PowerBI 回應中未包含 access_token")
            
        # 設定過期時間 (提前 5 分鐘過期，以確保安全)
        _access_token_expiry = current_time + expires_in - 300
        
        logger.info(f"PowerBI access token 成功取得，有效期至 {datetime.fromtimestamp(_access_token_expiry).isoformat()}")
        return _access_token

# Embed Token 快取與過期時間
_embed_token = None
_embed_token_expiry = 0
_embed_lock = Lock()

def get_powerbi_embed_token(access_token: str = None) -> str:
    """
    呼叫 PowerBI API 產生報表的 Embed Token
    使用記憶體快取，避免頻繁請求
    """
    global _embed_token, _embed_token_expiry
    
    # 使用鎖避免並發問題
    with _embed_lock:
        current_time = time.time()
        
        # 檢查是否有有效的 token 快取
        if _embed_token and _embed_token_expiry > current_time + 30:  # 30 秒緩衝
            return _embed_token
        
        # 若沒有提供 access_token，則獲取一個新的
        if not access_token:
            access_token = get_powerbi_access_token()
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{POWERBI_WORKSPACE_ID}/reports/{POWERBI_REPORT_ID}/GenerateToken"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        payload = {"accessLevel": "view"}
        
        logger.info("正在取得新的 PowerBI embed token")
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"取得 embed token 失敗：{response.text}")
            raise Exception("無法取得 PowerBI Embed Token")
            
        json_resp = response.json()
        _embed_token = json_resp.get("token")
        
        if not _embed_token:
            logger.error(f"回應中未包含 embed token：{json_resp}")
            raise Exception("PowerBI 回應中未包含 embed token")
            
        # 取得過期時間，默認為 1 小時
        expiry = json_resp.get("expiration")
        if expiry:
            try:
                # API 返回的是 ISO 格式日期字串
                expiry_dt = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
                _embed_token_expiry = expiry_dt.timestamp()
            except Exception as e:
                logger.warning(f"解析 token 過期時間失敗: {e}，使用預設值")
                _embed_token_expiry = current_time + 3600 - 300  # 默認 1 小時減 5 分鐘
        else:
            _embed_token_expiry = current_time + 3600 - 300  # 默認 1 小時減 5 分鐘
        
        logger.info(f"PowerBI embed token 成功取得，有效期至 {datetime.fromtimestamp(_embed_token_expiry).isoformat()}")
        return _embed_token

def get_powerbi_embed_config() -> dict:
    """
    組合 PowerBI 嵌入所需的設定，包含 embed URL 與 token
    """
    access_token = get_powerbi_access_token()
    embed_token = get_powerbi_embed_token(access_token)
    embed_url = f"https://app.powerbi.com/reportEmbed?reportId={POWERBI_REPORT_ID}&groupId={POWERBI_WORKSPACE_ID}"
    
    return {
        "embedUrl": embed_url,
        "accessToken": embed_token,
        "reportId": POWERBI_REPORT_ID,
        "workspaceId": POWERBI_WORKSPACE_ID,
        "settings": {
            "filterPaneEnabled": True,
            "navContentPaneEnabled": True
        }
    }

def get_report_pages() -> list:
    """
    取得報表中所有頁面的清單
    """
    try:
        access_token = get_powerbi_access_token()
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{POWERBI_WORKSPACE_ID}/reports/{POWERBI_REPORT_ID}/pages"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"取得報表頁面清單失敗：{response.text}")
            return []
            
        return response.json().get("value", [])
    except Exception as e:
        logger.error(f"取得報表頁面時發生錯誤：{e}")
        return []

def get_powerbi_dashboards() -> list:
    """
    取得工作區中的儀表板清單
    """
    try:
        access_token = get_powerbi_access_token()
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{POWERBI_WORKSPACE_ID}/dashboards"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"取得儀表板清單失敗：{response.text}")
            return []
            
        return response.json().get("value", [])
    except Exception as e:
        logger.error(f"取得儀表板清單時發生錯誤：{e}")
        return []

def get_powerbi_datasets() -> list:
    """
    取得工作區中的資料集清單
    """
    try:
        access_token = get_powerbi_access_token()
        
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{POWERBI_WORKSPACE_ID}/datasets"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"取得資料集清單失敗：{response.text}")
            return []
            
        return response.json().get("value", [])
    except Exception as e:
        logger.error(f"取得資料集清單時發生錯誤：{e}")
        return []

def get_dashboard_embed_config(dashboard_id) -> dict:
    """
    取得儀表板的嵌入設定
    """
    try:
        access_token = get_powerbi_access_token()
        
        # 取得儀表板的 Embed Token
        url = f"https://api.powerbi.com/v1.0/myorg/groups/{POWERBI_WORKSPACE_ID}/dashboards/{dashboard_id}/GenerateToken"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        payload = {"accessLevel": "view"}
        
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            logger.error(f"取得儀表板 embed token 失敗：{response.text}")
            raise Exception("無法取得儀表板 Embed Token")
            
        embed_token = response.json().get("token")
        
        # 組合嵌入設定
        embed_url = f"https://app.powerbi.com/dashboardEmbed?dashboardId={dashboard_id}&groupId={POWERBI_WORKSPACE_ID}"
        
        return {
            "embedUrl": embed_url,
            "accessToken": embed_token,
            "dashboardId": dashboard_id,
            "workspaceId": POWERBI_WORKSPACE_ID
        }
    except Exception as e:
        logger.error(f"取得儀表板嵌入設定時發生錯誤：{e}")
        raise

def export_report_to_pdf() -> bytes:
    """
    將報表匯出為 PDF
    """
    try:
        access_token = get_powerbi_access_token()
        
        # 啟動匯出作業
        export_url = f"https://api.powerbi.com/v1.0/myorg/groups/{POWERBI_WORKSPACE_ID}/reports/{POWERBI_REPORT_ID}/ExportTo"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        payload = {"format": "PDF"}
        
        response = requests.post(export_url, json=payload, headers=headers)
        
        if response.status_code != 202:  # 202 Accepted
            logger.error(f"啟動報表匯出失敗：{response.text}")
            raise Exception("無法啟動報表匯出")
            
        # 取得匯出作業 ID
        export_id = response.headers.get("Location").split('/')[-1]
        
        # 檢查匯出作業狀態
        status_url = f"https://api.powerbi.com/v1.0/myorg/groups/{POWERBI_WORKSPACE_ID}/reports/{POWERBI_REPORT_ID}/exports/{export_id}"
        
        max_attempts = 10
        attempt = 0
        
        while attempt < max_attempts:
            response = requests.get(status_url, headers={"Authorization": f"Bearer {access_token}"})
            
            if response.status_code != 200:
                logger.error(f"檢查匯出狀態失敗：{response.text}")
                raise Exception("無法檢查匯出狀態")
                
            status = response.json().get("status")
            
            if status == "Succeeded":
                # 取得匯出檔案
                file_url = f"{status_url}/file"
                response = requests.get(file_url, headers={"Authorization": f"Bearer {access_token}"})
                
                if response.status_code != 200:
                    logger.error(f"下載匯出檔案失敗：{response.text}")
                    raise Exception("無法下載匯出檔案")
                    
                return response.content
            elif status == "Failed":
                logger.error("報表匯出失敗")
                raise Exception("報表匯出失敗")
            
            # 等待一段時間後再檢查
            time.sleep(2)
            attempt += 1
        
        raise Exception("報表匯出超時")
    except Exception as e:
        logger.error(f"匯出報表時發生錯誤：{e}")
        raise