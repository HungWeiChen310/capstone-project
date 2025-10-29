# client.py
import requests

def send_json():
    url = 'https://127.0.0.1:443/alarms'
    payload = {
        "equipment_id": "EQ003",
        "alert_type": "轉速過低",
        "severity_level": "",
    }
    
    # 根據 alert_type 決定附加的欄位
    if payload["alert_type"] == "轉速過低":
        """
        情境1:轉速太低
        正常值:30000 
        嚴重程度:
        warning:範圍 24000~27000
        critical:範圍 18000~24000
        emergency:範圍 <18000
        """
        payload["rpm"] = 1500  # 填入實際轉速值
        payload["deformation_mm"] = 0  # 預設固定為0

    elif payload["alert_type"] == "刀具裂痕" or payload["alert_type"] == "刀具變形":
        """
        情境2:刀具裂痕 or 刀具變形 (單位皆為mm)
        正常值:0
        嚴重程度:
        warning:範圍 0.01~0.05
        critical:範圍 0.05~.01
        emergency:範圍 >.1
        """
        payload["rpm"] = 30000  # RPM 固定為 30000
        payload["deformation_mm"] = 0.8  # 填入實際變形值

    try:
        resp = requests.post(url, json=payload, timeout=5, verify=False)
        resp.raise_for_status()
    except requests.RequestException as e:
        print("Request failed:", e)
        return

if __name__ == '__main__':
    send_json()
