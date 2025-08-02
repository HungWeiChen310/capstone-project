# client.py
import requests

def send_json():
    url = 'https://127.0.0.1:443/alarms'
    payload = {
        "equipment_id": "EQ003",
        "alert_type": "轉速過低",
        "severity": "low"
    }

    try:
        resp = requests.post(url, json=payload, timeout=5, verify=False)
        resp.raise_for_status()
    except requests.RequestException as e:
        print("Request failed:", e)
        return

if __name__ == '__main__':
    send_json()
