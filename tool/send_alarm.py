# client.py
import requests


def send_json() -> None:
    url = "https://127.0.0.1:443/alarms"

    payload = {
        "equipment_id": "EQ001",
        "detected_anomaly_type": "轉速過低",
    }

    if payload["detected_anomaly_type"] == "轉速過低":
        rpm = 15000
        deformation_mm = 0  # 固定值
        payload["rpm"] = rpm
        payload["severity_level"] = severity_for_rpm_low(rpm)

    elif payload["detected_anomaly_type"] == "刀具裂痕":
        rpm = 30000  # 固定值
        deformation_mm = 0.8
        payload["deformation_mm"] = deformation_mm
        payload["severity_level"] = severity_for_deformation(deformation_mm)

    elif payload["detected_anomaly_type"] == "刀具變形":
        rpm = 30000  # 固定值
        deformation_mm = 0.8
        payload["deformation_mm"] = deformation_mm
        payload["severity_level"] = severity_for_deformation(deformation_mm)

    else:
        raise ValueError(f"Unsupported detected_anomaly_type: {payload['detected_anomaly_type']}")

    print("即將送出:", payload)

    try:
        resp = requests.post(url, json=payload, timeout=5, verify=False)
        resp.raise_for_status()
        print("Response:", resp.status_code, resp.text)
    except requests.RequestException as e:
        print("Request failed:", e)

def severity_for_rpm_low(rpm: float) -> str:
    if rpm < 18000:
        return "emergency"
    if rpm < 24000:
        return "critical"
    if rpm < 27000:
        return "warning"
    return "warning"

def severity_for_deformation(deformation_mm: float) -> str:
    if deformation_mm > 0.1:
        return "emergency"
    if deformation_mm > 0.05:
        return "critical"
    if deformation_mm >= 0.01:
        return "warning"
    return "warning"

if __name__ == "__main__":
    send_json()
