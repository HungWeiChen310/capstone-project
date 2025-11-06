# src/utils.py
import os
import secrets
import logging
import time
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)

SECRET_KEY_FILE = os.getenv("SECRET_KEY_FILE", "data/secret_key.txt")

def get_or_create_secret_key():
    """獲取或創建一個固定的 secret key"""
    env_key = os.getenv("SECRET_KEY")
    if env_key:
        return env_key

    os.makedirs(os.path.dirname(SECRET_KEY_FILE), exist_ok=True)
    try:
        if os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, "r") as f:
                key = f.read().strip()
                if key:
                    return key
        key = secrets.token_hex(24)
        with open(SECRET_KEY_FILE, "w") as f:
            f.write(key)
        return key
    except Exception as e:
        logger.warning(f"無法讀取或寫入密鑰文件: {e}，使用臨時密鑰")
        return secrets.token_hex(24)

# Global request counter and lock (thread-safe)
request_counts = defaultdict(list)
last_cleanup_time = time.time()
request_counts_lock = threading.Lock()


def cleanup_request_counts():
    """清理長時間未使用的 IP 地址"""
    global last_cleanup_time
    current_time = time.time()

    if current_time - last_cleanup_time < 3600:
        return

    with request_counts_lock:
        ips_to_remove = [
            ip for ip, timestamps in request_counts.items()
            if not timestamps or current_time - max(timestamps) > 3600
        ]
        for ip in ips_to_remove:
            del request_counts[ip]
        last_cleanup_time = current_time
        logger.info("已清理過期請求記錄")


def rate_limit_check(ip, max_requests=30, window_seconds=60):
    """
    簡單的 IP 請求限制，防止暴力攻擊
    """
    current_time = time.time()
    cleanup_request_counts()
    with request_counts_lock:
        request_counts[ip] = [
            timestamp for timestamp in request_counts[ip]
            if current_time - timestamp < window_seconds
        ]
        if len(request_counts[ip]) >= max_requests:
            return False
        request_counts[ip].append(current_time)
        return True
