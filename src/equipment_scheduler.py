# src/equipment_scheduler.py
import logging
import signal
import threading
import time
import schedule
from equipment_monitor import EquipmentMonitor
logger = logging.getLogger(__name__)
scheduler_running = False
scheduler_thread = None
monitor = EquipmentMonitor()


def run_scheduler():
    """執行排程任務"""
    global scheduler_running
    # 每隔 5 分鐘檢查一次設備狀態
    schedule.every(5).minutes.do(monitor.check_all_equipment)
    logger.info("設備監控排程器已啟動")
    scheduler_running = True
    while scheduler_running:
        schedule.run_pending()
        time.sleep(1)
    logger.info("設備監控排程器已停止")


def start_scheduler():
    """以背景執行緒啟動排程器"""
    global scheduler_thread, scheduler_running
    # 確保不會重複啟動
    if scheduler_thread and scheduler_thread.is_alive():
        logger.info("設備監控排程器已在運行中")
        return
    scheduler_running = True
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logger.info("設備監控背景執行緒已啟動")
    # 註冊信號處理函數，用於優雅關閉
    signal.signal(signal.SIGTERM, stop_scheduler_on_signal)
    signal.signal(signal.SIGINT, stop_scheduler_on_signal)


def stop_scheduler():
    global scheduler_running, scheduler_thread
    if not scheduler_thread or not scheduler_thread.is_alive():
        logger.info("設備監控排程器未在運行")
        return
    logger.info("正在停止設備監控排程器...")
    scheduler_running = False
    scheduler_thread.join(timeout=5)
    if scheduler_thread.is_alive():
        logger.warning("設備監控排程器無法正常停止")
    else:
        logger.info("設備監控排程器已成功停止")
        scheduler_thread = None


def stop_scheduler_on_signal(signum, frame):
    logger.info("信號處理函數，用於處理 SIGTERM 和 SIGINT")
    stop_scheduler()


def manual_check():
    monitor.check_all_equipment()
