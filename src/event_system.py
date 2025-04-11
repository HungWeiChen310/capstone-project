# src/event_system.py (新增檔案)
import logging
from typing import Any, Callable, Dict, List
logger = logging.getLogger(__name__)


class EventSystem:
    """簡單的事件系統，用於解耦模組間的依賴"""

    def __init__(self):
        self.handlers: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable):
        """訂閱事件"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        """取消訂閱事件"""
        if event_type in self.handlers:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)

    def publish(self, event_type: str, **kwargs) -> List[Any]:
        """發布事件並返回所有處理結果"""
        if event_type not in self.handlers:
            return []
        results = []
        for handler in self.handlers[event_type]:
            try:
                result = handler(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"事件處理失敗 '{event_type}': {e}")
        return results


# 全局事件系統實例
event_system = EventSystem()
