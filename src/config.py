import logging
import os
import sys
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger(__name__)


class Config:
    """應用程式配置，集中管理所有環境變數"""
    # 一般配置
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    PORT = int(os.getenv("PORT", 5000))
    # OpenAI 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # LINE Bot 配置
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
    # Database 配置
    DB_SERVER = os.getenv("DB_SERVER", "localhost")  # Default
    DB_NAME = os.getenv("DB_NAME", "conversations")  # Default
    DB_USER = os.getenv("DB_USER")  # For potential future use with non-trusted connections
    DB_PASSWORD = os.getenv("DB_PASSWORD")  # For potential future use
    # 驗證模式：嚴格 (strict) 或寬鬆 (loose)
    VALIDATION_MODE = os.getenv("VALIDATION_MODE", "strict")

    @classmethod
    def validate(cls, exit_on_failure=None):
        """
        驗證必需的環境變數是否存在
        參數:
            exit_on_failure:
                - 如果為 True，驗證失敗時會終止程序
                - 如果為 False，驗證失敗時只會拋出例外
                - 如果為 None，則根據 VALIDATION_MODE 環境變數決定行為
        """
        missing_vars = []
        # 檢查 OpenAI 設定
        if not cls.OPENAI_API_KEY:
            missing_vars.append("OPENAI_API_KEY")
        # 檢查 LINE Bot 設定
        if not cls.LINE_CHANNEL_ACCESS_TOKEN:
            missing_vars.append("LINE_CHANNEL_ACCESS_TOKEN")
        if not cls.LINE_CHANNEL_SECRET:
            missing_vars.append("LINE_CHANNEL_SECRET")
        # 檢查 Database 設定
        if not cls.DB_SERVER:
            missing_vars.append("DB_SERVER")
        if not cls.DB_NAME:
            missing_vars.append("DB_NAME")
        if missing_vars:
            error_msg = f"缺少以下必要環境變數: {', '.join(missing_vars)}"
            logger.error(error_msg)
            # 決定驗證失敗行為
            should_exit = (
                exit_on_failure
                if exit_on_failure is not None
                else cls.VALIDATION_MODE.lower() == "strict"
            )
            # 嚴格模式或明確要求下直接中斷程序
            if should_exit:
                logger.critical("驗證失敗，程序將終止")
                sys.exit(1)
            # 否則拋出例外，讓呼叫者決定如何處理
            raise ValueError(error_msg)
        return True


# 應用程序啟動時嘗試驗證配置
is_testing = os.environ.get("TESTING", "False").lower() == "true"
if not is_testing:
    try:
        # 這裡會根據 VALIDATION_MODE 環境變數決定驗證失敗行為
        Config.validate()
        logger.info("環境變數驗證成功")
    except ValueError as e:
        # 非測試環境且配置指定為寬鬆模式時，發出警告但不中斷
        logger.error(f"環境變數驗證失敗: {e}")
        # 注意：在寬鬆模式下，這裡沒有中斷程序，讓主程式決定如何處理
else:
    logger.debug("TESTING 模式啟用，略過環境變數驗證")
