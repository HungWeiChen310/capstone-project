import logging
import os
import sys
from dotenv import load_dotenv
# Load environment variables from .env file if it exists
load_dotenv()
# Determine APP_ENV first
APP_ENV = os.getenv("APP_ENV", "development").lower()
if os.getenv("TESTING", "False").lower() == "true": # Legacy TESTING var overrides APP_ENV
    APP_ENV = "testing"

# Configure logging - This will be configured based on APP_ENV
log_level_map = {
    "development": logging.DEBUG,
    "testing": logging.DEBUG,
    "production": logging.INFO,
}
current_log_level = log_level_map.get(APP_ENV, logging.INFO)

logging.basicConfig(
    level=current_log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(pathname)s:%(lineno)d]",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode='a'), # Append mode
    ],
)
logger = logging.getLogger(__name__)
logger.info(f"應用程式環境 (APP_ENV) 設置為: {APP_ENV}")
logger.info(f"日誌級別設置為: {logging.getLevelName(current_log_level)}")


class Config:
    """應用程式配置，集中管理所有環境變數"""
    APP_ENV = APP_ENV # Store determined APP_ENV in Config class

    # Default values, potentially overridden by environment-specific settings
    DEBUG = False
    PORT = 5000
    VALIDATION_MODE = "strict" # Default to strict
    LOG_LEVEL = current_log_level # Use already determined log level

    if APP_ENV == "production":
        DEBUG = False
        PORT = int(os.getenv("PORT", 443)) # Production often uses 443 for HTTPS
        VALIDATION_MODE = "strict"
        LOG_LEVEL = logging.INFO
    elif APP_ENV == "development":
        DEBUG = True
        PORT = int(os.getenv("PORT", 5000))
        VALIDATION_MODE = "loose" # Loose for development
        LOG_LEVEL = logging.DEBUG
    elif APP_ENV == "testing":
        DEBUG = True # Usually True for testing to get more error info
        PORT = int(os.getenv("PORT", 5001)) # Different port for tests
        VALIDATION_MODE = "loose" # Loose for testing, test setup should ensure necessary vars
        LOG_LEVEL = logging.DEBUG

    # Override with FLASK_DEBUG if explicitly set
    DEBUG = os.getenv("FLASK_DEBUG", str(DEBUG)).lower() == "true"

    logger.info(f"配置模式: {APP_ENV} | DEBUG: {DEBUG} | PORT: {PORT} | VALIDATION_MODE: {VALIDATION_MODE}")

    # Critical Application Variables
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
    SECRET_KEY = os.getenv("SECRET_KEY") # For Flask app.secret_key

    # Database 配置
    DB_SERVER = os.getenv("DB_SERVER", "localhost")
    DB_NAME = os.getenv("DB_NAME", "conversations")
    DB_USER = os.getenv("DB_USER") # Currently not used with Trusted_Connection=yes
    DB_PASSWORD = os.getenv("DB_PASSWORD") # Currently not used

    @classmethod
    def validate(cls, exit_on_failure=True): # Default to True for direct calls
        """
        驗證必需的環境變數是否存在。
        參數:
            exit_on_failure:
                - True: 驗證失敗時記錄嚴重錯誤並終止程序。
                - False: 驗證失敗時記錄錯誤並拋出 ValueError。
        """
        missing_vars = []
        critical_vars = {
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
            "LINE_CHANNEL_ACCESS_TOKEN": cls.LINE_CHANNEL_ACCESS_TOKEN,
            "LINE_CHANNEL_SECRET": cls.LINE_CHANNEL_SECRET,
            "ADMIN_USERNAME": cls.ADMIN_USERNAME,
            "ADMIN_PASSWORD": cls.ADMIN_PASSWORD,
            "SECRET_KEY": cls.SECRET_KEY, # SECRET_KEY is crucial for sessions
            "DB_SERVER": cls.DB_SERVER,
            "DB_NAME": cls.DB_NAME,
        }

        # Add DB_USER and DB_PASSWORD to critical_vars if not using trusted connection
        # For now, they are optional as Trusted_Connection=yes is used.
        # if not trusted_connection_assumed:
        #     if not cls.DB_USER: missing_vars.append("DB_USER")
        #     if not cls.DB_PASSWORD: missing_vars.append("DB_PASSWORD")

        for var_name, value in critical_vars.items():
            if not value:
                missing_vars.append(var_name)

        if missing_vars:
            error_msg = f"關鍵環境變數缺失: {', '.join(missing_vars)}。"
            if exit_on_failure:
                logger.critical(error_msg + " 程序將終止。")
                sys.exit(1)
            else:
                logger.error(error_msg + " 請檢查您的 .env 文件或環境設定。")
                raise ValueError(error_msg)

        logger.info(f"環境變數驗證成功 ({len(critical_vars)} 個關鍵變數已檢查)。")
        return True

# --- Initial Configuration Validation at Load Time ---
# This validation's behavior depends on the APP_ENV and its inherent VALIDATION_MODE.
logger.info(f"執行啟動時配置驗證 (VALIDATION_MODE: {Config.VALIDATION_MODE})...")

try:
    # For production, validate() will exit on failure due to VALIDATION_MODE='strict' default in constructor.
    # For dev/test, validate() will raise ValueError if exit_on_failure=False (derived from loose VALIDATION_MODE).
    # The exit_on_failure for this initial call is now effectively controlled by Config.VALIDATION_MODE's strictness.

    # If VALIDATION_MODE is 'strict', we want to exit.
    # If 'loose', we want to raise error and log, but not exit here.
    perform_strict_exit = Config.VALIDATION_MODE.lower() == 'strict'
    Config.validate(exit_on_failure=perform_strict_exit)

    logger.info(f"配置成功加載並驗證完畢。運行環境: {Config.APP_ENV.upper()}")

except ValueError as e:
    # This block will only be reached if VALIDATION_MODE was 'loose' and validation failed.
    logger.warning(
        f"環境: {Config.APP_ENV.upper()} - 關鍵配置變數驗證失敗。應用程式可能無法正常運行: {e}"
    )
    # No sys.exit here for loose validation; app might continue if some features don't depend on missing vars.
