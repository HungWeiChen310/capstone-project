import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log'),
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """應用程式配置，集中管理所有環境變數"""
    
    # 一般配置
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    
    # OpenAI 配置
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # LINE Bot 配置
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
    LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
    
    # PowerBI 配置
    POWERBI_CLIENT_ID = os.getenv('POWERBI_CLIENT_ID')
    POWERBI_CLIENT_SECRET = os.getenv('POWERBI_CLIENT_SECRET')
    POWERBI_TENANT_ID = os.getenv('POWERBI_TENANT_ID')
    POWERBI_WORKSPACE_ID = os.getenv('POWERBI_WORKSPACE_ID')
    POWERBI_REPORT_ID = os.getenv('POWERBI_REPORT_ID')
    
    @classmethod
    def validate(cls):
        """驗證必需的環境變數是否存在"""
        missing_vars = []
        
        # 檢查 OpenAI 設定
        if not cls.OPENAI_API_KEY:
            missing_vars.append('OPENAI_API_KEY')
        
        # 檢查 LINE Bot 設定
        if not cls.LINE_CHANNEL_ACCESS_TOKEN:
            missing_vars.append('LINE_CHANNEL_ACCESS_TOKEN')
        if not cls.LINE_CHANNEL_SECRET:
            missing_vars.append('LINE_CHANNEL_SECRET')
        
        # 檢查 PowerBI 設定
        if not all([cls.POWERBI_CLIENT_ID, cls.POWERBI_CLIENT_SECRET, 
                  cls.POWERBI_TENANT_ID, cls.POWERBI_WORKSPACE_ID, 
                  cls.POWERBI_REPORT_ID]):
            missing_vars.append('POWERBI_* (one or more PowerBI variables)')
        
        if missing_vars:
            error_msg = f"缺少以下必要環境變數: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return True

# Validate config when module is imported
try:
    Config.validate()
    logger.info("環境變數驗證成功")
except ValueError as e:
    logger.error(f"環境變數驗證失敗: {e}")
    # 不在此處中斷程式，而是讓主程式決定如何處理