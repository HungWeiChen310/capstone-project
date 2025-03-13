# src/app.py (新增檔案)
import logging
import os
import sys
from flask import Flask
from flask_talisman import Talisman
from werkzeug.middleware.proxy_fix import ProxyFix

# 導入自訂模組
from src.config import Config
from src.database import db
from src.equipment_scheduler import start_scheduler
from src.initial_data import initialize_equipment_data
from src.event_system import event_system

# 設定日誌
logger = logging.getLogger(__name__)

def create_app(testing=False):
    """創建並配置 Flask 應用程序"""
    try:
        # 驗證配置
        try:
            Config.validate()
            logger.info("環境變數驗證成功")
        except ValueError as e:
            if not testing:  # 測試模式下允許使用假數據
                logger.critical(f"環境變數驗證失敗: {e}")
                sys.exit(1)
            else:
                logger.warning(f"測試模式: 環境變數驗證失敗，但將繼續執行: {e}")
        
        # 初始化 Flask 應用
        app = Flask(__name__, 
                    template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'),
                    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'))
        
        # 設定密鑰
        from src.linebot_connect import get_or_create_secret_key
        app.secret_key = get_or_create_secret_key()
        
        # 處理代理頭信息
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)
        
        # 安全性設置
        if not testing:
            csp = {
                'default-src': "'self'",
                'script-src': [
                    "'self'",
                    'https://cdn.powerbi.com',
                    "'unsafe-inline'",
                ],
                'style-src': [
                    "'self'",
                    "'unsafe-inline'",
                ],
                'img-src': "'self'",
                'frame-src': [
                    'https://app.powerbi.com',
                    'https://cdn.powerbi.com',
                ],
                'connect-src': [
                    "'self'",
                    'https://api.powerbi.com',
                    'https://login.microsoftonline.com',
                ]
            }
            
            Talisman(app, 
                content_security_policy=csp,
                content_security_policy_nonce_in=['script-src'],
                force_https=True,
                session_cookie_secure=True,
                session_cookie_http_only=True,
                feature_policy="geolocation 'none'; microphone 'none'; camera 'none'"
            )
        
        # 初始化數據
        initialize_equipment_data()
        
        # 初始化設備監控
        start_scheduler()
        
        # 注冊關閉處理函數
        @app.teardown_appcontext
        def shutdown_app(exception=None):
            from src.equipment_scheduler import stop_scheduler
            stop_scheduler()
            
        # 注冊路由和處理函數
        from src.linebot_connect import register_routes
        register_routes(app)
        
        return app
    except Exception as e:
        logger.critical(f"應用程序初始化失敗: {e}")
        raise

# 提供一個便利函數來運行應用
def run_app(host="0.0.0.0", port=None, debug=None, ssl_context=None):
    """運行 Flask 應用程序"""
    port = port or int(os.environ.get("PORT", 5000))
    debug = debug or (os.environ.get("FLASK_DEBUG", "False").lower() == "true")
    
    app = create_app()
    app.run(host=host, port=port, debug=debug, ssl_context=ssl_context)

if __name__ == "__main__":
# Use Flask's built-in adhoc SSL certificates (requires pyOpenSSL)
    # ssl_context = 'adhoc'
    ssl_cert = os.environ.get("SSL_CERT_PATH")
    ssl_key = os.environ.get("SSL_KEY_PATH")
    if ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
    else:
        print("SSL_CERT_PATH and SSL_KEY_PATH not set. Running without SSL.")
        ssl_context = None
    
    print(f"Starting Flask app with SSL...")
    run_app(ssl_context=ssl_context)