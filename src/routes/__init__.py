# src/routes/__init__.py

from flask import Blueprint

# Create Blueprints
main_bp = Blueprint('main', __name__)
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')
alarm_bp = Blueprint('alarm', __name__)

# Import routes to register them with the blueprints
from . import main, admin, alarm  # noqa: E402,F401


def register_blueprints(app):
    """Registers all blueprints with the Flask app."""
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(alarm_bp)
