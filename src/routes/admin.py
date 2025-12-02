# src/routes/admin.py
from flask import request, session, redirect, url_for, flash, render_template, abort
from werkzeug.security import check_password_hash
from src.config import Config
from . import admin_bp
from ..services.auth_service import admin_required
from ..database import db
import datetime
import logging

logger = logging.getLogger(__name__)

@admin_bp.route("/login", methods=["GET", "POST"])
def admin_login():
    # Support both Hash (recommended) and Plaintext (legacy transition)
    admin_user = Config.ADMIN_USERNAME
    admin_pwd_hash = Config.ADMIN_PASSWORD_HASH

    # Fallback: Read legacy plaintext password directly from env if hash is missing
    # This ensures backward compatibility for deployments that haven't migrated yet.
    import os
    admin_pwd_legacy = os.getenv("ADMIN_PASSWORD")

    if not admin_user or (not admin_pwd_hash and not admin_pwd_legacy):
        logger.error("Admin login attempted, but ADMIN credentials are not properly configured.")
        return "Admin login is disabled because credentials are not configured.", 503

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        is_authenticated = False
        if username == admin_user:
            # Prioritize hash check if available
            if admin_pwd_hash and check_password_hash(admin_pwd_hash, password):
                is_authenticated = True
            # Fallback to legacy check if hash check failed or hash not set
            elif admin_pwd_legacy and password == admin_pwd_legacy:
                is_authenticated = True
                logger.warning("Admin logged in using legacy plaintext password. Please migrate to ADMIN_PASSWORD_HASH.")

        if is_authenticated:
            session["admin_logged_in"] = True
            session.permanent = True
            # Access app's config through the blueprint
            admin_bp.permanent_session_lifetime = datetime.timedelta(days=7)
            return redirect(request.args.get("next") or url_for("admin.admin_dashboard"))
        else:
            flash("登入失敗，請確認帳號密碼是否正確", "error")
    return render_template("admin_login.html")


@admin_bp.route("/logout")
def admin_logout():
    session.pop("admin_logged_in", None)
    return redirect(url_for("admin.admin_login"))


@admin_bp.route("/dashboard")
@admin_required
def admin_dashboard():
    conversation_stats = db.get_conversation_stats()
    recent_conversations = db.get_recent_conversations(limit=20)
    system_info = {
        "ollama_endpoint": f"{os.getenv('OLLAMA_HOST', '127.0.0.1')}:{os.getenv('OLLAMA_PORT', '11434')}",
        "ollama_model": os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
        "line_channel_secret": "已設置" if os.getenv("LINE_CHANNEL_SECRET") else "未設置",
        "db_server": os.getenv("DB_SERVER", "localhost"),
        "db_name": os.getenv("DB_NAME", "conversations")
    }
    return render_template(
        "admin_dashboard.html",
        stats=conversation_stats,
        recent=recent_conversations,
        system_info=system_info,
    )


@admin_bp.route("/conversation/<user_id>")
@admin_required
def admin_view_conversation(user_id):
    conversation = db.get_conversation_history(user_id, limit=50)
    user_info = db.get_user_preference(user_id)
    return render_template(
        "admin_conversation.html",
        conversation=conversation,
        user_id=user_id,
        user_info=user_info,
    )
