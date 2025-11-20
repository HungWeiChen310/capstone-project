# src/routes/admin.py
from flask import request, session, redirect, url_for, flash, render_template, abort
from . import admin_bp
from ..services.auth_service import admin_required
from ..database import db
import os
import datetime
import logging

logger = logging.getLogger(__name__)

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

@admin_bp.route("/login", methods=["GET", "POST"])
def admin_login():
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        logger.error("Admin login attempted, but ADMIN_USERNAME or ADMIN_PASSWORD is not set.")
        return "Admin login is disabled because credentials are not configured.", 503

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
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
