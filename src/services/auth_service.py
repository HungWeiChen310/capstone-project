# src/services/auth_service.py
import functools
from flask import session, redirect, url_for, request

def admin_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin.admin_login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function
