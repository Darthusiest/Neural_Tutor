from app.routes.admin import bp as admin_bp
from app.routes.auth import bp as auth_bp
from app.routes.chat import bp as chat_bp
from app.routes.health import bp as health_bp
from app.routes.lectures import bp as lectures_bp

__all__ = ["admin_bp", "auth_bp", "chat_bp", "health_bp", "lectures_bp"]
