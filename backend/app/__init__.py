import os

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS

from app.config import Config
from app.extensions import db, login_manager

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = None

    @login_manager.unauthorized_handler
    def _unauthorized():
        return jsonify({"error": "unauthorized"}), 401

    CORS(
        app,
        resources={r"/api/*": {"origins": [app.config["FRONTEND_ORIGIN"]]}},
        supports_credentials=True,
    )

    from app.routes import admin_bp, auth_bp, chat_bp, health_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(admin_bp, url_prefix="/api/admin")

    @app.cli.command("init-db")
    def init_db():
        """Create SQLite tables: `cd backend && flask --app wsgi init-db`"""
        db.create_all()
        print("Database initialized.")

    return app
