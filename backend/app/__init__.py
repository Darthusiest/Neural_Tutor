from pathlib import Path

import click
from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
from flask_wtf.csrf import CSRFError

from app.config import Config
from app.extensions import csrf, db, limiter, login_manager

load_dotenv()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = None

    csrf.init_app(app)
    limiter.init_app(app)

    @login_manager.unauthorized_handler
    def _unauthorized():
        return jsonify({"error": "unauthorized"}), 401

    @app.errorhandler(CSRFError)
    def _csrf_error(_e: CSRFError):
        return jsonify({"error": "csrf validation failed"}), 403

    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [app.config["FRONTEND_ORIGIN"]],
                "allow_headers": ["Content-Type", "X-CSRFToken"],
                "expose_headers": [],
            }
        },
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

    @app.cli.command("import-lectures")
    @click.argument(
        "json_path",
        type=click.Path(path_type=Path, exists=True),
        required=False,
    )
    def import_lectures(json_path):
        """Load lecture JSON into `lecture_chunks` (default: LECTURE_JSON_PATH / data file)."""
        from app.services.lecture_loader import import_lecture_json

        path = json_path or app.config["LECTURE_JSON_PATH"]
        if not path.exists():
            raise click.ClickException(f"Lecture file not found: {path}")
        with app.app_context():
            n = import_lecture_json(path)
        print(f"Imported {n} lecture sections into lecture_chunks.")

    return app
