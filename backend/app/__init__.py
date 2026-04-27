from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
from flask_wtf.csrf import CSRFError

from app.config import Config
from app.extensions import csrf, db, limiter, login_manager, migrate

# Always load `backend/.env` even when the process cwd is the repo root
# (e.g. `flask --app backend.wsgi` from the project root), so Resend and
# other secrets are consistently available.
_backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(_backend_dir / ".env")
load_dotenv()  # optional override: cwd .env for local one-offs


def create_app(config_object: type | None = None) -> Flask:
    import click

    app = Flask(__name__)
    app.config.from_object(config_object or Config)

    db.init_app(app)
    migrate.init_app(app, db)
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

    from app.routes import admin_bp, auth_bp, chat_bp, health_bp, lectures_bp, study_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(lectures_bp, url_prefix="/api/lectures")
    app.register_blueprint(study_bp, url_prefix="/api/study")
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
    @click.option(
        "--upsert",
        is_flag=True,
        help="Merge into existing lecture_chunks instead of replacing all rows.",
    )
    def import_lectures(json_path, upsert):
        """Load lecture JSON into `lecture_chunks` (default: LECTURE_JSON_PATH / data file)."""
        from app.services.lectures.lecture_loader import import_lecture_json
        from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

        path = json_path or app.config["LECTURE_JSON_PATH"]
        if not path.exists():
            raise click.ClickException(f"Lecture file not found: {path}")
        with app.app_context():
            n = import_lecture_json(path, upsert=upsert)
            invalidate_lecture_cache()
            load_lecture_cache()
        print(f"Imported {n} lecture sections into lecture_chunks.")

    @app.cli.command("embed-chunks")
    @click.option("--batch-size", default=32, show_default=True, type=int)
    def embed_chunks_cmd(batch_size):
        """Compute OpenAI embeddings for all lecture_chunks (requires OPENAI_API_KEY)."""
        from app.services.embed_chunks_job import run_embed_chunks

        with app.app_context():
            n, model = run_embed_chunks(batch_size=max(1, min(batch_size, 128)))
        print(f"Embedded {n} chunks with model {model}.")

    @app.cli.command("run-eval")
    @click.option(
        "--dataset",
        "dataset_path",
        type=click.Path(path_type=Path, exists=True, dir_okay=False),
        default=None,
        help="Path to eval JSON (default: data/eval/l487_eval_suite.json under backend/).",
    )
    @click.option(
        "--run-name",
        default=None,
        help="Label for this run (default: timestamp).",
    )
    @click.option("--user-mode", default="auto", show_default=True)
    @click.option("--top-k", default=8, show_default=True, type=int)
    @click.option(
        "--compare-last",
        is_flag=True,
        help="Print delta vs the previous run with the same dataset_name.",
    )
    def run_eval_cmd(dataset_path, run_name, user_mode, top_k, compare_last):
        """Run static eval suite against ``run_reasoning_pipeline``; persist evaluation_runs rows."""
        from datetime import datetime, timezone

        from app.services.eval_run import run_eval_suite

        backend_dir = Path(__file__).resolve().parent.parent
        path = dataset_path or (backend_dir / "data" / "eval" / "l487_eval_suite.json")
        if not path.exists():
            raise click.ClickException(f"Dataset not found: {path}")
        label = run_name or datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%SZ")
        with app.app_context():
            er = run_eval_suite(
                path,
                label,
                user_mode=user_mode,
                top_k=top_k,
                compare_last=compare_last,
            )
        print(
            f"Evaluation run id={er.id} dataset={er.dataset_name} "
            f"passed={er.passed_cases}/{er.total_cases} failed={er.failed_cases} "
            f"overall_score={er.overall_score}"
        )

    with app.app_context():
        from sqlalchemy import inspect
        from sqlalchemy.exc import OperationalError

        from app.services.retrieval import load_lecture_cache

        if inspect(db.engine).has_table("lecture_chunks"):
            try:
                load_lecture_cache()
            except OperationalError:
                app.logger.warning(
                    "lecture_chunks schema mismatch (remove ling487.db, run init-db + import-lectures). "
                    "Retrieval cache not loaded."
                )

    return app
