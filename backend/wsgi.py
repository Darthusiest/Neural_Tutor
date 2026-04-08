"""WSGI entry: `flask --app wsgi run` or Gunicorn on Render."""
from app import create_app

app = create_app()
