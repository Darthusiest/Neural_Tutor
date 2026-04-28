"""CORS frontend origin resolution (localhost vs 127.0.0.1 in dev)."""

from __future__ import annotations

import os

from app import config as config_module


def test_resolve_frontend_origins_adds_localhost_twin(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173")
    monkeypatch.delenv("PRODUCTION_LIKE", raising=False)
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("FRONTEND_ORIGIN_DEV_ALIASES", "1")
    out = config_module._resolve_frontend_origins(os.getenv("FRONTEND_ORIGIN"))
    assert "http://127.0.0.1:5173" in out
    assert "http://localhost:5173" in out


def test_resolve_frontend_origins_no_alias_in_production(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173")
    monkeypatch.setenv("FLASK_ENV", "production")
    out = config_module._resolve_frontend_origins(os.getenv("FRONTEND_ORIGIN"))
    assert out == ["http://127.0.0.1:5173"]


def test_resolve_frontend_origins_comma_list(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "http://a.example:5173,http://b.example:5173")
    monkeypatch.setenv("FLASK_ENV", "production")
    out = config_module._resolve_frontend_origins(os.getenv("FRONTEND_ORIGIN"))
    assert out == ["http://a.example:5173", "http://b.example:5173"]
