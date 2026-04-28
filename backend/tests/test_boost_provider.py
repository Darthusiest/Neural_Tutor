"""Boosted Explanation provider chain (BOOST_PRIMARY_PROVIDER / BOOST_FALLBACK_PROVIDER)."""

from __future__ import annotations

from app.services.generation.boost_provider import (
    BoostAttempt,
    boost_provider_chain,
    first_runnable_provider,
)


def test_default_chain_openai_then_gemini(app):
    with app.app_context():
        app.config["BOOST_PRIMARY_PROVIDER"] = "openai"
        app.config["BOOST_FALLBACK_PROVIDER"] = "gemini"
        app.config["OPENAI_API_KEY"] = "sk-test"
        app.config["GEMINI_API_KEY"] = "gk-test"
        app.config["GOOGLE_API_KEY"] = ""
        chain = boost_provider_chain()
        assert [a.provider for a in chain] == ["openai", "gemini"]
        assert all(a.has_key for a in chain)
        first = first_runnable_provider(chain)
        assert first is not None and first.provider == "openai"


def test_chain_skips_none_and_dedupes(app):
    with app.app_context():
        app.config["BOOST_PRIMARY_PROVIDER"] = "openai"
        app.config["BOOST_FALLBACK_PROVIDER"] = "openai"
        app.config["OPENAI_API_KEY"] = "sk-test"
        app.config["GEMINI_API_KEY"] = ""
        app.config["GOOGLE_API_KEY"] = ""
        chain = boost_provider_chain()
        assert [a.provider for a in chain] == ["openai"]


def test_chain_invalid_value_falls_back_to_none(app):
    with app.app_context():
        app.config["BOOST_PRIMARY_PROVIDER"] = "claude"
        app.config["BOOST_FALLBACK_PROVIDER"] = "gemini"
        app.config["OPENAI_API_KEY"] = ""
        app.config["GEMINI_API_KEY"] = "gk-test"
        chain = boost_provider_chain()
        assert [a.provider for a in chain] == ["gemini"]


def test_first_runnable_skips_keyless_primary(app):
    with app.app_context():
        app.config["BOOST_PRIMARY_PROVIDER"] = "openai"
        app.config["BOOST_FALLBACK_PROVIDER"] = "gemini"
        app.config["OPENAI_API_KEY"] = ""
        app.config["GEMINI_API_KEY"] = "gk-test"
        chain = boost_provider_chain()
        first = first_runnable_provider(chain)
        assert first is not None and first.provider == "gemini"


def test_first_runnable_returns_none_when_no_keys(app):
    with app.app_context():
        app.config["OPENAI_API_KEY"] = ""
        app.config["GEMINI_API_KEY"] = ""
        app.config["GOOGLE_API_KEY"] = ""
        first = first_runnable_provider(
            [BoostAttempt("openai", False), BoostAttempt("gemini", False)]
        )
        assert first is None


def test_legacy_openai_fallback_envvar_keeps_gemini_primary(monkeypatch):
    """Loading Config with OPENAI_BOOST_FALLBACK=1 (and no new vars) keeps Gemini primary."""
    from importlib import reload

    monkeypatch.delenv("BOOST_PRIMARY_PROVIDER", raising=False)
    monkeypatch.delenv("BOOST_FALLBACK_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_BOOST_FALLBACK", "1")
    import app.config as config_mod

    reload(config_mod)
    try:
        cfg = config_mod.Config
        assert cfg.BOOST_PRIMARY_PROVIDER == "gemini"
        assert cfg.BOOST_FALLBACK_PROVIDER == "openai"
    finally:
        monkeypatch.delenv("OPENAI_BOOST_FALLBACK", raising=False)
        reload(config_mod)
