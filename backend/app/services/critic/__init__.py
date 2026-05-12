"""LLM-as-judge critic (admin / eval tooling; not user-facing chat)."""

from app.services.critic.gemini_critic import CriticVerdict, run_gemini_critic

__all__ = ["CriticVerdict", "run_gemini_critic"]
