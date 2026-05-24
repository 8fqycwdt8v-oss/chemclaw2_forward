"""Runtime configuration loaded from env / .env."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Server configuration.

    Values are read from environment variables (case-insensitive) or a `.env` file
    in the working directory. Predictor selection is comma-separated:

        ENABLED_FORWARD_MODELS=reaction_t5_v2,molecular_transformer
        ENABLED_CONDITIONS_MODELS=rxn_insight,parrot
        DISABLED_MODELS=megan

    Special value `*` means "all registered models".
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8765

    # --- Model selection ---
    enabled_forward_models: str = Field(default="*", description="Comma list or '*' for all.")
    enabled_conditions_models: str = Field(default="*", description="Comma list or '*' for all.")
    disabled_models: str = Field(default="", description="Comma list of predictor IDs to force off.")

    # --- Caches & checkpoints ---
    model_cache_dir: Path = Field(default=Path.home() / ".cache" / "chemclaw2_forward")
    hf_home: Path | None = Field(default=None, description="Override HF_HOME for model downloads.")

    # --- Compute ---
    device: str = Field(default="auto", description="'cpu', 'cuda', 'cuda:0', or 'auto'.")
    default_top_k: int = 5

    # --- Anthropic / Claude predictor (Phase C) ---
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.",
    )
    claude_model_id: str = Field(
        default="claude-sonnet-4-6",
        description="Anthropic model identifier for the ClaudeForward / ClaudeConditions predictors.",
    )
    claude_max_tokens: int = Field(default=2048, ge=64, le=8192)
    claude_n_examples: int = Field(
        default=3, ge=0, le=10, description="Few-shot examples to include in Claude prompts."
    )

    # --- Prediction cache (Phase D) ---
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=86_400)  # 1 day

    # --- Mixture-of-Experts gating (Phase D) ---
    use_class_priors: bool = Field(
        default=True,
        description="Enable per-reaction-class trust priors in the aggregator.",
    )
    trust_priors_path: Path = Field(
        default=Path.home() / ".cache" / "chemclaw2_forward" / "trust_priors.json",
        description="JSON file with per-class trust priors (written by scripts/calibrate_priors.py).",
    )

    # --- Per-model trust priors used by the aggregator. Higher = more weight in voting.
    # These are sensible defaults from published benchmarks; can be overridden via env JSON.
    model_trust_priors: dict[str, float] = Field(
        default_factory=lambda: {
            # Forward
            "reaction_t5_v2": 1.00,         # ~97.5% top-1 USPTO-MIT (Sagawa 2024)
            "molecular_transformer": 0.90,  # ~90% top-1
            "t5chem": 0.92,
            "chemformer": 0.85,
            "megan": 0.80,
            "graphrxn": 0.80,
            "claude": 0.75,                 # LLM voter — diverse but less calibrated than seq2seq
            # Conditions
            "parrot": 0.95,                 # +13.44% top-3 over Coley baseline
            "rxn_insight": 0.70,            # rule-based, fast but coarse
            "two_stage_dnn": 0.90,          # 73% top-10 exact match
            "reagents_mt": 0.80,
            "askcos_condition": 0.85,
        }
    )

    # Per-reaction-class trust priors, populated by `scripts/calibrate_priors.py`.
    # Keys are reaction class labels (see meta/classifier.py). When empty (default),
    # the aggregator falls back to `model_trust_priors`.
    model_trust_priors_by_class: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("model_trust_priors", mode="before")
    @classmethod
    def _parse_priors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("model_trust_priors_by_class", mode="before")
    @classmethod
    def _parse_class_priors(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    def parse_enabled(self, raw: str) -> set[str] | None:
        """Return None for '*', else a set of normalised predictor IDs."""
        raw = raw.strip()
        if raw == "*" or raw == "":
            return None
        return {p.strip() for p in raw.split(",") if p.strip()}

    def parse_disabled(self) -> set[str]:
        return {p.strip() for p in self.disabled_models.split(",") if p.strip()}

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        s = Settings()
        s.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Load per-class priors from disk if available; in-env JSON overrides this.
        if not s.model_trust_priors_by_class and s.trust_priors_path.exists():
            from .meta.trust_priors import load_priors_file

            s.model_trust_priors_by_class = load_priors_file(s.trust_priors_path)
            if s.model_trust_priors_by_class:
                logger.info(
                    "Loaded per-class trust priors for %d classes from %s",
                    len(s.model_trust_priors_by_class),
                    s.trust_priors_path,
                )

        _settings = s
    return _settings


def reset_settings_for_tests() -> None:
    """Force a fresh Settings instance on next get_settings() call."""
    global _settings
    _settings = None
