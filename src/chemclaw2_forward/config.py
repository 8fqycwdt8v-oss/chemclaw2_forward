"""Runtime configuration loaded from env / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    host: str = "0.0.0.0"
    port: int = 8765

    enabled_forward_models: str = Field(default="*", description="Comma list or '*' for all.")
    enabled_conditions_models: str = Field(default="*", description="Comma list or '*' for all.")
    disabled_models: str = Field(default="", description="Comma list of predictor IDs to force off.")

    model_cache_dir: Path = Field(default=Path.home() / ".cache" / "chemclaw2_forward")
    hf_home: Path | None = Field(default=None, description="Override HF_HOME for model downloads.")

    device: str = Field(default="auto", description="'cpu', 'cuda', 'cuda:0', or 'auto'.")
    default_top_k: int = 5

    # Per-model trust priors used by the aggregator. Higher = more weight in voting.
    # These are sensible defaults from published benchmarks; can be overridden via env JSON.
    model_trust_priors: dict[str, float] = Field(
        default_factory=lambda: {
            # Forward
            "reaction_t5_v2": 1.00,        # ~97.5% top-1 USPTO-MIT (Sagawa 2024)
            "molecular_transformer": 0.90,  # ~90% top-1
            "t5chem": 0.92,
            "chemformer": 0.85,
            "megan": 0.80,
            "graphrxn": 0.80,
            # Conditions
            "parrot": 0.95,                 # +13.44% top-3 over Coley baseline
            "rxn_insight": 0.70,            # rule-based, fast but coarse
            "two_stage_dnn": 0.90,          # 73% top-10 exact match
            "reagents_mt": 0.80,
            "askcos_condition": 0.85,
        }
    )

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
        _settings = Settings()
        _settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return _settings
