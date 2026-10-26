"""S5 + Q3 speaker-verification system (VLSP 2026 SRED, team greenearth03). See ../README.md."""
from .audio import duration_s, load_wav, snr_est  # noqa: F401
from .model import count_params, embed, load_s5  # noqa: F401
from .qmf import Q3  # noqa: F401
