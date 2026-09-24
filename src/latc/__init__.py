"""LATC: exact inference and acquisition experiments for noisy affine prediction."""
from .posterior import regional_posterior, predict_two_regions
__version__ = "0.1.0"
__all__ = ["regional_posterior", "predict_two_regions", "__version__"]
