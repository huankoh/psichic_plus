"""Preserve the old module identity, including callers that replace Predictor."""
import sys

from .psichic_plus import predictor

sys.modules[__name__] = predictor
