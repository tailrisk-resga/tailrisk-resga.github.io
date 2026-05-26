"""Compatibility exports for competing baseline models."""

from models.garch import run_garch
from models.gas import run_gas

__all__ = ["run_garch", "run_gas"]
