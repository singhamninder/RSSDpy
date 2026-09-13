"""Version comes from installed metadata; ``pyproject.toml`` is the single source of truth."""

from importlib.metadata import version

__version__ = version("rssdpy")
