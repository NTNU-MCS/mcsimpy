"""Top-level package for mcsimpy."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcsimpy")
except PackageNotFoundError:
    __version__ = "0+unknown"