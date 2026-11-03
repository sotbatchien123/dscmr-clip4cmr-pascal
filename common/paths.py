"""Portable paths for files written into result JSON.

Results store paths relative to the repository root with '/' separators, so they do not
depend on the machine or operating system that produced them.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def to_repo(path):
    """Path as stored in results: repo-relative POSIX string (absolute only if outside the repo)."""
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()


def from_repo(stored):
    """Inverse of to_repo; also accepts older results written with Windows separators."""
    p = Path(str(stored).replace('\\', '/'))
    return p if p.is_absolute() else ROOT / p
