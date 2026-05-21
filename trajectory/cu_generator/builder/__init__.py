"""Phase 3 Canonical Unit builder package."""

from typing import Any, Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> Any:
    """Lazy wrapper around the Phase 3 CLI entrypoint."""

    from .phase3_builder import main as _main

    return _main(argv)


__all__ = ["main"]
