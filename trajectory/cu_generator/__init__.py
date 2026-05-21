"""Phase 2 segment data and Phase 3 build pipeline."""

from typing import Any, Optional, Sequence


def build_phase3(argv: Optional[Sequence[str]] = None) -> Any:
    """Lazy wrapper around the Phase 3 builder CLI."""

    from .builder import main as _main

    return _main(argv)


__all__ = ["build_phase3"]
