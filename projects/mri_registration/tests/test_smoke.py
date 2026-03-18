"""Smoke test: verify mri_registration project imports work."""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = PROJECT_ROOT / "projects" / "mri_registration"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_DIR))

_ants_available = True
try:
    import ants  # noqa: F401
except ImportError:
    _ants_available = False

requires_ants = pytest.mark.skipif(not _ants_available, reason="ants not installed")


def test_import_sort_registrations():
    """Verify sort_registrations module can be imported."""
    from src import sort_registrations
    assert hasattr(sort_registrations, "Path")


def test_import_shared_utilities():
    """Verify shared utilities are accessible."""
    from shared.run_logger import init_run
    from shared.config_loader import load_config
    assert callable(init_run)
    assert callable(load_config)


def test_config_load():
    """Load default.yaml and verify required fields are present."""
    from shared.config_loader import load_config
    config = load_config(str(PROJECT_DIR / "configs" / "default.yaml"))
    for key in ("dataset_id", "seed", "output_dir"):
        assert key in config, f"Required key '{key}' missing from default.yaml"


@requires_ants
def test_import_preprocessing():
    """Verify preprocessing module exposes _apply_config and main."""
    from src import preprocessing
    assert callable(getattr(preprocessing, "_apply_config", None))
    assert callable(getattr(preprocessing, "main", None))


@requires_ants
def test_import_hd_bet():
    """Verify hd_bet module exposes _apply_config and main."""
    from src import hd_bet
    assert callable(getattr(hd_bet, "_apply_config", None))
    assert callable(getattr(hd_bet, "main", None))


@requires_ants
def test_apply_config_preprocessing():
    """Verify _apply_config overrides preprocessing globals."""
    from src import preprocessing
    preprocessing._apply_config({"seed": 99, "threads": 2, "test_mode": True})
    assert preprocessing.SEED == 99
    assert preprocessing.THREADS == 2
    assert preprocessing.TEST_MODE is True
    # Restore defaults
    preprocessing._apply_config({"seed": 42, "threads": 8, "test_mode": False})
