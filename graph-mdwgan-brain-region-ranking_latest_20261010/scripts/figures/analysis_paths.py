"""Portable repository paths shared by manuscript figure scripts."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "results" / "figures"
REFERENCE_ROOT = REPO_ROOT / "results" / "reference"

PRIMARY_CV_RESULTS = REFERENCE_ROOT / "cv_results"
CORRECTED_SECONDARY_RESULTS = REFERENCE_ROOT / "corrected_secondary"
SEED_RESULTS = REFERENCE_ROOT / "seed_quality"
PUBMED_RESULTS = REFERENCE_ROOT / "pubmed_concordance_corrected"
LOW_DATA_RESULTS = REFERENCE_ROOT / "low_data"
ROBUSTNESS_SOURCES = REFERENCE_ROOT / "robustness_sources"
ROS_FIXUNIQUE_RESULTS = REFERENCE_ROOT / "ros_fixunique"


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}")
    return path
