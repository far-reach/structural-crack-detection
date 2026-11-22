"""Data locations. Nothing here lives inside the git repository.

Set INSPECTSWARM_ROOT to the folder that holds `datasets/` and `cache/`
(default: the parent folder of the repository).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("INSPECTSWARM_ROOT", REPO.parent))  # default: the folder that contains the repository
DATASETS = ROOT / "datasets"
CACHE = ROOT / "cache"
RESULTS = REPO / "results_v2"
WEIGHTS = REPO / "models"  # git-ignored; released through GitHub Releases

SDNET = DATASETS / "sdnet2018"
RAIDATHMANE = DATASETS / "raidathmane" / "spalling_corrosion_patches"
CODEBRIM = DATASETS / "codebrim" / "extracted" / "classification_dataset"
BAM = DATASETS / "bam_ultrasonic"
