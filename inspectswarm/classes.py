"""Single source of truth for every class list in the project.

The v1.0 scripts decoded model outputs with a hand-written list whose order
differed from the order used in training. Every module must import its class
order from here; tests/test_classes.py guards this.
"""

# Output order of the released v1.0 Stage 1 classifier. Training used
# torchvision ImageFolder, which sorts class folders alphabetically.
V1_CLASSES = ("corrosion", "crack", "intact", "spalling")

# Order that four v1.0 scripts used for decoding. Kept only so the published
# figures can be reproduced and audited. Never use it for new results.
V1_WRONG_DECODE_ORDER = ("crack", "intact", "corrosion", "spalling")

# CODEBRIM defect targets, in the order of the dataset's XML annotation.
# Background is the all-zero target vector.
CODEBRIM_DEFECTS = ("Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain")

# Mapping from the v1 single-label classes to CODEBRIM targets, used for
# cross-source evaluation. Corrosion maps to either corrosion-related target.
V1_TO_CODEBRIM = {
    "crack": ("Crack",),
    "spalling": ("Spallation",),
    "corrosion": ("CorrosionStain", "ExposedBars"),
    "intact": (),
}

SEVERITY = {"crack": "CRITICAL", "corrosion": "HIGH", "spalling": "MEDIUM", "intact": "NONE"}


def decode_v1(index: int) -> str:
    """Class name for an output index of the v1.0 Stage 1 classifier."""
    return V1_CLASSES[index]
