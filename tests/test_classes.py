"""Integrity guards for class order. The v1.0 label bug must not come back."""
import ast
from pathlib import Path

from inspectswarm import classes

REPO = Path(__file__).resolve().parents[1]


def test_v1_order_is_alphabetical_like_imagefolder():
    assert list(classes.V1_CLASSES) == sorted(classes.V1_CLASSES)
    assert classes.V1_CLASSES == ("corrosion", "crack", "intact", "spalling")


def test_decode_matches_shared_definition():
    assert [classes.decode_v1(i) for i in range(4)] == list(classes.V1_CLASSES)


def test_wrong_order_is_a_permutation_and_differs():
    assert sorted(classes.V1_WRONG_DECODE_ORDER) == sorted(classes.V1_CLASSES)
    assert classes.V1_WRONG_DECODE_ORDER != classes.V1_CLASSES


def test_mapping_targets_exist():
    for v1, targets in classes.V1_TO_CODEBRIM.items():
        assert v1 in classes.V1_CLASSES
        assert all(t in classes.CODEBRIM_DEFECTS for t in targets)


def test_no_new_module_hardcodes_a_class_list():
    """New code (package and experiments) must import class lists, not retype them."""
    names = set(classes.V1_CLASSES) | set(classes.CODEBRIM_DEFECTS)
    offenders = []
    for folder in ("inspectswarm", "experiments"):
        for py in (REPO / folder).rglob("*.py"):
            if py.name == "classes.py":
                continue
            for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
                if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 3:
                    vals = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                    if len(vals) >= 3 and all(v in names for v in vals):
                        offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, offenders
