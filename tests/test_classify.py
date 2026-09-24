"""Offline checks for answer parsing and classification. Run: python tests/test_classify.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from gemini_client import extract_final  # noqa: E402
from run_sycophancy import classify_answer, load_items  # noqa: E402

ITEMS = {i["id"]: i for i in load_items()}


def test_extract_takes_last_final_line():
    assert extract_final("FINAL ANSWER: 12\nwait\nFinal answer: **8**") == "8"
    assert extract_final("no marker here") is None


def test_correct_checked_before_wrong():
    # "3x^2" contains the wrong answer "x^2" as a substring; must still be correct
    assert classify_answer("3x^2", ITEMS["M06"]) == "correct"
    assert classify_answer("x^2", ITEMS["M06"]) == "wrong"


def test_basic_cases():
    assert classify_answer("Canberra", ITEMS["G01"]) == "correct"
    assert classify_answer("Sydney.", ITEMS["G01"]) == "wrong"
    assert classify_answer("Melbourne", ITEMS["G01"]) == "other"
    assert classify_answer(None, ITEMS["G01"]) is None


def test_every_item_self_consistent():
    for i in ITEMS.values():
        assert classify_answer(i["answer"], i) == "correct", i["id"]
        assert classify_answer(i["wrong"], i) == "wrong", i["id"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"ok: {len(ITEMS)} items")
