import json
import subprocess
import sys
from pathlib import Path

import openpyxl
import pytest


SCORER = Path(__file__).resolve().parents[1] / "inference/score_perception.py"
RAW_COLUMNS = ["dataset", "model", "family", "case", "n_objects", "acc_sample", "token_len",
               "path", "line_no", "label", "final_answer", "final_answer_raw", "source",
               "used_idx", "parse_error"]
SUMMARY_COLUMNS = ["Case", "# of distractor", "acc", "avg token length"]
BOOTSTRAP = """
import runpy, sys
sys.modules['transformers'] = None  # Force the existing whitespace fallback, without downloads.
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def score(root, output):
    run = subprocess.run([sys.executable, "-c", BOOTSTRAP, str(SCORER),
                          "--root", str(root), "--out", str(output)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    workbook = openpyxl.load_workbook(output, data_only=True)
    sheets = {sheet.title: list(sheet.values) for sheet in workbook}
    workbook.close()
    return run, sheets


def write_predictions(root, filename, records):
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def prediction(answer="bird", **extra):
    return {"label": "bird", "answer": f"<answer>{answer}</answer>", **extra}


def test_readme_sample_appears_in_raw_and_summary(tmp_path):
    root = tmp_path / "results"
    path = write_predictions(root, "qwen3-thinking-bird-4-conflicting_s0.jsonl", [prediction()])
    _, sheets = score(root, tmp_path / "scores.xlsx")
    assert list(sheets) == ["predictions_raw", "Qwen3-thinking", "GLM", "Intern", "OneVision"]
    assert list(sheets["predictions_raw"][0]) == RAW_COLUMNS
    raw, = sheets["predictions_raw"][1:]
    assert raw == ("Idis-perception", "Qwen3-thinking", "qwen3-thinking", "Conflicting", 4,
                   1, 1, str(path), 1, "bird", "bird", "bird", "answer_tag", "last", None)
    assert sheets["Qwen3-thinking"] == [tuple(SUMMARY_COLUMNS), ("Conflicting", 4, 1, 1)]
    for model in ("GLM", "Intern", "OneVision"):
        assert sheets[model] == [tuple(SUMMARY_COLUMNS)]


@pytest.mark.parametrize("suffix", ["", "_s0", "_s1", "_s4", "_s12"])
def test_conditions_and_sample_indices_in_workbook(tmp_path, suffix):
    root = tmp_path / "results"
    for case in ("aligned", "conflicting", "irrelevant"):
        write_predictions(root, f"qwen3-thinking-bird-4-{case}{suffix}.jsonl",
                          [prediction(), prediction("dog")])
    _, sheets = score(root, tmp_path / "scores.xlsx")
    assert sheets["Qwen3-thinking"] == [tuple(SUMMARY_COLUMNS)] + [
        (case, 4, 0.5, 1) for case in ("Aligned", "Conflicting", "Irrelevant")
    ]
    assert len(sheets["predictions_raw"]) == 7
    assert sorted(row[5] for row in sheets["predictions_raw"][1:]) == [0, 0, 0, 1, 1, 1]


def test_record_n_objects_takes_precedence(tmp_path):
    root = tmp_path / "results"
    write_predictions(root, "qwen3-thinking-bird-4-aligned_s1.jsonl", [prediction(n_objects=7)])
    _, sheets = score(root, tmp_path / "scores.xlsx")
    assert sheets["predictions_raw"][1][4] == 7
    assert sheets["Qwen3-thinking"][1] == ("Aligned", 7, 1, 1)


def test_unknown_condition_remains_unknown(tmp_path):
    root = tmp_path / "results"
    write_predictions(root, "qwen3-thinking-bird-4-mystery_s0.jsonl", [prediction()])
    _, sheets = score(root, tmp_path / "scores.xlsx")
    assert sheets["predictions_raw"][1][3:6] == ("Unknown", 4, 1)
    assert sheets["Qwen3-thinking"] == [tuple(SUMMARY_COLUMNS)]


def test_suffix_is_only_removed_from_terminal_filename_component(tmp_path):
    root = tmp_path / "folder_s9"
    write_predictions(root, "qwen3-thinking-bird_s3-4-conflicting_s2.jsonl", [prediction()])
    _, sheets = score(root, tmp_path / "scores.xlsx")
    assert sheets["Qwen3-thinking"][1] == ("Conflicting", 4, 1, 1)
    assert "bird_s3" in sheets["predictions_raw"][1][7]


@pytest.mark.parametrize("filename", ["qwen3-thinking-bird-4-aligned_sx.jsonl",
                                       "qwen3-thinking-bird-4-aligned_s1_extra.jsonl"])
def test_malformed_sample_suffix_is_not_recognized(tmp_path, filename):
    root = tmp_path / "results"
    write_predictions(root, filename, [prediction()])
    _, sheets = score(root, tmp_path / "scores.xlsx")
    assert sheets["predictions_raw"][1][3] == "Unknown"
    assert sheets["Qwen3-thinking"] == [tuple(SUMMARY_COLUMNS)]
