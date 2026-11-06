"""Проверки этапа 5."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from train import load_config, run
from training_utils import load_train_validation, read_csv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    return run(output_dir=a, verbose=False), run(output_dir=b, verbose=False), a, b


def test_inputs_joined_by_key_and_no_test():
    cfg = load_config()
    parts = load_train_validation(cfg["prepared_dir"], cfg["numeric_columns"], cfg["categorical_columns"])
    split = read_csv(Path(cfg["prepared_dir"]) / "split.csv")
    n = split["split"].value_counts()
    assert len(parts["train"]["y"]) == n["train"] and len(parts["validation"]["y"]) == n["validation"]
    assert set(parts) == {"train", "validation"}                       # test не загружается
    assert parts["train"]["keys"]["snapshot_at"].max() < parts["validation"]["keys"]["snapshot_at"].min()


def test_fit_only_on_train(runs):
    cfg = load_config()
    parts = load_train_validation(cfg["prepared_dir"], cfg["numeric_columns"], cfg["categorical_columns"])
    import joblib
    pipe = joblib.load(runs[2] / "models" / "candidate.joblib")["pipeline"]
    med = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"].statistics_
    assert np.allclose(med, parts["train"]["X"][cfg["numeric_columns"]].median().to_numpy(), equal_nan=True)


def test_same_K_and_base_for_all(runs):
    by = pd.read_csv(runs[2] / "metrics_by_date.csv", sep=";", encoding="utf-8-sig")
    assert by.groupby(["part", "snapshot_at"])[["n_active", "K", "n_churn"]].nunique().max().max() == 1


def test_baseline_matches_stage4(runs):
    r = runs[0]["results"].set_index(["id", "part"])
    assert (r.loc[("baseline", "train"), "hits"], r.loc[("baseline", "train"), "churn"]) == (175, 440)


def test_reproducible_training(runs):
    pd.testing.assert_frame_equal(runs[0]["results"], runs[1]["results"])
    a = pd.read_csv(runs[2] / "validation_predictions.csv", sep=";", encoding="utf-8-sig")
    b = pd.read_csv(runs[3] / "validation_predictions.csv", sep=";", encoding="utf-8-sig")
    pd.testing.assert_frame_equal(a, b)


def test_selection_rule(runs):
    r = runs[0]["results"]
    v = r[(r.part == "validation") & (r.id != "baseline")]
    assert runs[0]["chosen"] == v.sort_values("macro_recall_at_K", ascending=False, kind="stable").iloc[0]["id"]


def test_reload_without_fit_in_new_process():
    res = subprocess.run([sys.executable, str(ROOT / "check_candidate.py")], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    assert "загружен без fit" in res.stdout
