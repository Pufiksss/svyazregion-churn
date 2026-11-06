"""Проверка сохранённого кандидата в новом процессе без fit:
    python check_candidate.py
Загружает models/candidate.joblib, считает validation-score, сверяет ключи, score, rank, top-K и метрики
с output/validation_predictions.csv и output/metrics_summary.csv. Любой вызов fit запрещён."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import sklearn.base  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def _forbidden(*a, **k):
    raise RuntimeError("fit запрещён при проверке сохранённого кандидата")


Pipeline.fit = _forbidden
Pipeline.fit_transform = _forbidden

from check_saved_model import check  # noqa: E402  (стартовый скрипт курса)
from training_utils import campaign_metrics, read_csv, KEY  # noqa: E402

cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
prepared = (ROOT / cfg["prepared_dir"]).resolve()
out = ROOT / "output"
check(ROOT / "models" / "candidate.joblib", prepared, out / "reloaded_validation.csv", out / "validation_predictions.csv")

new = read_csv(out / "reloaded_validation.csv")
lab = read_csv(out / "validation_labels_for_check.csv")
m = new.merge(lab, on=KEY, validate="one_to_one")
summ, table, _ = campaign_metrics(m[KEY], m["churn_target"].astype(int).to_numpy(), m["score"].astype(float).to_numpy(),
                                  cfg["campaign_fraction"])
saved = pd.read_csv(out / "metrics_summary.csv", sep=";", encoding="utf-8-sig")
card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
ref = saved[(saved.id == card["selected_experiment"]) & (saved.part == "validation")].iloc[0]
assert np.isclose(summ["macro_recall_at_K"], ref.macro_recall_at_K), "Recall@K отличается"
assert np.isclose(summ["macro_average_precision"], ref.macro_AP), "AP отличается"
print(f"OK: {card['selected_experiment']} загружен без fit; validation Recall@K = {summ['macro_recall_at_K']:.4f} "
      f"({int(table.hits.sum())}/{int(table.n_churn.sum())}), AP = {summ['macro_average_precision']:.4f} — совпадает с обучением")
