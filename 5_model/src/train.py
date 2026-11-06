from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from model_factory import build_pipeline
from training_utils import KEY, campaign_metrics, load_train_validation

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    cfg = json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8"))
    for k in ("prepared_dir", "output_dir", "models_dir"):
        p = Path(cfg[k])
        cfg[k] = str(p if p.is_absolute() else (ROOT / p).resolve())
    return cfg


def _w(df, path):
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig", lineterminator="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def baseline_score(X: pd.DataFrame, cfg) -> np.ndarray:
    """Долг прошлого месяца; неизвестный долг -> ниже любого известного (после известных)."""
    return X[cfg["baseline_feature"]].fillna(cfg["baseline_unknown_score"]).to_numpy(dtype=float)


def evaluate(name, part_name, part, score, cfg):
    summ, table, ranked = campaign_metrics(part["keys"], part["y"], score, cfg["campaign_fraction"])
    table.insert(0, "part", part_name)
    table.insert(0, "id", name)
    row = dict(id=name, part=part_name, macro_recall_at_K=summ["macro_recall_at_K"],
               macro_precision_at_K=summ["macro_precision_at_K"], macro_AP=summ["macro_average_precision"],
               pooled_recall_at_K=summ["pooled_recall_at_K"], hits=int(table["hits"].sum()),
               churn=int(table["n_churn"].sum()), K_total=int(table["K"].sum()), n_rows=summ["n_rows"],
               n_dates=summ["n_dates"])
    return row, table, ranked


def score_of(pipe, X):
    pos = int(np.flatnonzero(pipe.classes_ == 1)[0])
    return pipe.predict_proba(X)[:, pos]


def fit_one(exp, cfg, X, y):
    pipe = build_pipeline(exp, cfg["numeric_columns"], cfg["categorical_columns"], cfg["seed"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        pipe.fit(X, y)
        fit_s = time.perf_counter() - t0
    msgs = sorted({f"{w.category.__name__}: {str(w.message)[:120]}" for w in caught})
    return pipe, fit_s, msgs


def select(results: pd.DataFrame, order: list[str]) -> str:
    v = results[results["part"] == "validation"].set_index("id").loc[order]
    v = v.assign(order=range(len(v)))
    v = v.sort_values(["macro_recall_at_K", "macro_AP", "order"], ascending=[False, False, True])
    return v.index[0]


def run(config_path=None, output_dir=None, verbose=True) -> dict:
    cfg = load_config(config_path)
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
        cfg["models_dir"] = str(Path(output_dir) / "models")
    out, mdir = Path(cfg["output_dir"]), Path(cfg["models_dir"])
    out.mkdir(parents=True, exist_ok=True); mdir.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    prepared = Path(cfg["prepared_dir"])

    parts = load_train_validation(prepared, cfg["numeric_columns"], cfg["categorical_columns"])
    tr, va = parts["train"], parts["validation"]
    feat_version = {f: sha(prepared / f) for f in ("features.csv", "labels.csv", "split.csv")}
    say(f"[1] train {tr['X'].shape}, validation {va['X'].shape}; уходов train {tr['y'].sum()}, validation {va['y'].sum()}")

    input_checks = {
        "train dates": sorted(tr["keys"]["snapshot_at"].str[:10].unique().tolist()),
        "validation dates": sorted(va["keys"]["snapshot_at"].str[:10].unique().tolist()),
        "train rows per T": tr["keys"].groupby("snapshot_at").size().to_dict(),
        "validation rows per T": va["keys"].groupby("snapshot_at").size().to_dict(),
        "X columns": len(cfg["numeric_columns"]) + len(cfg["categorical_columns"]),
    }

    summary_rows, by_date, journal = [], [], []
    for pname, part in (("train", tr), ("validation", va)):
        r, t, _ = evaluate("baseline", pname, part, baseline_score(part["X"], cfg), cfg)
        summary_rows.append(r); by_date.append(t)
    journal.append(dict(id="baseline", family="rule", params="debt_prev_rub ↓, customer_id ↑; неизвестный долг после известных",
                        seed="", feature_version="stage4_v1", fit_seconds=np.nan, status="ok", warnings="", why="планка этапа 4"))

    pipes, val_ranked = {}, {}
    for exp in cfg["experiments"]:
        try:
            pipe, fit_s, msgs = fit_one(exp, cfg, tr["X"], tr["y"])
            status = "ok" if not msgs else "ok_with_warnings"
        except Exception as e:
            journal.append(dict(id=exp["id"], family=exp["family"], params=json.dumps(exp["params"]), seed=cfg["seed"],
                                feature_version="stage4_v1", fit_seconds=np.nan, status=f"error: {e}", warnings="", why=exp["why"]))
            continue
        pipes[exp["id"]] = pipe
        for pname, part in (("train", tr), ("validation", va)):
            r, t, ranked = evaluate(exp["id"], pname, part, score_of(pipe, part["X"]), cfg)
            summary_rows.append(r); by_date.append(t)
            if pname == "validation":
                val_ranked[exp["id"]] = ranked
        journal.append(dict(id=exp["id"], family=exp["family"], params=json.dumps(exp["params"], ensure_ascii=False),
                            seed=cfg["seed"], feature_version="stage4_v1", fit_seconds=round(fit_s, 3), status=status,
                            warnings=" | ".join(msgs), why=exp["why"]))
        say(f"[2] {exp['id']}: fit {fit_s:.2f} c, {status}")

    results = pd.DataFrame(summary_rows)
    tv = results.pivot(index="id", columns="part", values="macro_recall_at_K")
    results["train_minus_validation_recall"] = results["id"].map(tv["train"] - tv["validation"])

    ic = cfg["inner_check"]
    t_dates = tr["keys"]["snapshot_at"].str[:10]
    fit_m = (t_dates <= ic["fit_until"]).to_numpy()
    ev_m = ((t_dates >= ic["eval_from"]) & (t_dates <= ic["eval_until"])).to_numpy()
    inner_part = {"keys": tr["keys"][ev_m].reset_index(drop=True), "y": tr["y"][ev_m]}
    inner = []
    r, _, _ = evaluate("baseline", "inner", inner_part, baseline_score(tr["X"][ev_m], cfg), cfg); inner.append(r)
    for exp in cfg["experiments"]:
        if exp["id"] not in pipes:
            continue
        p, _, _ = fit_one(exp, cfg, tr["X"][fit_m], tr["y"][fit_m])   # подготовка заново внутри фолда
        r, _, _ = evaluate(exp["id"], "inner", inner_part, score_of(p, tr["X"][ev_m]), cfg); inner.append(r)
    inner = pd.DataFrame(inner)

    order = [e["id"] for e in cfg["experiments"] if e["id"] in pipes]
    chosen = select(results, order)
    vb = results[(results.id == "baseline") & (results.part == "validation")].iloc[0]
    vc = results[(results.id == chosen) & (results.part == "validation")].iloc[0]
    say(f"[3] выбран {chosen}: validation Recall@K {vc.macro_recall_at_K:.3f} ({vc.hits}/{vc.churn}) "
        f"против baseline {vb.macro_recall_at_K:.3f} ({vb.hits}/{vb.churn})")
    for j in journal:
        j["decision"] = "выбран" if j["id"] == chosen else ("рабочая альтернатива" if j["id"] == "baseline" else "не выбран")

    _, _, base_rk = evaluate("baseline", "validation", va, baseline_score(va["X"], cfg), cfg)
    m = val_ranked[chosen][KEY + ["target", "selected"]].merge(
        base_rk[KEY + ["selected"]], on=KEY, suffixes=("_model", "_baseline"), validate="one_to_one")
    churn = m[m["target"] == 1]
    overlap = pd.DataFrame([{
        "churn_validation": len(churn),
        "found_both": int((churn.selected_model & churn.selected_baseline).sum()),
        "found_only_model": int((churn.selected_model & ~churn.selected_baseline).sum()),
        "found_only_baseline": int((~churn.selected_model & churn.selected_baseline).sum()),
        "found_none": int((~churn.selected_model & ~churn.selected_baseline).sum()),
        "topK_overlap_customers": int((m.selected_model & m.selected_baseline).sum()),
        "K": int(m.selected_model.sum()),
    }])

    _w(overlap, out / "validation_overlap_vs_baseline.csv")
    _w(pd.DataFrame(journal), out / "experiments_journal.csv")
    _w(results, out / "metrics_summary.csv")
    _w(pd.concat(by_date, ignore_index=True), out / "metrics_by_date.csv")
    _w(inner, out / "inner_temporal_check.csv")
    rk = val_ranked[chosen]
    pred = rk[KEY + ["score", "rank", "selected"]].copy()
    _w(pred, out / "validation_predictions.csv")
    _w(rk[KEY + ["target"]].rename(columns={"target": "churn_target"}), out / "validation_labels_for_check.csv")
    bundle = {"pipeline": pipes[chosen], "numeric_columns": cfg["numeric_columns"],
              "categorical_columns": cfg["categorical_columns"], "selected_experiment": chosen,
              "config": {k: v for k, v in cfg.items() if k not in ("prepared_dir", "output_dir", "models_dir")}}
    joblib.dump(bundle, mdir / "candidate.joblib")
    card = dict(selected_experiment=chosen, params=next(e for e in cfg["experiments"] if e["id"] == chosen),
                input_columns_in_order=cfg["numeric_columns"] + cfg["categorical_columns"],
                x_width_after_preprocess=len(pipes[chosen].named_steps["preprocess"].get_feature_names_out()),
                seed=cfg["seed"], feature_version=cfg["feature_version"], feature_files_sha256_16=feat_version,
                versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                              sklearn=sklearn.__version__, joblib=joblib.__version__),
                selection_rule=cfg["selection_rule"], input_checks=input_checks,
                validation=dict(candidate=vc.to_dict(), baseline=vb.to_dict(),
                                extra_hits_vs_baseline=int(vc.hits - vb.hits)),
                test_used=False, retrained_on_train_plus_validation=False)
    (mdir / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    say(f"Готово: {out}, модель: {mdir / 'candidate.joblib'}")
    return dict(results=results, journal=pd.DataFrame(journal), inner=inner, chosen=chosen, card=card)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    a = ap.parse_args()
    run(a.config)
