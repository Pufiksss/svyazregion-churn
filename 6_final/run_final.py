"""Итоговая оценка зафиксированного кандидата без fit и выбора по test."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from campaign_metrics import KEY, campaign_metrics
from evaluation_utils import load_holdout, predict_risk, segment_metrics


def sha(path, text=False):
    b = Path(path).read_bytes()
    return hashlib.sha256(b.replace(b'\r\n', b'\n') if text else b).hexdigest()


def freeze_inputs():
    """Проверка протокола выполняется до расчёта каких-либо test-метрик."""
    cfg = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    paths = {k: (ROOT / v).resolve() for k, v in cfg['paths'].items()}
    for name, path in paths.items():
        if sha(path, name != 'model') != cfg['input_sha256'][name]:
            raise ValueError(f'Изменилась зафиксированная версия входа: {name}. Не подбирать по test.')
    if cfg['test_used_for_selection'] or cfg['selected_experiment'] != 'exp04_hgb_leaf7':
        raise ValueError('Нарушен зафиксированный протокол выбора')
    return cfg, paths


def forbidden_fit(*args, **kwargs):
    raise RuntimeError('Обучение запрещено в итоговой оценке')


def load_frozen(paths, cfg):
    bundle = joblib.load(paths['model'])
    if bundle['selected_experiment'] != cfg['selected_experiment']:
        raise ValueError('Загружен другой кандидат')
    if list(bundle['pipeline'].feature_names_in_) != bundle['numeric_columns'] + bundle['categorical_columns']:
        raise ValueError('Неверный порядок входных столбцов')
    return bundle


def baseline_score(X, cfg):
    return X[cfg['baseline_feature']].fillna(cfg['baseline_unknown_score']).to_numpy(float)


def write_csv(frame, path):
    frame.to_csv(path, sep=';', encoding='utf-8-sig', index=False, lineterminator='\n')


def validation_record(path, model):
    df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    row = df[(df.id == model) & (df.part == 'validation')]
    if len(row) != 1:
        raise ValueError('Нужна одна строка прежней validation-оценки')
    r = row.iloc[0]
    return dict(period='validation', approach=model, snapshot_at='2026-08-01 00:00:00',
                n_active=int(r.n_rows), K=int(r.K_total), n_churn=int(r.churn), hits=int(r.hits),
                recall_at_K=float(r.macro_recall_at_K), precision_at_K=float(r.macro_precision_at_K),
                average_precision=float(r.macro_AP))


def run(output=None):
    cfg, paths = freeze_inputs()
    # Защита от случайного fit: также действует внутри загруженного Pipeline.
    Pipeline.fit = forbidden_fit
    Pipeline.fit_transform = forbidden_fit
    out = Path(output or ROOT / 'output')
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_frozen(paths, cfg)
    pre, estimator = bundle['pipeline'].named_steps.values()
    pre.fit = forbidden_fit
    pre.fit_transform = forbidden_fit
    estimator.fit = forbidden_fit

    # Валидация не используется для нового выбора: проверяется совместимость старой модели.
    val = load_holdout(paths['features'].parent, bundle, 'validation', cfg['observation_end'])
    _, val_table, val_rk = campaign_metrics(val['keys'], val['y'], predict_risk(bundle['pipeline'], val['X']), cfg['campaign_fraction'])
    expected = pd.read_csv(paths['validation_predictions'], sep=';', encoding='utf-8-sig')
    pair = val_rk.merge(expected, on=KEY, validate='one_to_one', suffixes=('_new','_old'))
    if len(pair) != len(expected) or not np.allclose(pair.score_new, pair.score_old, rtol=1e-12, atol=1e-12):
        raise ValueError('Сохранённая модель не воспроизводит прежние validation-score')
    if not (pair.selected_new == pair.selected_old).all() or not (pair.rank_new == pair.rank_old).all():
        raise ValueError('Изменился validation top-K')
    rows = [validation_record(paths['validation_metrics'], k) for k in ('baseline', cfg['selected_experiment'])]
    if int(val_table.hits.sum()) != rows[1]['hits']:
        raise ValueError('Validation не совпала с этапом 5')

    test = load_holdout(paths['features'].parent, bundle, 'test', cfg['observation_end'])
    if set(test['keys'].snapshot_at) != {cfg['test_snapshot']}:
        raise ValueError('Неверная дата test')
    lab = pd.read_csv(paths['labels'], sep=';', encoding='utf-8-sig', dtype=str)
    if set(pd.to_datetime(lab.loc[lab.snapshot_at.eq(cfg['test_snapshot']), 'target_end_exclusive'])) != {pd.Timestamp(cfg['test_end_exclusive'])}:
        raise ValueError('Неверный горизонт test')
    score = predict_risk(bundle['pipeline'], test['X'])
    _, table, ranked = campaign_metrics(test['keys'], test['y'], score, cfg['campaign_fraction'])
    _, bt, base = campaign_metrics(test['keys'], test['y'], baseline_score(test['X'], cfg), cfg['campaign_fraction'])
    for name, tab in ((cfg['selected_experiment'],table),('baseline',bt)):
        r = tab.iloc[0].to_dict()
        rows.append(dict(period='test', approach=name, **r))
    metrics = pd.DataFrame(rows)[['period','approach','snapshot_at','n_active','K','n_churn','hits','recall_at_K','precision_at_K','average_precision']]
    order = {'baseline':0,cfg['selected_experiment']:1}
    metrics = metrics.assign(_order=metrics.approach.map(order)).sort_values(['period','_order'],ascending=[False,True]).drop(columns='_order').reset_index(drop=True)
    write_csv(metrics,out/'metrics.csv')
    for name,rk in [('model',ranked),('baseline',base)]:
        pred = rk[KEY+['score','rank','selected']]
        write_csv(pred,out/f'test_{name}_predictions.csv')
        write_csv(pred[pred.selected],out/f'test_{name}_contacts.csv')
    write_csv(test['keys'].assign(churn_target=test['y']),out/'test_labels_for_evaluation.csv')

    groups = test['keys'].assign(group=test['X']['tariff_prev'].fillna('__MISSING__').to_numpy())
    segments=[]
    for name,rk in [('model',ranked),('baseline',base)]:
        st = segment_metrics(rk,groups,'tariff_prev')
        st.insert(0,'approach',name)
        segments.append(st)
    segment = pd.concat(segments,ignore_index=True)
    write_csv(segment,out/'segments_by_tariff.csv')
    checks = {'same_base':table.n_active.tolist()==bt.n_active.tolist(),
              'same_K':table.K.tolist()==bt.K.tolist(),
              'same_outcomes':table.n_churn.tolist()==bt.n_churn.tolist(),
              'campaign_limit':all(r.selected.sum()==int(t.K.iloc[0]) for r,t in ((ranked,table),(base,bt))),
              'segment_totals':all(int(p.n.sum())==len(test['y']) and int(p.hits.sum())==int(t.hits.sum()) and int(p.n_selected.sum())==int(t.K.sum()) for p,t in ((segments[0],table),(segments[1],bt))),
              'validation_matches_stage5':True,'test_used_for_selection':False,'fit_performed':False}

    # Независимая повторная загрузка тех же сохранённых весов, без обучения.
    reloaded = load_frozen(paths,cfg)
    score2 = predict_risk(reloaded['pipeline'],test['X'])
    _, table2,rk2 = campaign_metrics(test['keys'],test['y'],score2,cfg['campaign_fraction'])
    repeat = np.array_equal(score,score2) and ranked.equals(rk2) and table.equals(table2)
    checks['repeat_identical'] = bool(repeat)
    if not all(v for k,v in checks.items() if k not in ('test_used_for_selection','fit_performed')):
        raise AssertionError(checks)
    versions = dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__)
    summary = dict(selected_experiment=cfg['selected_experiment'],source_commit=cfg['source_commit'],versions=versions,
                   input_sha256=cfg['input_sha256'],checks=checks,
                   metrics=metrics.to_dict('records'),output_sha256={p.name:sha(p) for p in sorted(out.glob('*.csv'))})
    (out/'evaluation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(metrics.to_string(index=False))
    print('OK: сохранённый Pipeline, fit запрещён; общая база и K; повторная загрузка дала тот же результат.')
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path)
    args=a.parse_args()
    run(args.output)
