"""Frozen-model evaluation helpers. No estimator.fit or model selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from campaign_metrics import KEY,check_keys,campaign_metrics

def load_holdout(prepared,bundle,part,observation_end='2026-10-01T00:00:00+03:00'):
    if part not in ['validation','test']:raise ValueError('Use validation or test')
    read=lambda name:pd.read_csv(Path(prepared)/name,sep=';',encoding='utf-8-sig',dtype=str,keep_default_na=False)
    features=read('features.csv');labels=read('labels.csv');split=read('split.csv')
    for name,frame in [('features',features),('labels',labels),('split',split)]:check_keys(frame,name)
    keys=lambda frame:set(map(tuple,frame[KEY].values))
    if keys(features)!=keys(labels) or keys(features)!=keys(split):raise ValueError('Key sets differ')
    columns=bundle['numeric_columns']+bundle['categorical_columns']
    if len(set(columns))!=len(columns) or set(columns)&set(KEY+['churn_target','target_end_exclusive','split']):raise ValueError('Invalid feature schema')
    wanted=split.loc[split['split'].eq(part),KEY]
    frame=features[KEY+columns].merge(wanted,on=KEY,validate='one_to_one').merge(labels[KEY+['churn_target','target_end_exclusive']],on=KEY,validate='one_to_one').sort_values(KEY).reset_index(drop=True)
    if frame.empty:raise ValueError(f'Empty {part}')
    target=pd.to_numeric(frame.churn_target,errors='raise')
    if not target.isin([0,1]).all():raise ValueError('Unknown or invalid outcomes; do not turn them into zeros')
    # В старых таблицах даты без суффикса означают МСК, а не UTC.
    to_utc=lambda s:pd.to_datetime(s.map(lambda x:pd.Timestamp(x).tz_localize('Europe/Moscow') if pd.Timestamp(x).tzinfo is None else pd.Timestamp(x)),utc=True)
    end=to_utc(frame.target_end_exclusive);t=to_utc(frame.snapshot_at)
    if not ((end>t)&(end<=pd.Timestamp(observation_end))).all():raise ValueError('Target interval not fully observed')
    X=frame[columns].copy()
    for col in bundle['numeric_columns']:
        X[col]=pd.to_numeric(X[col].replace('',np.nan),errors='raise')
        if np.isinf(X[col]).any():raise ValueError('Infinite feature')
    for col in bundle['categorical_columns']:X[col]=X[col].replace('',np.nan).astype(object)
    return {'X':X,'y':target.to_numpy(dtype=int),'keys':frame[KEY].copy()}

def predict_risk(pipeline,X):
    index=int(np.flatnonzero(pipeline.classes_==1)[0]);score=pipeline.predict_proba(X)[:,index]
    if not np.isfinite(score).all() or ((score<0)|(score>1)).any():raise ValueError('Invalid risk score')
    return score

def segment_metrics(ranked,group_labels,group_name):
    """Use membership in GLOBAL top-K, never select a new 10% in each group."""
    groups=group_labels[KEY+['group']].copy();check_keys(groups,'segments')
    joined=ranked.merge(groups,on=KEY,how='left',validate='one_to_one')
    if joined.group.isna().any():raise ValueError('Missing segment')
    rows=[]
    for (t,g),p in joined.groupby(['snapshot_at','group'],sort=True):
        n=len(p);positive=int(p.target.sum());selected=int(p.selected.sum());hits=int((p.target.eq(1)&p.selected).sum())
        rows.append(dict(snapshot_at=t,segment=group_name,group=str(g),n=n,n_selected=selected,n_churn=positive,hits=hits,recall_at_global_K=hits/positive if positive else np.nan,precision_in_selected=hits/selected if selected else np.nan,small_group=n<30,few_positive_outcomes=positive<5))
    return pd.DataFrame(rows)

def permutation_recall(pipeline,data,fraction=.10,repeats=5,seed=42):
    """Validation only. Permute original columns within each snapshot date."""
    original=campaign_metrics(data['keys'],data['y'],predict_risk(pipeline,data['X']),fraction)[0]['macro_recall_at_K']
    if original is None:raise ValueError('Need positive validation outcomes')
    rng=np.random.default_rng(seed);rows=[]
    for col in data['X']:
        drops=[]
        for repeat in range(repeats):
            shuffled=data['X'].copy()
            for _,indices in data['keys'].groupby('snapshot_at').groups.items():
                idx=list(indices);shuffled.loc[idx,col]=rng.permutation(shuffled.loc[idx,col].to_numpy())
            value=campaign_metrics(data['keys'],data['y'],predict_risk(pipeline,shuffled),fraction)[0]['macro_recall_at_K']
            drops.append(original-value)
        rows.append(dict(feature=col,mean_recall_drop=float(np.mean(drops)),std_recall_drop=float(np.std(drops)),repeats=repeats,split='validation'))
    return pd.DataFrame(rows).sort_values('mean_recall_drop',ascending=False,kind='stable')
