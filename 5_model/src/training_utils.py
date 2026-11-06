from pathlib import Path
from decimal import Decimal, ROUND_FLOOR
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
KEY=['customer_id','snapshot_at']
FORBIDDEN={'churn_target','target','target_end_exclusive','split','customer_id','snapshot_at','termination','event_date'}

def read_csv(path):
    return pd.read_csv(path,sep=';',encoding='utf-8-sig',dtype=str,keep_default_na=False)

def check_keys(frame,name):
    if not set(KEY)<=set(frame):raise ValueError(f'{name}: missing keys')
    if frame[KEY].eq('').any().any() or frame.duplicated(KEY).any():raise ValueError(f'{name}: blank or duplicate keys')

def load_train_validation(prepared,numeric,categorical):
    prepared=Path(prepared);features=read_csv(prepared/'features.csv');labels=read_csv(prepared/'labels.csv');split=read_csv(prepared/'split.csv')
    for name,frame in [('features',features),('labels',labels),('split',split)]:check_keys(frame,name)
    keys=lambda frame:set(map(tuple,frame[KEY].itertuples(index=False,name=None)))
    if keys(features)!=keys(labels) or keys(features)!=keys(split):raise ValueError('Key sets must match without lost rows')
    if not set(split['split'])<= {'train','validation','test'}:raise ValueError('Unexpected split name')
    selected=list(numeric)+list(categorical)
    if not selected or len(set(selected))!=len(selected) or set(selected)&FORBIDDEN:raise ValueError('Use explicit disjoint lists without keys or target')
    if not set(selected)<=set(features):raise ValueError('Configured feature is absent')
    wanted=split.loc[split['split'].isin(['train','validation'])]
    chosen=features[KEY+selected].merge(wanted,on=KEY,how='inner',validate='one_to_one')
    wanted_labels=labels.merge(wanted[KEY],on=KEY,how='inner',validate='one_to_one')
    if 'churn_target' not in wanted_labels or 'target_end_exclusive' not in wanted_labels:raise ValueError('Need target and exclusive horizon end')
    chosen=chosen.merge(wanted_labels[KEY+['churn_target','target_end_exclusive']],on=KEY,how='left',validate='one_to_one')
    chosen=chosen.sort_values(KEY,kind='stable').reset_index(drop=True)
    chosen['churn_target']=pd.to_numeric(chosen.churn_target,errors='raise')
    if not chosen.churn_target.isin([0,1]).all():raise ValueError('Train/validation labels must be known and binary')
    for col in numeric:
        chosen[col]=pd.to_numeric(chosen[col].replace('',np.nan),errors='raise')
        if np.isinf(chosen[col]).any():raise ValueError(f'Infinite numeric feature: {col}')
    for col in categorical:chosen[col]=chosen[col].replace('',np.nan).astype(object)
    result={}
    for part in ['train','validation']:
        frame=chosen.loc[chosen['split'].eq(part)].copy().reset_index(drop=True)
        if frame.empty:raise ValueError(f'Empty {part} split')
        result[part]={'X':frame[selected].copy(),'y':frame.churn_target.astype(int).to_numpy(),'keys':frame[KEY].copy(),'target_end':frame.target_end_exclusive.copy()}
    train_times=pd.to_datetime(result['train']['keys'].snapshot_at,utc=True)
    validation_times=pd.to_datetime(result['validation']['keys'].snapshot_at,utc=True)
    train_ends=pd.to_datetime(result['train']['target_end'],utc=True)
    for part in ['train','validation']:
        if not (pd.to_datetime(result[part]['target_end'],utc=True)>pd.to_datetime(result[part]['keys'].snapshot_at,utc=True)).all():raise ValueError('Target horizon must end after snapshot')
    if train_times.max()>=validation_times.min() or train_ends.max()>validation_times.min():raise ValueError('Training dates or target horizons reach beyond validation start')
    if set(result['train']['y'])!={0,1}:raise ValueError('Both classes are needed in train')
    return result

def campaign_metrics(keys,y,score,fraction=.10):
    check_keys(keys,'prediction keys')
    y=np.asarray(y);score=np.asarray(score,dtype=float)
    if y.ndim!=1 or score.ndim!=1 or len(keys)!=len(y) or len(y)!=len(score):raise ValueError('Mismatched prediction arrays')
    if not np.isfinite(score).all() or not np.isin(y,[0,1]).all():raise ValueError('Scores must be finite, labels binary')
    if not 0<float(fraction)<=.10:raise ValueError('Campaign fraction must be in (0, .10]')
    frame=keys.reset_index(drop=True).copy();frame['target']=y;frame['score']=score
    frame=frame.sort_values(['snapshot_at','score','customer_id'],ascending=[True,False,True],kind='stable')
    tables=[];ranked=[]
    for t,part in frame.groupby('snapshot_at',sort=True):
        part=part.copy();n=len(part);k=int((Decimal(str(fraction))*n).to_integral_value(rounding=ROUND_FLOOR))
        part['rank']=np.arange(1,n+1);part['selected']=part['rank']<=k
        positives=int(part.target.sum());hits=int(part.loc[part.selected,'target'].sum())
        tables.append(dict(snapshot_at=t,n_active=n,K=k,n_churn=positives,hits=hits,recall_at_K=hits/positives if positives else np.nan,precision_at_K=hits/k if k else np.nan,average_precision=float(average_precision_score(part.target,part.score)) if positives else np.nan))
        ranked.append(part)
    table=pd.DataFrame(tables)
    summary=dict(macro_recall_at_K=float(table.recall_at_K.mean()) if table.recall_at_K.notna().any() else None,macro_precision_at_K=float(table.precision_at_K.mean()) if table.precision_at_K.notna().any() else None,macro_average_precision=float(table.average_precision.mean()) if table.average_precision.notna().any() else None,pooled_recall_at_K=int(table.hits.sum())/int(table.n_churn.sum()) if table.n_churn.sum() else None,n_rows=len(frame),n_dates=len(table))
    return summary,table,pd.concat(ranked,ignore_index=True)
