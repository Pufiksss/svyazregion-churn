"""Same top-K definition as Lesson 5; no fitting."""
from decimal import Decimal, ROUND_FLOOR
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
KEY=['customer_id','snapshot_at']

def check_keys(frame,name):
    if not set(KEY)<=set(frame) or frame[KEY].eq('').any().any() or frame.duplicated(KEY).any():
        raise ValueError(f'{name}: missing/blank/duplicate keys')

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
