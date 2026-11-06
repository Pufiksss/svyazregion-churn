import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from training_utils import load_train_validation,campaign_metrics,KEY

def check(model_path,prepared,out,expected=None):
    bundle=joblib.load(model_path)
    data=load_train_validation(prepared,bundle['numeric_columns'],bundle['categorical_columns'])['validation']
    pipeline=bundle['pipeline'];positive=int(np.flatnonzero(pipeline.classes_==1)[0]);score=pipeline.predict_proba(data['X'])[:,positive]
    _,_,ranked=campaign_metrics(data['keys'],np.zeros(len(score),dtype=int),score,bundle['config']['campaign_fraction'])
    result=ranked[KEY+['score','rank','selected']].copy()
    if expected is not None:
        old=pd.read_csv(expected,sep=';',encoding='utf-8-sig',dtype={'customer_id':str,'snapshot_at':str})
        if old.duplicated(KEY).any() or set(map(tuple,old[KEY].values))!=set(map(tuple,result[KEY].values)):raise ValueError('Expected prediction keys differ')
        paired=result.merge(old[KEY+['score']],on=KEY,validate='one_to_one',suffixes=('_new','_old'))
        if not np.allclose(paired.score_new,paired.score_old,atol=1e-12,rtol=1e-12):raise ValueError('Reloaded scores differ')
        if {'rank','selected'}<=set(old):
            order=result.merge(old[KEY+['rank','selected']],on=KEY,validate='one_to_one',suffixes=('_new','_old'))
            if not (order.rank_new.astype(int)==order.rank_old.astype(int)).all():raise ValueError('Reloaded ranks differ')
            if not (order.selected_new.astype(str).str.lower()==order.selected_old.astype(str).str.lower()).all():raise ValueError('Reloaded top-K differs')
    out.parent.mkdir(parents=True,exist_ok=True);result.to_csv(out,sep=';',encoding='utf-8-sig',index=False)
    print(f'Reloaded without fit; validation scores: {len(result)}; expected match: {expected is not None}')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected',type=Path);a=p.parse_args();check(a.model,a.prepared,a.out,a.expected)
