from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
import pytest
from run_final import ROOT, run, freeze_inputs, sha, load_frozen
from evaluation_utils import load_holdout

@pytest.fixture(scope='module')
def result(tmp_path_factory):
    out=tmp_path_factory.mktemp('final')
    return run(out),out


def test_common_base_and_global_K(result):
    _,out=result
    m=pd.read_csv(out/'test_model_predictions.csv',sep=';')
    b=pd.read_csv(out/'test_baseline_predictions.csv',sep=';')
    assert set(zip(m.customer_id,m.snapshot_at))==set(zip(b.customer_id,b.snapshot_at))
    assert len(m)==1916
    assert m.selected.sum()==b.selected.sum()==191
    s=pd.read_csv(out/'segments_by_tariff.csv',sep=';')
    assert s.groupby('approach').n_selected.sum().eq(191).all()


def test_frozen_model_and_no_fit(result):
    cfg,paths=freeze_inputs()
    assert result[0]['checks']['fit_performed'] is False
    assert sha(paths['model'])==cfg['input_sha256']['model']
    assert result[0]['checks']['test_used_for_selection'] is False


def test_repeat_identical_and_saved_validation(result,tmp_path):
    second=run(tmp_path)
    assert second['output_sha256']==result[0]['output_sha256']
    assert result[0]['checks']['validation_matches_stage5']
    assert result[0]['checks']['repeat_identical']


def test_naive_moscow_horizon_at_observation_end():
    # Сентябрь заканчивается ровно в 00:00 МСК 1 октября; это не UTC.
    cfg,paths=freeze_inputs()
    b=load_frozen(paths,cfg)
    d=load_holdout(paths['features'].parent,b,'test','2026-10-01T00:00:00+03:00')
    assert len(d['y'])==1916
    with pytest.raises(ValueError,match='fully observed'):
        load_holdout(paths['features'].parent,b,'test','2026-09-30T23:59:59+03:00')
