from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from preprocessing_template import make_preprocessor

def build_pipeline(experiment,numeric_columns,categorical_columns,seed=42):
    params=dict(experiment.get('params',{}))
    family=experiment['family']
    if family=='logistic_regression':
        options=dict(max_iter=2000,solver='lbfgs',random_state=seed);options.update(params)
        estimator=LogisticRegression(**options)
    elif family=='hist_gradient_boosting':
        if params.get('early_stopping',False) is not False:raise ValueError('Use a custom temporal early-stopping protocol instead of a random internal split')
        options=dict(early_stopping=False,random_state=seed,categorical_features=None);options.update(params)
        estimator=HistGradientBoostingClassifier(**options)
    else:raise ValueError(f'Unknown family: {family}')
    return Pipeline([('preprocess',make_preprocessor(numeric_columns,categorical_columns)),('model',estimator)])
