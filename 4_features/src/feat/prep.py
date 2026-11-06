from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

FORBIDDEN = {"customer_id", "snapshot_at", "churn_target", "target", "target_end_exclusive", "split"}

NUMERIC = [
    "tenure_days", "debt_prev_rub", "payment_delay_prev_days", "traffic_prev_gb", "voice_prev_min",
    "debt_change_1m_rub", "traffic_change_1m_gb", "debt_months_3m", "pay_to_charge_ratio_3m",
    "traffic_per_active_day_3m", "tariff_changes_3m", "support_3m", "support_prev", "support_billing_3m",
    "support_network_3m", "support_open_at_T", "app_sessions_prev", "payment_attempts_prev",
    "payment_fail_rate_prev", "sessions_per_observed_day_prev", "selfservice_prev", "app_sessions_per_day_3m",
    "digital_last_obs_age_months", "digital_last_obs_sessions", "digital_observed_months_3m",
    "n_billing_months_3m", "n_traffic_known_3m", "history_months_available",
]
CATEGORICAL = ["tariff_prev", "region", "customer_segment", "acquisition_channel", "digital_status_prev"]


def _as_object(values):
    return values.astype(object)


def make_preprocessor(numeric_columns=NUMERIC, categorical_columns=CATEGORICAL) -> ColumnTransformer:
    numeric_columns, categorical_columns = list(numeric_columns), list(categorical_columns)
    cols = numeric_columns + categorical_columns
    if not cols or len(set(cols)) != len(cols):
        raise ValueError("Списки признаков должны быть непустыми и без повторов")
    if set(cols) & FORBIDDEN:
        raise ValueError("Ключи, метка, разбиение и служебные поля не являются признаками")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("as_object", FunctionTransformer(_as_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("numeric", numeric, numeric_columns),
                              ("categorical", categorical, categorical_columns)],
                             remainder="drop", verbose_feature_names_out=True)
