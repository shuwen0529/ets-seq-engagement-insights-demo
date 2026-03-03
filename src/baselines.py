import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def train_logreg_baseline(train_df: pd.DataFrame, seed: int = 42):
    y = train_df["outcome_label"].astype(int).values
    X = train_df.drop(columns=["outcome_label", "user_id"])

    cat = ["segment", "platform_pref"]
    num = [c for c in X.columns if c not in cat]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ]
    )

    model = Pipeline(steps=[("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(X, y)
    return model

def train_xgb_optional(train_df: pd.DataFrame, seed: int = 42):
    try:
        import xgboost as xgb
    except Exception:
        return None

    y = train_df["outcome_label"].astype(int).values
    X = train_df.drop(columns=["outcome_label", "user_id"])
    X2 = pd.get_dummies(X, columns=["segment", "platform_pref"], drop_first=True)

    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric="logloss",
    )
    clf.fit(X2, y)
    clf._train_columns = X2.columns.tolist()
    return clf
