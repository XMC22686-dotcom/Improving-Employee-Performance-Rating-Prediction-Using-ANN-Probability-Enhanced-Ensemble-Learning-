
from __future__ import annotations

import argparse
import random
import time
import warnings
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from scipy.stats import friedmanchisquare, wilcoxon

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 42

TARGET_COL = "PerformanceRating"
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_DEVELOPMENT = 0.20                                 

ANN_EPOCHS = 40
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
N_WEIGHT_SEARCH = 5000
N_ESTIMATORS = 300

DROP_COLS = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18",
    "PercentSalaryHike",
    "Attrition",
]

ANN_CONFIGS = [
    {"name": "ANN-1", "hidden_layers": [64], "dropout": 0.15},
    {"name": "ANN-2", "hidden_layers": [128, 64], "dropout": 0.20},
    {"name": "ANN-3", "hidden_layers": [256, 128, 64], "dropout": 0.25},
    {"name": "ANN-4", "hidden_layers": [128, 128, 64], "dropout": 0.20},
    {"name": "ANN-5", "hidden_layers": [256, 128, 64, 32], "dropout": 0.30},
]

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def metrics_row(name: str, y_true, y_pred) -> dict:
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }

def load_dataset(path: str):
    df = pd.read_csv(path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    expected_shape = (1470, 35)
    if df.shape != expected_shape:
        print(
            f"[Warning] Dataset shape is {df.shape}, while the commonly used "
            f"IBM HR file has shape {expected_shape}."
        )

    y_raw = df[TARGET_COL].astype(int)
    observed = sorted(y_raw.unique().tolist())

    if observed != [3, 4]:
        print(
            f"[Warning] Observed PerformanceRating classes are {observed}; expected classes are 3 and 4."
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X_raw = df.drop(columns=[TARGET_COL] + drop_cols).copy()

    return df, X_raw, y, y_raw, label_encoder, drop_cols

def build_preprocessor(X_base_raw: pd.DataFrame):
    cat_cols = X_base_raw.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    num_cols = [
        c for c in X_base_raw.columns
        if c not in cat_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_onehot_encoder()),
                ]),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    return preprocessor, num_cols, cat_cols

def split_and_preprocess(X_raw: pd.DataFrame, y: np.ndarray):
    X_dev_raw, X_test_raw, y_dev, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    X_base_raw, X_val_raw, y_base, y_val = train_test_split(
        X_dev_raw,
        y_dev,
        test_size=VAL_SIZE_WITHIN_DEVELOPMENT,
        random_state=SEED,
        stratify=y_dev,
    )

    preprocessor, num_cols, cat_cols = build_preprocessor(X_base_raw)

    X_base = np.asarray(
        preprocessor.fit_transform(X_base_raw),
        dtype=np.float32,
    )
    X_val = np.asarray(
        preprocessor.transform(X_val_raw),
        dtype=np.float32,
    )
    X_test = np.asarray(
        preprocessor.transform(X_test_raw),
        dtype=np.float32,
    )
    X_dev = np.asarray(
        preprocessor.transform(X_dev_raw),
        dtype=np.float32,
    )

    feature_groups = {}
    idx = 0

    for col in num_cols:
        feature_groups[col] = [idx]
        idx += 1

    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(cat_cols, ohe.categories_):
            feature_groups[col] = list(range(idx, idx + len(cats)))
            idx += len(cats)

    data = {
        "X_dev_raw": X_dev_raw.reset_index(drop=True),
        "X_test_raw": X_test_raw.reset_index(drop=True),
        "X_base_raw": X_base_raw.reset_index(drop=True),
        "X_val_raw": X_val_raw.reset_index(drop=True),
        "y_dev": np.asarray(y_dev),
        "y_test": np.asarray(y_test),
        "y_base": np.asarray(y_base),
        "y_val": np.asarray(y_val),
        "X_dev": X_dev,
        "X_test": X_test,
        "X_base": X_base,
        "X_val": X_val,
        "preprocessor": preprocessor,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_groups": feature_groups,
    }
    return data

class ANNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: list[int],
        dropout: float,
    ):
        super().__init__()
        layers = []
        prev = input_dim

        for h in hidden_layers:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.alpha,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()

def make_criterion(y_train, num_classes, device, mode="weighted_ce"):
    counts = np.bincount(y_train, minlength=num_classes)
    weights = len(y_train) / (num_classes * counts + 1e-12)
    weights_t = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    if mode == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights_t)
    if mode == "plain_ce":
        return nn.CrossEntropyLoss()
    if mode == "focal":
        return FocalLoss(alpha=weights_t, gamma=2.0)

    raise ValueError(f"Unknown loss mode: {mode}")

def train_ann(
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    num_classes,
    device,
    loss_mode="weighted_ce",
):
    seed_everything(SEED)

    model = ANNModel(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_layers=config["hidden_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = make_criterion(
        y_train,
        num_classes,
        device,
        mode=loss_mode,
    )

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    X_val_t = torch.tensor(
        X_val, dtype=torch.float32, device=device
    )
    y_val_t = torch.tensor(
        y_val, dtype=torch.long, device=device
    )

    best_f1 = -np.inf
    best_state = None
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
    }

    for _epoch in range(ANN_EPOCHS):
        model.train()
        running = 0.0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)

        train_loss = running / len(ds)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_pred = val_logits.argmax(dim=1).cpu().numpy()

        val_f1 = f1_score(
            y_val,
            val_pred,
            average="macro",
            zero_division=0,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("ANN training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    return model, history

def predict_proba_ann(model, X, device, batch_size=4096):
    model.eval()
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    outputs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            p = F.softmax(model(xb), dim=1)
            outputs.append(p.cpu().numpy())

    return np.vstack(outputs)

def fit_ann_pool(data, num_classes, device, loss_mode="weighted_ce"):
    models = []
    histories = []
    dev_probas = []
    val_probas = []
    test_probas = []

    for config in ANN_CONFIGS:
        model, history = train_ann(
            data["X_base"],
            data["y_base"],
            data["X_val"],
            data["y_val"],
            config=config,
            num_classes=num_classes,
            device=device,
            loss_mode=loss_mode,
        )
        models.append(model)
        histories.append(history)

        dev_probas.append(
            predict_proba_ann(model, data["X_dev"], device)
        )
        val_probas.append(
            predict_proba_ann(model, data["X_val"], device)
        )
        test_probas.append(
            predict_proba_ann(model, data["X_test"], device)
        )

    return models, histories, dev_probas, val_probas, test_probas

def search_best_ann_weights(proba_list, y_true, n_iter=N_WEIGHT_SEARCH):
    rng = np.random.default_rng(SEED)
    n_models = len(proba_list)

    best_score = -np.inf
    best_weights = None

    for _ in range(n_iter):
        w = rng.dirichlet(np.ones(n_models))
        avg = np.zeros_like(proba_list[0])

        for wi, p in zip(w, proba_list):
            avg += wi * p

        pred = avg.argmax(axis=1)
        score = f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_weights = w

    return best_weights, best_score

def weighted_predict(proba_list, weights):
    avg = np.zeros_like(proba_list[0])
    for w, p in zip(weights, proba_list):
        avg += w * p
    return avg.argmax(axis=1), avg

def build_meta_models():
    return {
        "ANNProb-RF": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "ANNProb-ExtraTrees": ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "ANNProb-DT": DecisionTreeClassifier(
            random_state=SEED,
            class_weight="balanced",
        ),
        "ANNProb-KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
        "ANNProb-HistGBDT": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=SEED,
        ),
        "ANNProb-AdaBoost": AdaBoostClassifier(
            n_estimators=120,
            learning_rate=0.06,
            random_state=SEED,
        ),
        "ANNProb-Logistic": LogisticRegression(
            max_iter=3000,
            C=2.0,
            random_state=SEED,
            class_weight="balanced",
        ),
        "ANNProb-LinearSVM": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=SEED,
            class_weight="balanced",
        ),
    }

def build_classical_baselines():
    return {
        "RF": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "DT": DecisionTreeClassifier(
            random_state=SEED,
            class_weight="balanced",
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
        "HistGBDT": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=0.05,
            random_state=SEED,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=120,
            learning_rate=0.06,
            random_state=SEED,
        ),
        "Logistic": LogisticRegression(
            max_iter=3000,
            C=2.0,
            random_state=SEED,
            class_weight="balanced",
        ),
        "LinearSVM": LinearSVC(
            C=1.0,
            max_iter=5000,
            random_state=SEED,
            class_weight="balanced",
        ),
    }

def build_modern_baselines():
    missing = []
    models = {}

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=SEED,
            class_weight="balanced",
            verbosity=-1,
        )
    except Exception:
        missing.append("lightgbm")

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )
    except Exception:
        missing.append("xgboost")

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_seed=SEED,
            auto_class_weights="Balanced",
            verbose=False,
        )
    except Exception:
        missing.append("catboost")

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        models["__TabNetClass__"] = TabNetClassifier
    except Exception:
        missing.append("pytorch-tabnet")

    if missing:
        print(
            "[Warning] Optional modern-baseline packages unavailable: "
            + ", ".join(missing)
        )

    return models

def fit_tabnet(TabNetClassifier, X_train, y_train, X_test):
    model = TabNetClassifier(
        seed=SEED,
        verbose=0,
    )
    model.fit(
        X_train,
        y_train,
        max_epochs=200,
        patience=20,
        batch_size=256,
        virtual_batch_size=128,
        weights=1,
    )
    pred = model.predict(X_test).reshape(-1).astype(int)
    return model, pred

def run_main_experiment(data, num_classes, device):
    (
        ann_models,
        histories,
        dev_probas,
        val_probas,
        test_probas,
    ) = fit_ann_pool(
        data,
        num_classes=num_classes,
        device=device,
        loss_mode="weighted_ce",
    )

    results = []
    fitted = {}
    predictions = {}

    for config, model, p_test in zip(
        ANN_CONFIGS, ann_models, test_probas
    ):
        name = config["name"]
        pred = p_test.argmax(axis=1)
        results.append(metrics_row(name, data["y_test"], pred))
        fitted[name] = model
        predictions[name] = pred

    weights, best_val_f1 = search_best_ann_weights(
        val_probas,
        data["y_val"],
    )
    weighted_pred, _ = weighted_predict(test_probas, weights)
    results.append(
        metrics_row(
            "Weighted-ANN-Ensemble",
            data["y_test"],
            weighted_pred,
        )
    )
    predictions["Weighted-ANN-Ensemble"] = weighted_pred

    X_dev_stack = np.concatenate(
        [data["X_dev"]] + dev_probas,
        axis=1,
    )
    X_test_stack = np.concatenate(
        [data["X_test"]] + test_probas,
        axis=1,
    )

    meta_models = build_meta_models()
    for name, model in meta_models.items():
        model.fit(X_dev_stack, data["y_dev"])
        pred = model.predict(X_test_stack)
        results.append(metrics_row(name, data["y_test"], pred))
        fitted[name] = model
        predictions[name] = pred

    classical = build_classical_baselines()
    for name, model in classical.items():
        model.fit(data["X_dev"], data["y_dev"])
        pred = model.predict(data["X_test"])
        results.append(metrics_row(name, data["y_test"], pred))
        fitted[name] = model
        predictions[name] = pred

    modern = build_modern_baselines()
    TabNetClassifier = modern.pop("__TabNetClass__", None)

    for name, model in modern.items():
        model.fit(data["X_dev"], data["y_dev"])
        pred = model.predict(data["X_test"]).reshape(-1).astype(int)
        results.append(metrics_row(name, data["y_test"], pred))
        fitted[name] = model
        predictions[name] = pred

    if TabNetClassifier is not None:
        tabnet, pred = fit_tabnet(
            TabNetClassifier,
            data["X_dev"],
            data["y_dev"],
            data["X_test"],
        )
        results.append(metrics_row("TabNet", data["y_test"], pred))
        fitted["TabNet"] = tabnet
        predictions["TabNet"] = pred

    result_df = (
        pd.DataFrame(results)
        .sort_values("F1", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "result_df": result_df,
        "ann_models": ann_models,
        "histories": histories,
        "dev_probas": dev_probas,
        "val_probas": val_probas,
        "test_probas": test_probas,
        "weights": weights,
        "best_val_f1": best_val_f1,
        "fitted": fitted,
        "predictions": predictions,
        "X_dev_stack": X_dev_stack,
        "X_test_stack": X_test_stack,
    }

def run_sensitivity(data, main):
    rows = []
    best_rows = []

    for n_ann in range(1, len(ANN_CONFIGS) + 1):
        current = []

        w, _ = search_best_ann_weights(
            main["val_probas"][:n_ann],
            data["y_val"],
            n_iter=3000,
        )
        pred_w, _ = weighted_predict(
            main["test_probas"][:n_ann],
            w,
        )

        row = metrics_row(
            "Weighted-ANN-Ensemble",
            data["y_test"],
            pred_w,
        )
        row["ANN_Number"] = n_ann
        rows.append(row)
        current.append(row)

        X_dev_stack = np.concatenate(
            [data["X_dev"]] + main["dev_probas"][:n_ann],
            axis=1,
        )
        X_test_stack = np.concatenate(
            [data["X_test"]] + main["test_probas"][:n_ann],
            axis=1,
        )

        for name, base_model in build_meta_models().items():
            model = clone(base_model)
            model.fit(X_dev_stack, data["y_dev"])
            pred = model.predict(X_test_stack)

            row = metrics_row(name, data["y_test"], pred)
            row["ANN_Number"] = n_ann
            rows.append(row)
            current.append(row)

        current_df = pd.DataFrame(current)
        best_rows.append(
            current_df.sort_values(
                "F1", ascending=False
            ).iloc[0].to_dict()
        )

    return pd.DataFrame(rows), pd.DataFrame(best_rows)

def predict_annprob_model(
    model,
    ann_models,
    X_input,
    device,
):
    probas = [
        predict_proba_ann(m, X_input, device)
        for m in ann_models
    ]
    X_stack = np.concatenate([X_input] + probas, axis=1)
    return model.predict(X_stack)

def feature_masking_importance(
    data,
    main,
    device,
    model_name="ANNProb-AdaBoost",
):
    if model_name not in main["fitted"]:
        raise KeyError(f"{model_name} was not fitted.")

    model = main["fitted"][model_name]
    base_pred = predict_annprob_model(
        model,
        main["ann_models"],
        data["X_test"],
        device,
    )
    base_f1 = f1_score(
        data["y_test"],
        base_pred,
        average="macro",
        zero_division=0,
    )

    rows = []
    for feature, indices in data["feature_groups"].items():
        X_mask = data["X_test"].copy()
        X_mask[:, indices] = 0.0

        pred = predict_annprob_model(
            model,
            main["ann_models"],
            X_mask,
            device,
        )
        masked_f1 = f1_score(
            data["y_test"],
            pred,
            average="macro",
            zero_division=0,
        )

        rows.append({
            "Feature": feature,
            "Base_F1": base_f1,
            "F1_after_masking": masked_f1,
            "F1_Decrease": base_f1 - masked_f1,
        })

    out = pd.DataFrame(rows)
    out["Positive_Decrease"] = out["F1_Decrease"].clip(lower=0)

    vals = out["Positive_Decrease"].to_numpy()
    if len(vals) and np.any(vals > 0):
        e = np.exp(vals - vals.max())
        out["Softmax_Importance"] = e / e.sum()
    else:
        out["Softmax_Importance"] = 0.0

    return out.sort_values(
        "F1_Decrease", ascending=False
    ).reset_index(drop=True)

def _sync_device(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

def median_time(fn, repeats=30):
    fn()

    times = []
    for _ in range(repeats):
        _sync_device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.perf_counter()
        fn()
        _sync_device("cuda" if torch.cuda.is_available() else "cpu")
        times.append(time.perf_counter() - t0)

    return float(np.median(times))

def run_inference_timing(data, main, device):
    rows = []

    for config, model in zip(ANN_CONFIGS, main["ann_models"]):
        name = config["name"]
        sec = median_time(
            lambda m=model: predict_proba_ann(
                m, data["X_test"], device
            )
        )
        rows.append({"Model": name, "PredictionTime_s": sec})

    def weighted_inference():
        probas = [
            predict_proba_ann(m, data["X_test"], device)
            for m in main["ann_models"]
        ]
        weighted_predict(probas, main["weights"])

    rows.append({
        "Model": "Weighted-ANN-Ensemble",
        "PredictionTime_s": median_time(weighted_inference),
    })

    for name, model in main["fitted"].items():
        if name.startswith("ANNProb-"):
            def _fn(m=model):
                p = [
                    predict_proba_ann(
                        a, data["X_test"], device
                    )
                    for a in main["ann_models"]
                ]
                Xs = np.concatenate(
                    [data["X_test"]] + p, axis=1
                )
                m.predict(Xs)

            rows.append({
                "Model": name,
                "PredictionTime_s": median_time(_fn),
            })

    for name, model in main["fitted"].items():
        if name.startswith("ANNProb-") or name in {
            cfg["name"] for cfg in ANN_CONFIGS
        }:
            continue

        if name == "TabNet":
            fn = lambda m=model: m.predict(data["X_test"])
        else:
            fn = lambda m=model: m.predict(data["X_test"])

        try:
            sec = median_time(fn)
            rows.append(
                {"Model": name, "PredictionTime_s": sec}
            )
        except Exception:
            pass

    return (
        pd.DataFrame(rows)
        .drop_duplicates("Model")
        .sort_values("PredictionTime_s")
        .reset_index(drop=True)
    )

def _oversample(X, y, method):
    if method == "SMOTE":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=SEED).fit_resample(X, y)

    if method == "ADASYN":
        from imblearn.over_sampling import ADASYN
        return ADASYN(random_state=SEED).fit_resample(X, y)

    raise ValueError(method)

def fit_ann_pool_custom_training(
    data,
    num_classes,
    device,
    strategy,
):
    if strategy in {"SMOTE", "ADASYN"}:
        X_train, y_train = _oversample(
            data["X_base"],
            data["y_base"],
            strategy,
        )
        loss_mode = "plain_ce"
    elif strategy == "FocalLoss":
        X_train, y_train = data["X_base"], data["y_base"]
        loss_mode = "focal"
    elif strategy == "WeightedCE":
        X_train, y_train = data["X_base"], data["y_base"]
        loss_mode = "weighted_ce"
    else:
        raise ValueError(strategy)

    models = []
    dev_probas = []
    test_probas = []

    for config in ANN_CONFIGS:
        model, _ = train_ann(
            X_train,
            y_train,
            data["X_val"],
            data["y_val"],
            config=config,
            num_classes=num_classes,
            device=device,
            loss_mode=loss_mode,
        )
        models.append(model)
        dev_probas.append(
            predict_proba_ann(model, data["X_dev"], device)
        )
        test_probas.append(
            predict_proba_ann(model, data["X_test"], device)
        )

    X_dev_stack = np.concatenate(
        [data["X_dev"]] + dev_probas, axis=1
    )
    X_test_stack = np.concatenate(
        [data["X_test"]] + test_probas, axis=1
    )

    model = AdaBoostClassifier(
        n_estimators=120,
        learning_rate=0.06,
        random_state=SEED,
    )
    model.fit(X_dev_stack, data["y_dev"])
    pred = model.predict(X_test_stack)

    row = metrics_row(
        f"ANNProb-AdaBoost-{strategy}",
        data["y_test"],
        pred,
    )
    row["Strategy"] = strategy
    return row

def run_imbalance_comparison(data, num_classes, device):
    rows = []
    for strategy in [
        "WeightedCE",
        "SMOTE",
        "ADASYN",
        "FocalLoss",
    ]:
        print(f"Running imbalance strategy: {strategy}")
        rows.append(
            fit_ann_pool_custom_training(
                data,
                num_classes,
                device,
                strategy,
            )
        )
    return pd.DataFrame(rows)

def _prepare_fold_data(X_train_raw, y_train, X_test_raw, y_test):
    X_base_raw, X_val_raw, y_base, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=VAL_SIZE_WITHIN_DEVELOPMENT,
        random_state=SEED,
        stratify=y_train,
    )

    prep, num_cols, cat_cols = build_preprocessor(X_base_raw)
    X_base = np.asarray(
        prep.fit_transform(X_base_raw),
        dtype=np.float32,
    )
    X_val = np.asarray(
        prep.transform(X_val_raw),
        dtype=np.float32,
    )
    X_dev = np.asarray(
        prep.transform(X_train_raw),
        dtype=np.float32,
    )
    X_test = np.asarray(
        prep.transform(X_test_raw),
        dtype=np.float32,
    )

    return {
        "X_base": X_base,
        "y_base": np.asarray(y_base),
        "X_val": X_val,
        "y_val": np.asarray(y_val),
        "X_dev": X_dev,
        "y_dev": np.asarray(y_train),
        "X_test": X_test,
        "y_test": np.asarray(y_test),
        "feature_groups": {},
    }

def run_repeated_cv_stats(
    X_raw,
    y,
    num_classes,
    device,
):
    rkf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=SEED,
    )

    records = []

    for fold, (tr_idx, te_idx) in enumerate(
        rkf.split(X_raw, y),
        start=1,
    ):
        print(f"Repeated CV fold {fold}/10")

        X_tr_raw = X_raw.iloc[tr_idx].reset_index(drop=True)
        X_te_raw = X_raw.iloc[te_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        fd = _prepare_fold_data(
            X_tr_raw, y_tr, X_te_raw, y_te
        )

        (
            ann_models,
            _hist,
            dev_probas,
            _val_probas,
            test_probas,
        ) = fit_ann_pool(
            fd,
            num_classes=num_classes,
            device=device,
            loss_mode="weighted_ce",
        )

        X_dev_stack = np.concatenate(
            [fd["X_dev"]] + dev_probas,
            axis=1,
        )
        X_test_stack = np.concatenate(
            [fd["X_test"]] + test_probas,
            axis=1,
        )

        proposed = AdaBoostClassifier(
            n_estimators=120,
            learning_rate=0.06,
            random_state=SEED,
        )
        proposed.fit(X_dev_stack, fd["y_dev"])
        pred = proposed.predict(X_test_stack)

        records.append({
            "Fold": fold,
            "Model": "ANNProb-AdaBoost",
            "F1": f1_score(
                fd["y_test"],
                pred,
                average="macro",
                zero_division=0,
            ),
        })

        modern = build_modern_baselines()
        modern.pop("__TabNetClass__", None)

        for name in ["LightGBM", "XGBoost", "CatBoost"]:
            if name not in modern:
                continue

            m = modern[name]
            m.fit(fd["X_dev"], fd["y_dev"])
            p = m.predict(fd["X_test"]).reshape(-1).astype(int)

            records.append({
                "Fold": fold,
                "Model": name,
                "F1": f1_score(
                    fd["y_test"],
                    p,
                    average="macro",
                    zero_division=0,
                ),
            })

    cv_df = pd.DataFrame(records)

    pivot = cv_df.pivot(
        index="Fold",
        columns="Model",
        values="F1",
    ).dropna(axis=1)

    stats_rows = []

    if pivot.shape[1] >= 3:
        stat, p = friedmanchisquare(
            *[pivot[c].values for c in pivot.columns]
        )
        stats_rows.append({
            "Test": "Friedman",
            "Comparison": "All available CV models",
            "Statistic": stat,
            "p_value": p,
        })

    if "ANNProb-AdaBoost" in pivot.columns:
        for name in pivot.columns:
            if name == "ANNProb-AdaBoost":
                continue
            try:
                stat, p = wilcoxon(
                    pivot["ANNProb-AdaBoost"].values,
                    pivot[name].values,
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            except ValueError:
                stat, p = np.nan, np.nan

            stats_rows.append({
                "Test": "Wilcoxon signed-rank",
                "Comparison": f"ANNProb-AdaBoost vs {name}",
                "Statistic": stat,
                "p_value": p,
            })

    return cv_df, pd.DataFrame(stats_rows)

def main():
    parser = argparse.ArgumentParser(
        description="ANN probability-enhanced employee performance pipeline"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to WA_Fn-UseC_-HR-Employee-Attrition.csv",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Output directory",
    )
    parser.add_argument(
        "--run-cv",
        action="store_true",
        help="Run repeated 5-fold stratified CV (2 repeats) and statistical tests",
    )
    parser.add_argument(
        "--run-imbalance",
        action="store_true",
        help="Run Weighted CE / SMOTE / ADASYN / Focal Loss comparison",
    )
    args = parser.parse_args()

    seed_everything(SEED)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    (
        df,
        X_raw,
        y,
        y_raw,
        label_encoder,
        drop_cols,
    ) = load_dataset(args.data)

    data = split_and_preprocess(X_raw, y)

    num_classes = len(label_encoder.classes_)

    main_res = run_main_experiment(
        data,
        num_classes=num_classes,
        device=device,
    )
    main_res["result_df"].to_csv(
        out_dir / "all_model_performance_results.csv",
        index=False,
    )

    pd.DataFrame({
        "ANN": [c["name"] for c in ANN_CONFIGS],
        "Weight": main_res["weights"],
    }).to_csv(
        out_dir / "weighted_ann_ensemble_weights.csv",
        index=False,
    )

    cm_rows = []
    for model_name, pred in main_res["predictions"].items():
        cm = confusion_matrix(data["y_test"], pred)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm_rows.append({
                    "Model": model_name,
                    "TrueClassEncoded": i,
                    "PredClassEncoded": j,
                    "Count": int(cm[i, j]),
                })
    pd.DataFrame(cm_rows).to_csv(
        out_dir / "confusion_matrices_long_format.csv",
        index=False,
    )

    sens_all, sens_best = run_sensitivity(data, main_res)
    sens_all.to_csv(
        out_dir / "sensitivity_ann_number_all_methods.csv",
        index=False,
    )
    sens_best.to_csv(
        out_dir / "sensitivity_ann_number_best_method.csv",
        index=False,
    )

    imp = feature_masking_importance(
        data,
        main_res,
        device=device,
        model_name="ANNProb-AdaBoost",
    )
    imp.to_csv(
        out_dir / "feature_masking_importance_annprob_adaboost.csv",
        index=False,
    )

    timing = run_inference_timing(
        data,
        main_res,
        device=device,
    )
    timing.to_csv(
        out_dir / "prediction_only_inference_time.csv",
        index=False,
    )

    if args.run_imbalance:
        imbalance = run_imbalance_comparison(
            data,
            num_classes=num_classes,
            device=device,
        )
        imbalance.to_csv(
            out_dir / "imbalance_learning_comparison.csv",
            index=False,
        )

    if args.run_cv:
        cv_df, stats_df = run_repeated_cv_stats(
            X_raw,
            y,
            num_classes=num_classes,
            device=device,
        )
        cv_df.to_csv(
            out_dir / "repeated_5fold_2repeat_cv_scores.csv",
            index=False,
        )
        stats_df.to_csv(
            out_dir / "friedman_and_wilcoxon_tests.csv",
            index=False,
        )

    print("\nCompleted. Generated files are in:")
    print(out_dir.resolve())

if __name__ == "__main__":
    main()
