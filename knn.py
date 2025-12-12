from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

from dataset import Spec, fit_encoders
from trainer import compute_metrics

PRETTY_NAMES = {
    "acc": "Accuracy",
    "f1": "F1 Score",
    "auroc": "ROC AUC (AUROC)",
    "auprc": "Average Precision (AUPRC)",
    "logloss": "Log Loss",
    "acc_mean": "Accuracy (mean over folds)",
    'recall_mean': 'Recall (mean over folds)',
    'recall_std': 'Recall (std over folds)',
    "acc_std":  "Accuracy (std over folds)",
    "f1_mean":  "F1 Score (mean over folds)",
    "f1_std":   "F1 Score (std over folds)",
    "auroc_mean": "ROC AUC (mean over folds)",
    "auroc_std":  "ROC AUC (std over folds)",
    "auprc_mean": "Average Precision (mean over folds)",
    "auprc_std":  "Average Precision (std over folds)",
    "logloss_mean": "Log Loss (mean over folds)",
    "logloss_std":  "Log Loss (std over folds)",
}


@dataclass
class KNNSpec:
    num_cols: Iterable[str]
    cat_cols: Iterable[str]
    target_col: str = 'class'
    drop_cols: Iterable[str] = ()

@dataclass
class KNNConfig:
    seed: int = 42
    k_grid: Iterable[int] = (1,3,5,7,9,11,15,21,31,41)
    weights: Iterable[str] = ('uniform', 'distance')
    p: int = 2
    n_jobs: Optional[int] = None
    onehot_threshold: int = 2
    
def _build_fold_matrices(
    df: pd.DataFrame,
    spec: KNNSpec,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    onehot_threshold: int
):
    spec_ds = Spec(spec.num_cols, spec.cat_cols, spec.target_col, drop_cols=spec.drop_cols)
    enc = fit_encoders(df.iloc[idx_tr], spec_ds)

    def _map_num(df_part: pd.DataFrame):
        Xn = df_part[spec.num_cols].astype(float).values
        Xn = (Xn - enc.num_mean) / enc.num_std
        return Xn.astype(np.float32)
    
    def _map_cat_block(df_part: pd.DataFrame):
        if not spec.cat_cols:
            return np.zeros((len(df_part), 0), dtype=np.float32), [], None

        mapped = []
        cardinalities = []
        for c in spec.cat_cols:
            m = enc.cat_maps[c]
            arr = df_part[c].map(m).fillna(len(m)).astype(int).values
            mapped.append(arr)
            cardinalities.append(len(m) + 1)
        Xc_int = np.stack(mapped,1)
    
        cols_to_oh = [j for j, card in enumerate(cardinalities) if card > (onehot_threshold + 1)]
        return Xc_int, cols_to_oh, cardinalities

    Xc_tr_int, cols_to_oh, cardinalities = _map_cat_block(df.iloc[idx_tr])
    if cols_to_oh:
        enc_oh = OneHotEncoder(handle_unknown='ignore', sparse=False)
        Xc_sel = Xc_tr_int[:, cols_to_oh]
        enc_oh.fit(Xc_sel.reshape(len(Xc_sel), -1))
    else:
        enc_oh = None
    
    def _finalize(df_part: pd.DataFrame):
        Xn = _map_num(df_part)
        if spec.cat_cols:
            Xc_int, cols_to_oh_local, cards = _map_cat_block(df_part)
            if cols_to_oh_local != cols_to_oh:
                raise ValueError('Cardinality of categorical columns has to be the same across folds')
            if cols_to_oh:
                Xc_sel = Xc_int[:, cols_to_oh] if len(cols_to_oh) else np.zeros((len(df_part),0),dtype=np.int64)
                Xc_oh = enc_oh.transform(Xc_sel.reshape(len(Xc_sel),-1)).astype(np.float32) if len(cols_to_oh) else \
                    np.zeros((len(df_part),0),dtype=np.float32)
                keep_cols = [j for j in range(len(cards)) if j not in cols_to_oh]
                Xc_keep = Xc_int[:, keep_cols].astype(np.float32) if keep_cols else np.zeros ((len(df_part),0),dtype=np.float32)
                Xc = np.concatenate([Xc_keep, Xc_oh], axis=1).astype(np.float32)
            else:
                Xc = Xc_int.astype(np.float32)
        else:
            Xc = np.zeros((len(df_part), 0), dtype=np.float32)
        
        X = np.concatenate([Xn, Xc], axis=1).astype(np.float32)
        y = df_part[spec.target_col].to_numpy(np.int8)
        return X, y

    X_tr, y_tr = _finalize(df.iloc[idx_tr])
    X_va, y_va = _finalize(df.iloc[idx_va])
    return X_tr, y_tr, X_va, y_va

def _proba_to_logits(p, eps: float = 1e-9):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1-p))

def evaluate_knn_cv(
    df: pd.DataFrame,
    spec: KNNSpec,
    cfg: KNNConfig,
    n_splits: int = 5,
    shuffle: bool = True,
) -> Dict[str, Any]:
    y = df[spec.target_col].to_numpy(np.int8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=cfg.seed)

    all_rows = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(df, y), start=1):
        X_tr, y_tr, X_va, y_va = _build_fold_matrices(df, spec, idx_tr, idx_va, onehot_threshold=cfg.onehot_threshold)

        for w in cfg.weights:
            for k in cfg.k_grid:
                clf = KNeighborsClassifier(n_neighbors=int(k), weights=w, p=cfg.p, n_jobs=cfg.n_jobs)
                clf.fit(X_tr, y_tr)

                proba = clf.predict_proba(X_va)[:,1]
                pred = (proba >= 0.5).astype(int)

                acc = accuracy_score(y_va, pred)
                f1 = f1_score(y_va, pred, zero_division=0)
                auroc = roc_auc_score(y_va, proba)
                auprc = average_precision_score(y_va, proba)
                ll = log_loss(y_va, proba, labels=[0,1])

                logits = _proba_to_logits(proba)
                cm = compute_metrics(y_va, logits)
                rc = cm['recall']

                row = dict(fold=fold, k=int(k), weights=w,
                    acc=acc, f1=f1, auroc=auroc, auprc=auprc, logloss=ll, recall=rc,
                    val_precision=cm['precision'],val_fpr=cm['fpr'])
                all_rows.append(row)
                
    grid_folds = pd.DataFrame(all_rows)
    agg = grid_folds.groupby(['k','weights']).agg(
        acc_mean=('acc','mean'), acc_std=('acc','std'), f1_mean=('f1','mean'), f1_std=('f1','std'),
        auroc_mean=('auroc','mean'), auroc_std=('auroc','std'), auprc_mean=('auprc','mean'), auprc_std=('auprc','std'),
        logloss_mean=('logloss','mean'), logloss_std=('logloss','std'),
        recall_mean=('recall','mean'), recall_std=('recall','std')
    ).reset_index()

    agg = agg.sort_values(['recall_mean','acc_mean','f1_mean'], ascending=[False, False, False]).reset_index(drop=True)
    best = agg.iloc[0].to_dict()
    return {'grid_folds': grid_folds, 'grid_agg': agg, 'best': best}

def plot_cv_metric_vs_k(grid_agg: pd.DataFrame, metric: str='auprc_mean'):
    std_col = metric.replace('_mean', '_std')
    if std_col not in grid_agg.columns:
        std_col = metric + '_std' if (metric + '_std') in grid_agg.columns else None
    plt.figure(figsize=(6,4))
    for w in grid_agg['weights'].unique():
        sub = grid_agg[grid_agg['weights'] == w].sort_values('k')
        k = sub['k'].to_numpy()
        m = sub[metric].to_numpy()
        plt.plot(k, m, marker='o', label=f'weights={w}')
        if std_col and std_col in sub.columns:
            s = sub[std_col].to_numpy()
            plt.fill_between(k, m-s, m+s, alpha=0.2)
    plt.xlabel('n_neighbours (k)')
    plt.ylabel(PRETTY_NAMES[metric])
    plt.title(f'{PRETTY_NAMES[metric]} as function of k')
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def _proba_to_logits(p, eps: float = 1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1-p))

def eval_and_plot_confusion(
        clf,
        X, y,
        *,
        title: Optional[str] = None,
        neigh: Optional[int] = None
):
    proba = clf.predict_proba(X)[:,1]
    pred = (proba >= 0.5).astype(np.int8)
    
    cm = confusion_matrix(y, pred, labels=[0,1])
    fmt = 'd'
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    ll = log_loss(y, proba, labels=[0, 1])
    logits = _proba_to_logits(proba)
    tm = compute_metrics(y, logits)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Reds', cbar=False,
        xticklabels=['Pred 0', 'Pred 1'],
        yticklabels=['True 0', 'True 1'],
    )
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    neigh_str = 'Number of neighbours k = ' + str(neigh) if neigh else ''
    ax.set_title(title or f'KNN Confusion --- ' + neigh_str)
    plt.tight_layout(); plt.show()

    return {
        'acc': acc, 'f1': f1, 'auroc': auroc, 'auprc': auprc, 'logloss': ll,
        'precision': tm['precision'], 'recall': tm['recall'],
        'fpr': tm['fpr']
    }
