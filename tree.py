from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any, Tuple, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, average_precision_score, log_loss
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold

from dataset import Spec, fit_encoders
from trainer import compute_metrics

PRETTY_NAMES = {
    "acc": "Accuracy",
    "f1": "F1 Score",
    "auroc": "ROC AUC (AUROC)",
    "auprc": "Average Precision (AUPRC)",
    "logloss": "Log Loss",
    "acc_mean": "Accuracy (mean over folds)",
    "acc_std": "Accuracy (std over folds)",
    "f1_mean": "F1 Score (mean over folds)",
    "f1_std": "F1 Score (std over folds)",
    "recall_mean": "Recall (mean over folds)",
    "recall_std": "Recall (std over folds)",
    "auroc_mean": "ROC AUC (mean over folds)",
    "auroc_std":  "ROC AUC (std over folds)",
    "auprc_mean": "Average Precision (mean over folds)",
    "auprc_std":  "Average Precision (std over folds)",
    "logloss_mean": "Log Loss (mean over folds)",
    "logloss_std":  "Log Loss (std over folds)",
    "depth": "Max Depth",
}

@dataclass
class TreeSpec:
    num_cols: Iterable[str]
    cat_cols: Iterable[str]
    target_col: str = 'class'
    drop_cols: Iterable[str] = ()

@dataclass
class TreeConfig:
    seed: int = 42
    depth_grid: Iterable[Optional[int]] = (None, 2, 3, 4, 5, 7, 9, 12, 15, 20)
    criterion_grid: Iterable[str] = ('gini', 'entropy')
    min_samples_leaf_grid: Iterable[int] = (1,2,5)
    class_weight: Optional[str] = None
    onehot_threshold: int = 2

def _proba_to_logits(p, eps: float = 1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def _build_fold_matrices(
    df: pd.DataFrame,
    spec: TreeSpec,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    onehot_threshold: int,
):
    spec_ds = Spec(spec.num_cols, spec.cat_cols, spec.target_col, drop_cols=spec.drop_cols)
    enc = fit_encoders(df.iloc[idx_tr], spec_ds)

    def _map_num(df_part: pd.DataFrame):
        Xn = df_part[spec.num_cols].astype(np.float32).values
        Xn = (Xn - enc.num_mean) / enc.num_std
        return Xn
    
    def _map_cat_block(df_part: pd.DataFrame):
        if not spec.cat_cols:
            return np.zeros((len(df_part), 0), dtype=np.float32), [], None
        mapped, cards = [], []
        for c in spec.cat_cols:
            m = enc.cat_maps[c]
            arr = df_part[c].map(m).fillna(len(m)).astype(np.int8).values
            mapped.append(arr)
            cards.append(len(m)+1)
        Xc_int = np.stack(mapped, 1)
        cols_to_oh = [j for j, card in enumerate(cards) if card > (onehot_threshold + 1)]
        return Xc_int, cols_to_oh, cards
    
    Xc_tr_int, cols_to_oh, cards = _map_cat_block(df.iloc[idx_tr])
    if cols_to_oh:
        enc_oh = OneHotEncoder(handle_unknown='ignore', sparse=False)
        Xc_sel = Xc_tr_int[:, cols_to_oh]
        enc_oh.fit(Xc_sel.reshape(len(Xc_sel), -1))
    else:
        enc_oh = None

    def _finalize(df_part: pd.DataFrame):
        Xn = _map_num(df_part)
        if spec.cat_cols:
            Xc_int, cols_to_oh_local, cards_local = _map_cat_block(df_part)
            if cols_to_oh_local != cols_to_oh:
                raise ValueError('Cardinality of categorical columns has to be the same across folds')
            if cols_to_oh:
                Xc_sel = Xc_int[:, cols_to_oh] if len(cols_to_oh) else np.zeros((len(df_part), 0), dtype=np.int8)
                Xc_oh = enc_oh.transform(Xc_sel.reshape(len(Xc_sel), -1)).astype(np.float32) if len(cols_to_oh) \
                    else np.zeros((len(df_part), 0), dtype=np.float32)
                keep = [j for j in range(len(cards_local)) if j not in cols_to_oh]
                Xc_keep = Xc_int[:,keep].astype(np.float32) if keep else np.zeros((len(df_part), 0), dtype=np.float32)
                Xc = np.concatenate([Xc_keep, Xc_oh], axis=1).astype(np.float32)
            else:
                Xc = Xc_int.astype(np.float32)
        else:
            Xc = np.zeros((len(df_part), 0), dtype=np.float32)
        X = np.concatenate([Xn, Xc], axis = 1).astype(np.float32)
        y = df_part[spec.target_col].to_numpy(np.int8)
        return X, y
    
    X_tr, y_tr = _finalize(df.iloc[idx_tr])
    X_va, y_va = _finalize(df.iloc[idx_va])
    return X_tr, y_tr, X_va, y_va

def evaluate_tree_cv(
        df: pd.DataFrame,
        spec: TreeSpec,
        cfg: TreeConfig,
        n_splits: int = 5,
        shuffle: bool = True,
):
    y = df[spec.target_col].to_numpy(np.int8)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=cfg.seed)

    rows = []
    for fold, (idx_tr, idx_va) in enumerate(skf.split(df,y),start=1):
        X_tr, y_tr, X_va, y_va = _build_fold_matrices(df, spec, idx_tr, idx_va, onehot_threshold=cfg.onehot_threshold)
        for crit in cfg.criterion_grid:
            for depth in cfg.depth_grid:
                for mleaf in cfg.min_samples_leaf_grid:
                    try:
                        clf = DecisionTreeClassifier(
                            criterion=crit,
                            max_depth=depth,
                            min_samples_leaf=mleaf,
                            class_weight=cfg.class_weight,
                            random_state=cfg.seed
                        )
                        clf.fit(X_tr, y_tr)
                        proba = clf.predict_proba(X_va)[:,1]
                    except Exception:
                        continue

                    pred = (proba >= 0.5).astype(np.int8)

                    acc = accuracy_score(y_va, pred)
                    f1 = f1_score(y_va, pred, zero_division=0)
                    auroc = roc_auc_score(y_va, proba)
                    auprc = average_precision_score(y_va, proba)
                    ll = log_loss(y_va, proba, labels=[0,1])

                    logits = _proba_to_logits(proba)
                    cm = compute_metrics(y_va, logits, policy=('default', 0))

                    rows.append(dict(
                        fold=fold,
                        depth=None if depth is None else int(depth),
                        criterion=crit,
                        min_samples_leaf=int(mleaf),
                        acc=acc, f1=f1, auroc=auroc, auprc=auprc, 
                        logloss=ll, recall=cm['recall'],
                    ))

    grid_folds = pd.DataFrame(rows)
    if grid_folds.empty:
        raise RuntimeError('No valid tree; check grid!')
    
    agg = grid_folds.groupby(['depth','criterion','min_samples_leaf']).agg(
        acc_mean=('acc','mean'), acc_std=('acc','std'), f1_mean=('f1','mean'), f1_std=('f1', 'std'),
        auroc_mean=('auroc','mean'), auroc_std=('auroc','std'), auprc_mean=('auprc','mean'),
        auprc_std=('auprc','std'), logloss_mean=('logloss','mean'), logloss_std=('logloss','std'),
        recall_mean=('recall','mean'), recall_std=('recall','std')
    ).reset_index()
    
    agg = agg.sort_values(['recall_mean', 'acc_mean', 'f1_mean'], ascending=[False, False, False]).reset_index(drop=True)
    best = agg.iloc[0].to_dict()
    return {
        'grid_folds': grid_folds,
        'grid_agg': agg,
        'best': best
    }

def plot_cv_metric_vs_depth(grid_agg: pd.DataFrame, metric: str='auprc_mean'):
    std_col = metric.replace('_mean', '_std')
    if std_col not in grid_agg.columns and f'{metric}_std' in grid_agg.columns:
        std_col = f'{metric}_std'
    plt.figure()
    for crit in grid_agg['criterion'].unique():
        sub = grid_agg[grid_agg['criterion']==crit].copy()
        sub = sub.groupby('depth').agg({metric:'mean', std_col:'mean'}).reset_index()
        sub = sub.sort_values('depth', key=lambda s: s.fillna(-1))
        x = sub['depth'].fillna(-1).to_numpy()
        y = sub[metric].to_numpy()
        plt.plot(x, y, marker='o', label=f'criterion={crit}')
        if std_col in sub.columns:
            s = sub[std_col].to_numpy()
            plt.fill_between(x, y-s, y+s, alpha=0.2)
    plt.xlabel(PRETTY_NAMES.get('depth', 'Max Depth'))
    plt.ylabel(PRETTY_NAMES.get(metric, metric))
    plt.title(f'CV {PRETTY_NAMES.get(metric, metric)} vs Max Depth')
    plt.legend(); plt.tight_layout()
    return plt.gcf()

def eval_and_plot_confusion(
        clf,
        X: np.ndarray,
        y: np.ndarray,
        *,
        threshold: float = 0.5,
        title: Optional[str] = None,
        depth: Optional[int] = None,
):
    proba = clf.predict_proba(X)[:,1]
    pred = (proba >= threshold).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0,1])
    fmt='d'

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    auroc = roc_auc_score(y, proba)
    auprc = average_precision_score(y, proba)
    ll = log_loss(y, proba, labels=[0,1])
    logits = _proba_to_logits(proba)
    tm = compute_metrics(y, logits, policy=('default', 0))

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Reds', cbar=False,
        xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'], ax=ax)
    dep = 'unlimited' if depth is None else depth
    ax.set_title(title or f'Decision Tree Confusion (depth={dep})')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout(); plt.show()

    return {
        'acc': acc, 'f1': f1, 'auroc': auroc, 'auprc': auprc, 'logloss': ll, 'threshold': tm['threshold'],
        'precision': tm['precision'], 'recall': tm['recall'], 'fpr': tm['fpr'],
    }

def refit_on_full(
        df: pd.DataFrame,
        spec: TreeSpec,
        best: Dict[str, Any],
        cfg: TreeConfig,
):
    df = df.drop(columns=[c for c in spec.drop_cols if c in df.columns], errors='ignore')
    spec_ds = Spec(spec.num_cols, spec.cat_cols, spec.target_col, drop_cols=spec.drop_cols)
    enc = fit_encoders(df, spec_ds)

    def _design(df_part: pd.DataFrame):
        Xn = df_part[spec.num_cols].astype(np.float32).values
        Xn = (Xn - enc.num_mean) / enc.num_std
        if spec.cat_cols:
            mapped, cards = [], []
            for c in spec.cat_cols:
                m = enc.cat_maps[c]
                arr = df_part[c].map(m).fillna(len(m)).astype(np.int8).values
                mapped.append(arr)
                cards.append(len(m)+1)
            Xc_int = np.stack(mapped, 1)
            cols_to_oh = [j for j, card in enumerate(cards) if card > (cfg.onehot_threshold + 1)]
            if cols_to_oh:
                enc_oh = OneHotEncoder(handle_unknown='ignore', sparse=False)
                Xc_sel = Xc_int[:, cols_to_oh]
                enc_oh.fit(Xc_sel.reshape(len(Xc_sel), -1))
                Xc_oh = enc_oh.transform(Xc_sel.reshape(len(Xc_sel), -1)).astype(np.float32)
                keep = [j for j in range(len(cards)) if j not in cols_to_oh]
                Xc_keep = Xc_int[:, keep].astype(np.float32) if keep else np.zeros((len(df_part), 0), dtype=np.float32)
                Xc = np.concatenate([Xc_keep, Xc_oh], 1).astype(np.float32)
            else:
                enc_oh = None
                Xc = Xc_int.astype(np.float32)
        else:
            enc_oh = None
            Xc = np.zeros((len(df_part), 0), dtype=np.float32)
        X = np.concatenate([Xn, Xc], 1).astype(np.float32)
        y = df_part[spec.target_col].to_numpy(np.int8)
        return X, y, enc_oh
    
    X_full, y_full, enc_oh = _design(df)
    clf = DecisionTreeClassifier(
        criterion=str(best.get('criterion','gini')),
        max_depth=None if pd.isna(best.get('depth')) else int(best.get('depth')),
        min_samples_leaf=int(best.get('min_samples_leaf', 1)),
        class_weight=cfg.class_weight,
        random_state=cfg.seed,
    )
    clf.fit(X_full, y_full)
    return clf, enc, enc_oh