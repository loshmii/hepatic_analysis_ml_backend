from typing import Mapping, Union, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def collect_both(model, dl, device: str = 'cpu'):
    model.eval()
    xs_raw, xs_lat, ys = [], [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        x_num, x_cat = batch['x_num'], batch['x_cat']
        y = batch['y'].view(-1)

        # raw
        xs_raw.append(torch.cat([x_num, x_cat], dim=1).cpu())

        # latent
        z_cat = model.emb(x_cat)
        h = model.mlp(torch.cat([x_num, z_cat], dim=1))
        xs_lat.append(h.cpu())

        ys.append(y.cpu())

    X_raw = torch.cat(xs_raw, dim=0).numpy().astype(np.float64)
    X_lat = torch.cat(xs_lat, dim=0).numpy().astype(np.float64)
    Y     = torch.cat(ys, dim=0).numpy().astype(int)
    return X_lat, X_raw, Y

def _safe_perp(n: int, default: int = 30):
    if n <= 3: return 5
    return max(5, min(default, (n-1) // 3))

def fit_tsne(X: np.ndarray, perplexity: Optional[int] = None, max_iter: int = 2000, seed: int = 42):
    if perplexity is None:
        perplexity = _safe_perp(len(X))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        max_iter=max_iter,
        random_state=seed
    )
    return tsne.fit_transform(X)

def tsne_comapre(
    emb_latent: np.ndarray, emb_raw: np.ndarray,
    y: np.ndarray, title: str, figsize: Tuple[float,float] = (12,5),
    perplexity: int | None = None, max_iter: int = 2000, seed: int = 42
):
    Z_lat = fit_tsne(emb_latent, perplexity=perplexity, max_iter=max_iter, seed=seed)
    Z_raw = fit_tsne(emb_latent, perplexity=perplexity, max_iter=max_iter, seed=seed)

    classes = np.unique(y).tolist()
    df_lat = pd.DataFrame({'x': Z_lat[:,0], 'y': Z_lat[:,1], 'label': y})
    df_raw = pd.DataFrame({'x': Z_raw[:,0], 'y': Z_raw[:,1], 'label': y})

    fig,axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharex=False, sharey=False)

    sns.scatterplot(
        data=df_lat, x='x', y='y', hue='label', style='label', hue_order=classes,
        s=20, alpha=0.9, edgecolor='none', ax=axes[0]
    )
    axes[0].set_title(f't-SNE of {title} (latent)')
    axes[0].set_xlabel('t-SNE x'); axes[0].set_ylabel('t-SNE y')

    sns.scatterplot(
        data=df_raw, x='x', y='y', hue='label', style='label', hue_order=classes,
        s=20, alpha=0.9, edgecolor='none', ax=axes[1]
    )
    axes[1].set_title(f't-SNE of {title} (raw)')
    axes[1].set_xlabel('t-SNE x'); axes[1].set_ylabel('t-SNE y')

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, title='Class', loc='upper right', frameon=True)

    fig.subplots_adjust(right=0.85)
    plt.tight_layout()
    return fig, np.atleast_2d(axes)

def quick_compare(
    model,
    splits: Mapping[str, object],
    device: str = 'cpu',
    perplexity: int | None = None,
    max_iter: int = 2000,
    seed: int = 42,
    figsize_per_row: Tuple[float, float] = (9,12)
):
    
    split_items = list(splits.items())
    n_rows = len(split_items)
    if n_rows == 0:
        raise ValueError('Must pass non-empty splits')
    
    all_labels = []
    cached = []
    for name, dl in split_items:
        X_lat, X_raw, y = collect_both(model, dl, device=device)
        cached.append((name, (X_lat, y), (X_raw, y)))
        all_labels.append(y)

    classes = np.unique(np.concatenate(all_labels)).tolist()
    fig_w = figsize_per_row[0]; fig_h = figsize_per_row[1]
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    handles, labels = None, None
    for r, (name, (X_lat, y), (X_raw, _)) in enumerate(cached):
        Z_lat = fit_tsne(X_lat, perplexity, max_iter, seed)
        Z_raw = fit_tsne(X_raw, perplexity, max_iter, seed)
        df_lat = pd.DataFrame({'x': Z_lat[:,0], 'y': Z_lat[:,1], 'label': y})
        df_raw = pd.DataFrame({'x': Z_raw[:,0], 'y': Z_raw[:,1], 'label': y})
        axL, axR = axes[r,0], axes[r,1]

        sns.scatterplot(
            data=df_lat, x='x', y='y', hue='label', style='label', hue_order=classes,
            s=20, alpha=0.9, edgecolor='none', ax=axL
        )
        axL.set_title(f'{name} (latent)')
        axL.set_xlabel('t-SNE x coord.'); axL.set_ylabel('t-SNE y coord.')
        sns.scatterplot(
            data=df_raw, x='x', y='y', hue='label', style='label', hue_order=classes,
            s=20, alpha=0.9, edgecolor='none', ax=axR
        )
        axR.set_title(f'{name} (raw)')
        axR.set_xlabel('t-SNE x coord.'); axR.set_ylabel('t-SNE y coord.')

        for ax in (axL, axR):
            leg = ax.get_legend()
            if leg is not None:
                if handles is None:
                    handles, labels = ax.get_legend_handles_labels()
                leg.remove()

    if handles is not None:
        fig.legend(handles, labels, title='Class', loc='upper right', frameon=True)

    fig.subplots_adjust(right=0.85, hspace=0.3, wspace=0.15)
    plt.tight_layout()
    return fig, axes
