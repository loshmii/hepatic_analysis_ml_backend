import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Spec:
    def __init__(self, num_cols, cat_cols, target_col, drop_cols=()):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.target = target_col
        self.drop_cols = list(drop_cols)

class Encoders:
    def __init__(self, num_mean, num_std, cat_maps):
        self.num_mean = num_mean.astype(np.float32)
        self.num_std = np.where(num_std < 1e-12, 1.0, num_std).astype(np.float32)
        self.cat_maps = cat_maps

def fit_encoders(train_df, spec: Spec) -> Encoders:
    X = train_df[spec.num_cols].astype(float).values
    mean, std = X.mean(0), X.std(0)
    cat_maps = {}
    for c in spec.cat_cols:
        cats = pd.Series(train_df[c].astype('category')).cat.categories.tolist()
        cat_maps[c] = {v:i for i,v in enumerate(cats)}
    return Encoders(mean, std, cat_maps)

class TabularDS(Dataset):
    def __init__(self, df, spec: Spec, enc: Encoders):
        df = df.drop(columns=[c for c in spec.drop_cols if c in df.columns], errors='ignore').copy()
        Xn = df[spec.num_cols].astype(float).values
        Xn = ((Xn - enc.num_mean)/enc.num_std).astype(np.float32)
        self.x_num = torch.from_numpy(Xn)
        if spec.cat_cols:
            cats = []
            for c in spec.cat_cols:
                m = enc.cat_maps[c]
                cats.append(df[c].map(m).fillna(len(m)).astype(np.int64).values)
            self.x_cat = torch.from_numpy(np.stack(cats,1))
        else:
            self.x_cat = torch.empty((len(df), 0), dtype=torch.long)
        self.y = torch.from_numpy(df[spec.target].to_numpy(np.float32))
        self.idx = torch.from_numpy(df.index.to_numpy())

    def __len__(self) : return len(self.y)
    def __getitem__(self, i):
        return {'x_num': self.x_num[i], 'x_cat': self.x_cat[i], 'y': self.y[i], 'idx': self.idx[i]}

def collate(batch):
    B = len(batch)
    x_num = torch.stack([b['x_num'] for b in batch]) if batch[0]['x_num'].numel() \
        else torch.empty((B, 0), dtype=torch.float32)
    x_cat = torch.stack([b['x_cat'] for b in batch]) if batch[0]['x_cat'].numel() \
        else torch.empty((B, 0), dtype=torch.long)
    y = torch.stack([b['y'] for b in batch])
    idx = torch.stack([b['idx'] for b in batch])
    return {'x_num': x_num, 'x_cat': x_cat, 'y': y, 'idx': idx}


def main():
    df = pd.read_csv('/home/losmi/faks/som/processed_data.csv')
    spec = Spec(
        num_cols=['Age','ALB','ALP','ALT','BIL','CHE','CHOL','CREA','GGT','PROT'],
        cat_cols=['is_male'],
        target_col='class',
        drop_cols=['Unnamed: 0','Sex','Category','Category_norm']
    )

    y = df['class'].to_numpy()
    idx = np.arange(len(df))

    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx, y, test_size=0.3, stratify=y, random_state=42,
    )
    idx_va, idx_te, y_va, y_te = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42,
    )

    enc= fit_encoders(df.iloc[idx_tr], spec)

    ds_tr = TabularDS(df.iloc[idx_tr], spec, enc)
    ds_va = TabularDS(df.iloc[idx_va], spec, enc)
    ds_te = TabularDS(df.iloc[idx_te], spec, enc)

    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True, collate_fn=collate,
        num_workers=2, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, collate_fn=collate,
        num_workers=2, pin_memory=True, persistent_workers=True)
    dl_te = DataLoader(ds_te, batch_size=64, shuffle=False, collate_fn=collate,
        num_workers=2, pin_memory=True, persistent_workers=True)

    neg = (y_tr == 0).sum()
    pos = (y_tr == 1).sum()
    print(f'Negative instances {neg} - positive instances {pos}')

if __name__ == '__main__':
    main()