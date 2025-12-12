import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Union

@dataclass
class TabularConfig:
    d_num: int
    cat_cardinalities: List[int] = None
    emb_dims: Union[int, List[int]] = 4
    hidden: List[int] = (128,64)
    activation: str = 'ReLU'
    dropout: float = 0.1
    use_batchnorm: bool = True
    output_dim: int = 1

def make_activation(name):
    return {'ReLU': nn.ReLU, 'GELU': nn.GELU}[name]()

class EmbeddingBlock(nn.Module):
    def __init__(self, cardinals: List[int], emb_dims: Union[int, List[int]]):
        super().__init__()
        if not cardinals:
            self.embs = nn.ModuleList()
            self.out_dim = 0
            return
        dims = ([emb_dims]*len(cardinals)) if isinstance(emb_dims, int) else emb_dims
        assert len(dims) == len(cardinals)
        self.embs = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinals, dims)])
        self.out_dim = sum(dims)
    def forward(self, x_cat):
        if not self.embs:
            return torch.empty(x_cat.size(0), 0, device=x_cat.device)
        zs = [emb(x_cat[:,j]) for j, emb in enumerate(self.embs)]
        return torch.cat(zs, dim=1)
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, act, dropout, use_bn):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)]
            if use_bn: layers += [nn.BatchNorm1d(h)]
            layers += [act, nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d
    def forward(self, x): return self.net(x)

class TabularClassifier(nn.Module):
    def __init__(self, cfg: TabularConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = EmbeddingBlock(cfg.cat_cardinalities or [], cfg.emb_dims)
        in_dim = cfg.d_num + self.emb.out_dim
        self.mlp = MLP(in_dim, list(cfg.hidden), make_activation(cfg.activation), cfg.dropout, cfg.use_batchnorm)
        self.head = nn.Linear(self.mlp.out_dim, cfg.output_dim) # logits
    def forward(self, batch):
        x = batch['x_num']
        z = self.emb(batch['x_cat'] if (self.cfg.cat_cardinalities) else torch.empty(x.size(0), 0, device=x.device))
        x = torch.cat([x, z], dim = 1) if z.numel() else x
        logits = self.head(self.mlp(x)).squeeze(-1)
        return logits