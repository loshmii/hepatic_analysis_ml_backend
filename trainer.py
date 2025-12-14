import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix
)
from callbacks import Callback, TrainerState
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Utils

def analyze_confusion(y_true, y_scores):
    y_hat = (y_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0,1]).ravel()
    return tn, fp, fn, tp

def rate_metrics(tn, fp, fn, tp):
    eps = 1e-12
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    spe = tn / max(1, (tn + fp))
    fpr = fp / max(1, (tn + fp))
    f1 = 2 * prec * rec / max(eps, (prec + rec))
    return dict(
        acc=acc,
        precision=prec,
        recall=rec,
        specificity=spe,
        fpr=fpr,
        f1=f1
    )

def compute_metrics(y_true, logits):
    y_true = np.asarray(y_true).astype(int)
    scores = 1 / (1 + np.exp(-np.asarray(logits)))

    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    
    tn, fp, fn, tp = analyze_confusion(y_true, scores)
    rates = rate_metrics(tn, fp, fn, tp)

    metrics = {
        'auprc' : float(auprc),
        'auroc': float(auroc),
        **{k: float(v) for k,v in rates.items()}
    }
    return metrics

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, callbacks=(), cfg=None,
            device='cpu', compile_model: bool = False):
        self.model, self.opt, self.crit = model, optimizer, criterion
        self.sched, self.cbs, self.cfg = scheduler, list(callbacks), cfg
        self.device = torch.device(device)
        self.state = TrainerState()

        self.amp_device = "cuda" if (self.device.type == "cuda") else "cpu"
        self.amp_dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        )

        self.scaler = torch.amp.GradScaler(device="cuda", enabled=(self.device.type == "cuda"))

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if compile_model:
            self.model = torch.compile(self.model)
        self.model.to(self.device)

    def _cb(self, hook):
        for cb in self.cbs:
            getattr(cb, hook)(self, self.state)

    def train_one_epoch(self, dl_tr):
        self.model.train()
        tr_loss_sum, n_tr = 0.0, 0 

        for step, batch in enumerate(dl_tr, start=1):
            self.state.step = step
            self.state.global_step += 1

            batch = {k: v.to(self.device, non_blocking=True) for k,v in batch.items()}
            self.opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
                logits = self.model(batch)
                loss = self.crit(logits, batch["y"])

            self.scaler.scale(loss).backward()
            clip = getattr(getattr(self.cfg, "train", None), "grad_clip_norm", None)
            if clip is not None:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            bs = batch["y"].size(0)
            tr_loss_sum += loss.item() * bs
            n_tr += bs

            self.state.metrics["train_loss"] = float(loss.item())
            self._cb("on_batch_end")
            if getattr(self.state, "stop_training", False):
                break
        
        tr_loss = tr_loss_sum / max(1, n_tr)
        self.state.metrics["train_loss"] = float(tr_loss)
        return tr_loss

    @torch.no_grad()
    def evaluate(self, dl_va):
        self.model.eval()
        val_loss_sum, n_val = 0.0, 0
        all_logits, all_y = [], []

        with torch.no_grad(), torch.amp.autocast(device_type=self.amp_device, dtype=self.amp_dtype):
            for batch in dl_va:
                batch = {k: v.to(self.device, non_blocking=True) for k,v in batch.items()}
                logits = self.model(batch)
                loss = self.crit(logits, batch["y"])

                bs = batch["y"].size(0)
                val_loss_sum += loss.item() * bs
                n_val += bs

                all_logits.append(logits.detach().float().cpu())
                all_y.append(batch["y"].detach().cpu())
        
        val_loss = val_loss_sum / max(1,n_val)
        logits_val = torch.cat(all_logits).numpy()
        y_val = torch.cat(all_y).numpy()
        val_metrics = compute_metrics(y_val, logits_val)

        self.state.metrics.update({
            'val_loss': float(val_loss),
            **{f'val_{k}': v for k,v in val_metrics.items()}
        })

        return float(val_loss), val_metrics
            
    def fit(self, dl_tr, dl_va, epochs):
        self._cb('on_fit_start')
        for epoch in range(1, epochs+1):
            self.state.epoch = epoch
            self._cb('on_epoch_start')

            # train logic
            tr_loss = self.train_one_epoch(dl_tr)
            if getattr(self.state, "stop_training", False):
                break

            val_loss, _ = self.evaluate(dl_va)

            self._cb('on_validation_end')
            self._cb('on_epoch_end')

            if self.state.stop_training: break
            if self.sched:
                if self.sched.__class__.__name__ == "ReduceLROnPlateau":
                    self.sched.step(self.state.metrics.get("val_loss", val_loss))
                else:
                    self.sched.step()
        self._cb('on_fit_end')

# Plotting util 

def eval_and_plot_confusion(model, dl, device='cpu', title='Eval'):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            all_logits.append(logits.detach().cpu())
            all_y.append(batch['y'].detach().cpu())

    logits = torch.cat(all_logits).numpy().ravel()
    y_true = torch.cat(all_y).numpy().astype(int).ravel()
    
    scores = 1.0 / (1.0 + np.exp(-logits))

    tn, fp, fn, tp = analyze_confusion(y_true, logits)
    rates = rate_metrics(tn, fp, fn, tp)
    metrics = {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        **rates
    }

    ax = sns.heatmap(
        np.array([[tn,fp],[fn,tp]]), annot=True, fmt='d', cbar=False, cmap='Reds',
        xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True1']
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title}')
    plt.tight_layout(); plt.show()

    return metrics