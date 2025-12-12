import math, os, json
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import pandas as pd


@dataclass
class TrainerState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    best: Dict[str, float] = field(default_factory=dict)
    stop_training: bool = False

class Callback:
    def on_fit_start(self, trainer, state: TrainerState): pass
    def on_epoch_start(self, trainer, state: TrainerState): pass
    def on_batch_end(self, trainer, state: TrainerState): pass
    def on_validation_end(self, trainer, state: TrainerState): pass
    def on_epoch_end(self, trainer, state: TrainerState): pass
    def on_fit_end(self, trainer, state: TrainerState): pass
    def on_exception(self, trainer, state: TrainerState, exc: BaseException): pass

# Utils
def _is_better(curr: float, best: Optional[float], mode: str, min_delta: float) -> bool:
    if best is None or math.isnan(best):
        return True
    if mode == 'max':
        return curr >= (best + min_delta)
    elif mode == 'min':
        return curr <= (best - min_delta)
    raise ValueError('mode must be min or max')

# Callbacks

class MetricLogger(Callback):
    def __init__(self, log_every_n_steps: int = 50):
        self.n = log_every_n_steps
    def on_batch_end(self, trainer, state):
        if state.global_step % self.n == 0:
            msg = f'epoch {state.epoch} step {state.step} | '
            if 'train_loss' in state.metrics:
                msg += f"train_loss={state.metrics['train_loss']:.4f} "
            print(msg)
    def on_epoch_end(self, trainer, state: TrainerState):
        print(f'[epoch {state.epoch}] ' + ' '.join(f'{k}={v:.4f}' for k,v in state.metrics.items()))

class History(Callback):
    def __init__(self):
        self.rows = []
    def on_epoch_end(self, trainer, state: TrainerState):
        row = {'epoch': state.epoch, **state.metrics}
        self.rows.append({k: float(v) if isinstance(v, (int, float)) else v for k, v in row.items()})
    def to_frame(self): return pd.DataFrame(self.rows)

class TerminateOnNaN(Callback):
    def on_batch_end(self, trainer, state: TrainerState):
        for k,v in state.metrics.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                print(f'TerminateOnNaN: {k} became {v}')
                state.stop_training = True

class EarlyStopping(Callback):
    def __init__(self, metric: str, mode: str = 'max', patience: int = 10, min_delta: float = 0.0):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._best = None
        self._wait = 0
    def on_epoch_end(self, trainer, state: TrainerState):
        if self.metric not in state.metrics: return
        curr = state.metrics[self.metric]
        if _is_better(curr, self._best, self.mode, self.min_delta):
            self._best = curr
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                print(f'EarlyStopping: no improvement in {self.patience} epochs. Best {self.metric}={self._best:.4f}')
                state.stop_training = True

class ModelCheckpoint(Callback):
    def __init__(self, dirpath: str,filename: str, monitor: str, mode: str = 'max',
        save_best_only: bool = True, save_last: bool = True):
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self._best = None
        os.makedirs(dirpath, exist_ok=True)
    def _save(self, trainer, path: str, extra: Optional[dict] = None):
        payload = {
            'epoch': trainer.state.epoch,
            'global_step': trainer.state.global_step,
            'config': getattr(trainer, 'cfg', None),
            'metrics': trainer.state.metrics,
        }
        if extra: payload.update(extra)
        torch.save(trainer.model.state_dict(), path + '.pt')
        with open(path + '.json', 'w') as f:
            json.dump(payload, f)
    def on_epoch_end(self, trainer, state: TrainerState):
        if self.save_last:
            self._save(trainer, os.path.join(self.dirpath, 'model_last'))
        if self.monitor in state.metrics:
            curr = state.metrics[self.monitor]
            if _is_better(curr, self._best, self.mode, 0.0):
                self._best = curr
                if self.save_best_only:
                    self._save(trainer, os.path.join(self.dirpath, self.filename),
                        extra={'best_metric': {self.monitor: curr}})
            elif not self.save_best_only:
                self._save(trainer, os.path.join(self.dirpath, self.filename))

class LRSchedulerCallback(Callback):
    def __init__(self, scheduler, step_on: str = 'epoch'): # 'batch' or 'epoch'
        self.scheduler = scheduler
        self.step_on = step_on
    def on_batch_end(self, trainer, state: TrainerState):
        if self.step_on == 'batch' and self.scheduler:
            self.scheduler.step()
    def on_epoch_end(self, trainer, state: TrainerState):
        if self.step_on == 'epoch' and self.scheduler:
            if hasattr(self.scheduler, 'step') and self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                val_loss = state.metrics.get('val_loss')
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()