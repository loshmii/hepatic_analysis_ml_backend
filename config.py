from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = False

@dataclass
class TrainConfig:
    device: str = 'cpu'
    seed: int = 42
    amp: bool = True
    grad_clip_norm: Optional[float] = None
    log_every_n_steps: int = 50

@dataclass
class OptimConfig:
    name: str = 'adamw'
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

@dataclass
class SchedConfig:
    name: Optional[str] = None
    warmup_epochs: int = 0
    max_epochs: Optional[int] = None
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4

@dataclass
class MonitorConfig:
    metric: str = 'val_auprc'
    mode: str = 'max'
    min_delta: float = 0.0

@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    patience: int = 10

@dataclass
class CheckpointConfig:
    dirpath: str = 'checkpoints'
    filename: str = 'model_best.pt'
    save_last: bool = True
    save_best_only: bool = True
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedConfig = field(default_factory=SchedConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    pos_weight: Optional[float] = None
    notes: str = ''