"""
Lightweight experiment tracking (PR8).

Provides simple logging + optional TensorBoard writer.
Avoids heavy deps like MLflow/W&B for this research prototype.
"""

import os
import logging
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def get_logger(name: str = "chord_trainer"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class ExperimentTracker:
    """Simple tracker that writes run json (already done in train) + optional TB + CSV."""

    def __init__(self, log_dir: str = "runs", use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = get_logger()
        self.writer = None
        if use_tensorboard and HAS_TB:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
            self.logger.info(f"TensorBoard logging to {log_dir}/{run_name}")
        elif use_tensorboard and not HAS_TB:
            self.logger.warning("TensorBoard requested but torch.utils.tensorboard not available. Install torch[tensorboard] or use pip install tensorboard.")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer:
            self.writer.add_scalar(tag, value, step)
        # Always log to python logger at INFO for basic visibility
        self.logger.info(f"[{step}] {tag}: {value:.4f}")

    def log_hparams(self, hparams: dict, metrics: dict = None):
        if self.writer and metrics:
            self.writer.add_hparams(hparams, metrics)
        self.logger.info(f"HParams: {hparams}")

    def close(self):
        if self.writer:
            self.writer.close()
