"""Portable asset locations; no scientific parameters."""
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

def extend_offline_cfg(cfg):
    cfg.offline = CN()
    cfg.offline.enabled = False
    cfg.offline.model_dir = ''
    cfg.offline.dataset_dir = ''
    cfg.offline.metric_dir = ''
    cfg.offline.diagnostics = False

register_config('offline', extend_offline_cfg)
