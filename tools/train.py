import os
from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile('local_configs/armformer_config.py')
print(cfg.train_dataloader)

cfg.work_dir = "work_dirs"
runner = Runner.from_cfg(cfg)

runner.train()

