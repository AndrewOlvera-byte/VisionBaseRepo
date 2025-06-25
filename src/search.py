from typing import Dict

import hydra
from omegaconf import DictConfig
from ray import air, tune

from train import run_train


def run_search(cfg: DictConfig):
    """Launch a Ray Tune hyper-parameter sweep."""
    search_cfg: Dict = cfg.get("search", {})

    param_space = search_cfg.get("param_space", {})
    num_samples = search_cfg.get("num_samples", 10)
    metric = search_cfg.get("metric", "val_accuracy")
    mode = search_cfg.get("mode", "max")

    tuner = tune.Tuner(
        tune.with_parameters(run_train, cfg=cfg),
        param_space=param_space,
        tune_config=tune.TuneConfig(metric=metric, mode=mode, num_samples=num_samples),
        run_config=air.RunConfig(name="ray_tune", local_dir="multirun"),
    )

    tuner.fit() 