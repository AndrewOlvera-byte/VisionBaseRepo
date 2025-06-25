import sys
from typing import Dict

import hydra
from omegaconf import DictConfig

from train import run_train
from evaluate import run_eval
from search import run_search


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def entry(cfg: DictConfig):
    """Hydra-powered command router.

    Usage::
        python src/main.py train
        python src/main.py eval checkpoint_path=<path>
        python src/main.py search
    """
    # First CLI arg after the script selects the sub-command;
    # fallback to "train" when none is given (docker-compose). 
    cmd = "train"
    if len(sys.argv) > 1 and sys.argv[1] in {"train", "eval", "search"}:
        cmd = sys.argv[1]

    dispatch: Dict[str, callable] = {
        "train": run_train,
        "eval": run_eval,
        "search": run_search,
    }

    dispatch[cmd](cfg)


if __name__ == "__main__":
    entry() 