#!/usr/bin/env python3
import hydra
from hydra.utils import call


@hydra.main(config_path="conf", config_name="fit")
def fit(cfg):
    call(cfg.entrypoint)
