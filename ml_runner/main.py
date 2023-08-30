#!/usr/bin/env python3
import hydra
from hydra.utils import call


@hydra.main(config_path="conf")
def main(cfg):
    call(cfg.entrypoint)
