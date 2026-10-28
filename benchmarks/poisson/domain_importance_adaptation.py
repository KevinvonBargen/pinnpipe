"""Domain importance adaptation study: the collocation samplers uniform, R3,
RAD, LAS and vRBA, on the 5-D Poisson equation.

    python domain_importance_adaptation.py [seed ...]    # run from this directory; default: seed 2

To test a new sampler, implement it in pinnpipe/domain_importance_adapt.py,
register it in get_sampler (selected by config.dia.type), add a setting for it
to configs/sampler_sweep.py and append its name to SAMPLERS.
"""
import os
import sys

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # deterministic cuDNN kernels

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import pinnpipe.train as train
import wandb
from configs import sampler_sweep

import models
from utils import get_dataset

SEEDS = [2]
DIM = 5  # spatial dimension

# settings of configs/sampler_sweep.py
SAMPLERS = ["uniform", "R3", "rad", "las", "vRBA"]


def main():
    seeds = [int(s) for s in sys.argv[1:]] or SEEDS
    for seed in seeds:
        for dia_method in SAMPLERS:
            config = sampler_sweep.get_config(dia_method=dia_method, seed=seed, dim=DIM)

            train.init_and_train_and_evaluate(
                config,
                dataset_fn=lambda: get_dataset(dim=DIM),
                model_cls=models.PoissonND,
                evaluator_cls=models.PoissonNDEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            if stats:
                print(f"Peak GPU memory usage: {stats['peak_bytes_in_use'] / 1024**3:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
