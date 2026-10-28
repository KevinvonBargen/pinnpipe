"""Loss-balancing study: fixed weights, GradNorm, GradNorm-MTL, ReLoBRaLo and
SoftAdapt, on the Korteweg-de Vries equation.

    python loss_balancing.py [seed ...]    # run from this directory; default: seed 2

To test a new scheme, implement a balancer in pinnpipe/loss_balancing.py,
register it in create_loss_balancer (selected by config.weighting.scheme), add a
setting for it to configs/loss_balancing_sweep.py and append its name to SCHEMES.
"""
import os
import sys

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # deterministic cuDNN kernels

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import pinnpipe.train as train
import wandb
from configs import loss_balancing_sweep

import models
from utils import get_dataset

SEEDS = [2]

# settings of configs/loss_balancing_sweep.py
SCHEMES = ["fixed", "gradnorm", "gradnorm_mtl", "relobralo", "softadapt"]


def main():
    seeds = [int(s) for s in sys.argv[1:]] or SEEDS
    for seed in seeds:
        for weighting_scheme in SCHEMES:
            config = loss_balancing_sweep.get_config(weighting_scheme=weighting_scheme, seed=seed)

            train.init_and_train_and_evaluate(
                config,
                dataset_fn=get_dataset,
                model_cls=models.KDV,
                evaluator_cls=models.KDVEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            if stats:
                print(f"Peak GPU memory usage: {stats['peak_bytes_in_use'] / 1024**3:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
