"""Network study: MLP, ModifiedMLP and PirateNet, each in three sizes, on the Allen-Cahn equation.

    python nn_size.py [seed ...]    # run from this directory; default: seed 2

To test a new architecture, implement it in pinnpipe/archs.py, register its
arch_name in _create_arch (pinnpipe/models.py), add a setting for it to
configs/nn_sweep.py and append that setting's name to ARCHITECTURES.
"""
import os
import sys

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # deterministic cuDNN kernels

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import pinnpipe.train as train
import wandb
from configs import nn_sweep

import models
from utils import get_dataset

SEEDS = [2]

# settings of configs/nn_sweep.py
ARCHITECTURES = [
    "MLP_small", "MLP_med", "MLP_large",
    "ModifiedMLP_small", "ModifiedMLP_med", "ModifiedMLP_large",
    "PirateNet_small", "PirateNet_med", "PirateNet_large",
]


def main():
    seeds = [int(s) for s in sys.argv[1:]] or SEEDS
    for seed in seeds:
        for nn_arch in ARCHITECTURES:
            config = nn_sweep.get_config(nn_arch=nn_arch, seed=seed)

            train.init_and_train_and_evaluate(
                config,
                dataset_fn=get_dataset,
                model_cls=models.AllenCahn,
                evaluator_cls=models.AllenCahnEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            if stats:
                print(f"Peak GPU memory usage: {stats['peak_bytes_in_use'] / 1024**3:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
