"""Optimizer study, float64 runs: SOAP -> SSBroyden with an MLP, and SOAP with
an MLP and with a PirateNet (network/precision ablation; the float32 runs are
in optim_f32.py), on the Rayleigh-Taylor problem.

    python optim_f64.py [seed ...]    # run from this directory; default: seed 2

To test a new optimizer, implement it in pinnpipe (first-order: _create_optimizer
in pinnpipe/models.py; quasi-Newton: get_quasi_newton_opt in pinnpipe/train.py),
add settings for it to configs/optimizer_sweep.py and add a case to CASES.
"""
import os
import sys

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # deterministic cuDNN kernels
os.environ["JAX_ENABLE_X64"] = "True"  # float64; must be set before jax is imported

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import pinnpipe.train as train
import wandb
from configs import optimizer_sweep

import models
from utils import get_dataset

SEEDS = [2]

# (optimizer, network, W&B run-name suffix). Network None keeps the optimizer's
# network from configs/optimizer_sweep.py; the suffix keeps the ablation runs
# apart from the optimizer comparison in W&B.
CASES = [
    ("Soap_SSBroyden2", None, ""),  # SOAP -> SSBroyden, MLP
    ("Soap", "Mlp", "_f64"),  # SOAP, MLP (ablation)
    ("Soap", "PirateNet", "_f64"),  # SOAP, PirateNet (ablation)
]


def main():
    seeds = [int(s) for s in sys.argv[1:]] or SEEDS
    for seed in seeds:
        for opt, nn_type, suffix in CASES:
            config = optimizer_sweep.get_config(opt=opt, seed=seed, nn_type=nn_type)
            config.wandb.name += suffix

            train.train_time_windows(
                config,
                dataset_fn=get_dataset,
                model_cls=models.NavierStokes,
                evaluator_cls=models.NavierStokesEvaluator,
                log=True,
            )

            stats = jax.local_devices()[0].memory_stats()
            if stats:
                print(f"Peak GPU memory usage: {stats['peak_bytes_in_use'] / 1024**3:.2f} GiB")

            wandb.finish()


if __name__ == "__main__":
    main()
