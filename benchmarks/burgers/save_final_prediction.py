"""Train one optimizer-sweep run and save its prediction on the reference grid
for plot_saved_prediction.py (pinnpipe's train.py does not save predictions).

    python save_final_prediction.py [--opt Soap_SSBroyden2] [--seed 2]

Writes <run name>/final_predictions.npz (key "u", shape (len(t), len(x))) and
prints the relative L2 error against the reference. W&B is off unless
WANDB_MODE is set, so the plotting run never adds a duplicate to the sweep
projects.
"""
import argparse
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # deterministic cuDNN kernels
os.environ["JAX_ENABLE_X64"] = "True"  # float64, as in optim_f64.py
os.environ.setdefault("WANDB_MODE", "disabled")

import jax
jax.config.update("jax_default_matmul_precision", "highest")
import numpy as np

import pinnpipe.train as train
from configs import optimizer_sweep

import models
from utils import get_dataset

MODEL, EVALUATOR = models.Burgers, models.BurgersEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default="Soap_SSBroyden2")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    config = optimizer_sweep.get_config(opt=args.opt, seed=args.seed)
    model, _ = train.init_and_train_and_evaluate(
        config, dataset_fn=get_dataset, model_cls=MODEL, evaluator_cls=EVALUATOR, log=True)

    sol, axes, _, _ = get_dataset()
    pred = np.asarray(model.u_pred_fn(model.state.params, axes["t"], axes["x"]))
    ref = np.asarray(sol["u"])
    run_dir = os.path.join(os.getcwd(), config.wandb.name)
    os.makedirs(run_dir, exist_ok=True)
    np.savez(os.path.join(run_dir, "final_predictions.npz"), u=pred)
    print(f"Relative L2 error: {np.linalg.norm(ref - pred) / np.linalg.norm(ref):.6e}")
    print(f"Saved {os.path.join(run_dir, 'final_predictions.npz')}")


if __name__ == "__main__":
    main()
