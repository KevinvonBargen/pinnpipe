import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# float64 reference: in float32 the reference's rounding (~3e-8 relative) is as
# large as the error of the SSBroyden runs and would show up in the error panel
os.environ.setdefault("JAX_ENABLE_X64", "True")
from utils import get_dataset  # noqa: E402


def parse_args():
    example_dir = Path(__file__).resolve().parent
    default_run_dir = example_dir / "Soap_SSBroyden2_Mlp_64_5_2"
    default_output = example_dir / "figures" / "Soap_SSBroyden2_Mlp_64_5_2"

    parser = argparse.ArgumentParser(
        description="Plot wave reference, saved prediction, and absolute error."
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=default_run_dir / "final_predictions.npz",
        help="Path to the saved final_predictions.npz file.",
    )
    parser.add_argument(
        "--prediction-key",
        default="u",
        help="Array key to load from the prediction .npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where the figure files will be saved.",
    )
    parser.add_argument(
        "--filename",
        default="wave_prediction",
        help="Output filename without extension.",
    )
    parser.add_argument(
        "--error-percentile",
        type=float,
        default=99.5,
        help="Upper percentile used for the absolute-error color scale.",
    )
    return parser.parse_args()


def load_reference(example_dir):
    previous_cwd = Path.cwd()
    try:
        os.chdir(example_dir)
        sol, axes, _, _ = get_dataset()
    finally:
        os.chdir(previous_cwd)

    return np.asarray(sol["u"]), np.asarray(axes["t"]), np.asarray(axes["x"])


def load_prediction(prediction_file, prediction_key):
    with np.load(prediction_file) as data:
        if prediction_key not in data:
            keys = ", ".join(data.files)
            raise KeyError(
                f"Prediction key '{prediction_key}' not found in {prediction_file}. "
                f"Available keys: {keys}"
            )
        return np.asarray(data[prediction_key])


def plot_wave(reference, prediction, t, x, output_dir, filename, error_percentile):
    if reference.shape != prediction.shape:
        raise ValueError(
            "Reference and prediction shapes must match, got "
            f"{reference.shape} and {prediction.shape}."
        )

    if reference.shape != (len(t), len(x)):
        raise ValueError(
            "Expected solution shape to match axes as (len(t), len(x)), got "
            f"{reference.shape} for len(t)={len(t)} and len(x)={len(x)}."
        )

    tt, xx = np.meshgrid(t, x, indexing="ij")
    error = np.abs(reference - prediction)
    error_max = np.nanpercentile(error, error_percentile)
    if not np.isfinite(error_max) or error_max <= 0.0:
        error_max = np.nanmax(error)

    value_min = min(np.nanmin(reference), np.nanmin(prediction))
    value_max = max(np.nanmax(reference), np.nanmax(prediction))
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6), constrained_layout=True)

    reference_image = axes[0].pcolormesh(
        tt, xx, reference, cmap="BrBG", shading="auto", vmin=value_min, vmax=value_max
    )
    prediction_image = axes[1].pcolormesh(
        tt, xx, prediction, cmap="BrBG", shading="auto", vmin=value_min, vmax=value_max
    )
    error_image = axes[2].pcolormesh(
        tt, xx, error, cmap="viridis", shading="auto", vmin=0.0, vmax=error_max
    )

    panels = (
        ("Reference", axes[0]),
        ("Prediction", axes[1]),
        ("Absolute error", axes[2]),
    )

    for title, ax in panels:
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title(title)

    colorbar_kwargs = {"location": "right", "pad": 0.015, "fraction": 0.035, "aspect": 30}
    fig.colorbar(reference_image, ax=axes[:2], **colorbar_kwargs)

    error_formatter = ScalarFormatter(useMathText=True)
    error_formatter.set_powerlimits((0, 0))
    error_colorbar = fig.colorbar(
        error_image, ax=axes[2], format=error_formatter, extend="max", **colorbar_kwargs
    )
    error_colorbar.update_ticks()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename}.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return png_path


def main():
    args = parse_args()
    example_dir = Path(__file__).resolve().parent

    reference, t, x = load_reference(example_dir)
    prediction = load_prediction(args.prediction_file, args.prediction_key)
    png_path = plot_wave(
        reference,
        prediction,
        t,
        x,
        args.output_dir,
        args.filename,
        args.error_percentile,
    )

    rel_l2 = np.linalg.norm(reference - prediction) / np.linalg.norm(reference)
    print(f"Relative L2 error: {rel_l2:.6e}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
