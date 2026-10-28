import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

# float64 reference: in float32 the reference's rounding would show up in the
# error panels of the float64 runs
os.environ.setdefault("JAX_ENABLE_X64", "True")
from utils import get_dataset  # noqa: E402

# predicted fields, one row each: (key in the .npz file, colormap); diverging
# colormaps get a color scale symmetric about zero
FIELDS = (("u", "BrBG"), ("v", "BrBG"))
# field whose absolute error is shown in the last row (one of FIELDS)
ERROR_KEY = "u"


def parse_args():
    example_dir = Path(__file__).resolve().parent
    default_run_dir = example_dir / "Soap_SSBroyden2_Mlp_64_5_2"
    default_output = example_dir / "figures" / "Soap_SSBroyden2_Mlp_64_5_2"

    parser = argparse.ArgumentParser(
        description="Plot the saved Ginzburg-Landau prediction and its absolute error at selected times."
    )
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=default_run_dir / "final_predictions.npz",
        help="Path to the saved final_predictions.npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where the figure files will be saved.",
    )
    parser.add_argument(
        "--filename",
        default="ginzburg_landau_prediction",
        help="Output filename without extension.",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.0, 0.33, 0.66, 1.0],
        help="Times to plot, one column each; the nearest reference time is used.",
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

    reference = {key: np.asarray(value) for key, value in sol.items()}
    return reference, np.asarray(axes["t"]), np.asarray(axes["x"]), np.asarray(axes["y"])


def load_prediction(prediction_file, keys):
    with np.load(prediction_file) as data:
        missing = [key for key in keys if key not in data]
        if missing:
            available = ", ".join(data.files)
            raise KeyError(
                f"Prediction keys {missing} not found in {prediction_file}. "
                f"Available keys: {available}"
            )
        return {key: np.asarray(data[key]) for key in keys}


def plot_ginzburg_landau(reference, prediction, t, x, y, times, output_dir, filename, error_percentile):
    for key in prediction:
        if reference[key].shape != prediction[key].shape:
            raise ValueError(
                f"Reference and prediction shapes of '{key}' must match, got "
                f"{reference[key].shape} and {prediction[key].shape}."
            )
        if reference[key].shape != (len(t), len(x), len(y)):
            raise ValueError(
                "Expected solution shape to match axes as (len(t), len(x), len(y)), got "
                f"{reference[key].shape} for len(t)={len(t)}, len(x)={len(x)} and len(y)={len(y)}."
            )

    indices = [int(np.argmin(np.abs(t - time))) for time in times]
    xx, yy = np.meshgrid(x, y, indexing="ij")

    error = np.abs(reference[ERROR_KEY] - prediction[ERROR_KEY])[indices]
    error_max = np.nanpercentile(error, error_percentile)
    if not np.isfinite(error_max) or error_max <= 0.0:
        error_max = np.nanmax(error)

    num_rows = len(FIELDS) + 1
    fig, axes = plt.subplots(
        num_rows,
        len(indices),
        figsize=(3.1 * len(indices), 2.75 * num_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    colorbar_kwargs = {"location": "right", "pad": 0.015, "fraction": 0.035, "aspect": 15}

    for row, (key, cmap) in enumerate(FIELDS):
        values = prediction[key][indices]
        if cmap == "BrBG":
            value_max = np.nanmax(np.abs(values))
            value_min = -value_max
        else:
            value_min, value_max = np.nanmin(values), np.nanmax(values)
        for col, field in enumerate(values):
            image = axes[row, col].pcolormesh(
                xx, yy, field, cmap=cmap, shading="auto", vmin=value_min, vmax=value_max
            )
        fig.colorbar(image, ax=axes[row, :], **colorbar_kwargs)
        axes[row, 0].set_ylabel(f"{key} prediction\ny")

    for col, field in enumerate(error):
        error_image = axes[-1, col].pcolormesh(
            xx, yy, field, cmap="viridis", shading="auto", vmin=0.0, vmax=error_max
        )
    error_formatter = ScalarFormatter(useMathText=True)
    error_formatter.set_powerlimits((0, 0))
    error_colorbar = fig.colorbar(
        error_image, ax=axes[-1, :], format=error_formatter, extend="max", **colorbar_kwargs
    )
    error_colorbar.update_ticks()
    axes[-1, 0].set_ylabel(f"|{ERROR_KEY} ref - {ERROR_KEY} pred|\ny")

    for col, index in enumerate(indices):
        axes[0, col].set_title(f"t = {t[index]:.2f}")
        axes[-1, col].set_xlabel("x")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{filename}.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    return png_path


def main():
    args = parse_args()
    example_dir = Path(__file__).resolve().parent

    reference, t, x, y = load_reference(example_dir)
    prediction = load_prediction(args.prediction_file, [key for key, _ in FIELDS])
    png_path = plot_ginzburg_landau(
        reference,
        prediction,
        t,
        x,
        y,
        args.times,
        args.output_dir,
        args.filename,
        args.error_percentile,
    )

    for key in prediction:
        rel_l2 = np.linalg.norm(reference[key] - prediction[key]) / np.linalg.norm(reference[key])
        print(f"Relative L2 error {key}: {rel_l2:.6e}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
