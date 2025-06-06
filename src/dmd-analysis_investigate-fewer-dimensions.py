import numpy as np
import matplotlib.pyplot as plt
import statistics

from pydmd.utils import compute_rank
from pydmd import DMD
from pydmd.preprocessing import hankel_preprocessing

import warnings

warnings.filterwarnings("ignore")
plt.rcParams["text.usetex"] = True


def get_rmse(X_true, X_pred):
    return np.sqrt(np.mean((X_true - X_pred) ** 2, axis=1))


def get_min_max_scaling(data):
    _min = data.min(axis=1, keepdims=True)
    _max = data.max(axis=1, keepdims=True)
    denom = _max - _min
    denom = np.where(denom < epsilon, 1.0, denom)
    return ((data - _min) / denom, _max, _min)


def decompose_and_get_rmse(X, exact=True, opt=True):
    svd_rank = compute_rank(X, svd_rank=0)

    dmd = DMD(
        svd_rank,
        exact=exact,
        opt=opt,
    )
    dmd = hankel_preprocessing(dmd, d=delay)
    dmd.fit(X)
    return np.mean(get_rmse(X, dmd.reconstructed_data.real))


num_orbits = 500

data_sets = [
    ["data-set-1", [], 1],
    ["data-set-2", [], 2],
    ["data-set-3", [], 1],
    ["data-set-4", [], 18],
]

data_folder = "sim_data"

epsilon = 1e-12

starting_snapshot = 500

#############################
exact = True
opt = True
delay = 1

save = False
show = True
#############################

figsize = (12, 8.5)

tick_font_size = 25
marker_size = 8
legend_font_size = 20
axes_label_font_size = 28
line_width = 2
line_alpha = 0.7

colors = ["navy", "chocolate", "magenta", "lawngreen"]
markers = ["<", ">", "^", "v"]

max_snapshot_count = 100

dimension_delete_start_index = 10
dimension_delete_max = 3

for deleted_current in range(dimension_delete_max + 1):
    plt.figure(figsize=figsize)
    for i, set in enumerate(data_sets):
        set[1] = []
        filename = f"{set[0]}_{num_orbits}-orbits.npy"
        path = f"{data_folder}/{filename}"
        data = np.load(path)
        data = data[:, 1:].astype(float)

        for number_of_snapshots in range(max((delay + 1), 3), max_snapshot_count + 1):
            sampling_interval = set[2]
            end_snapshot = (
                starting_snapshot
                + number_of_snapshots * sampling_interval
                - sampling_interval
            )
            # duration from "total_time"
            duration = data[0][end_snapshot] - data[0][starting_snapshot]

            # remove the dimensions
            start = dimension_delete_start_index
            end = start + deleted_current
            data_subset = np.delete(data, np.s_[start:end], axis=0)

            # remove total_time
            data_subset = np.delete(data_subset, 0, axis=0)

            # subset for decomposition
            data_subset = data_subset[
                :,
                starting_snapshot : end_snapshot + sampling_interval : (
                    sampling_interval
                ),
            ]

            x, t = data_subset.shape

            data_subset_min_max, _, _ = get_min_max_scaling(data_subset)

            try:
                min_max_rmse = decompose_and_get_rmse(data_subset_min_max, exact, opt)
            except Exception:
                min_max_rmse = -1
            set[1].append((number_of_snapshots, min_max_rmse.real))

        print(f"\n{set[0]}")
        min_rmse_key, min_rmse_value = min(set[1], key=lambda x: x[1])
        max_rmse_key, max_rmse_value = max(set[1], key=lambda x: x[1])

        print(f"min rmse: key={min_rmse_key}, value={min_rmse_value}")
        print(f"max rmse: key={max_rmse_key}, value={max_rmse_value}")

        sum_rmse = sum(v for (_, v) in set[1])
        mean_rmse = sum_rmse / len(set[1])
        median_rmse = statistics.median(v for (_, v) in set[1])
        print("sum & mean & median rmse:")
        print(f"{sum_rmse:.4f} & {mean_rmse:.4f} & {median_rmse:.4f}")

        print("--------------")

        keys, values = zip(*set[1])
        color = colors[i]
        marker = markers[i]

        plt.plot(
            keys,
            values,
            marker=marker,
            markersize=marker_size,
            alpha=line_alpha,
            linestyle="dotted",
            linewidth=line_width,
            color=color,
            label=f"{set[0]}, median RMSE "
            r"$\approx$"
            f" {median_rmse:.4f}",
        )

    plt.tick_params(axis="both", labelsize=tick_font_size)
    plt.xlabel("Number of Snapshots", fontsize=axes_label_font_size)
    plt.ylabel("RMSE(reconstructed data, actual data)", fontsize=axes_label_font_size)
    plt.grid(True)
    plt.legend(loc="upper right", fontsize=legend_font_size)
    plt.tight_layout()

    if save:
        plots_path = "plots/investigate_fewer_dimension/"
        plt.savefig(
            f"{plots_path}remove-dimensions_{deleted_current}-fewer-dimensions_from-{dimension_delete_start_index}_"
            f"delay-{delay}"
            f"{('_exact' if exact else '')}"
            f"{('_opt' if opt else '')}.pdf",
            format="pdf",
        )
    if show:
        plt.show()
    plt.clf()
