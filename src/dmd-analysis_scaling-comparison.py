import numpy as np
import matplotlib.pyplot as plt

import global_simulation_config as config

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


def get_max_absolute_scaling(data):
    _max = data.max(axis=1, keepdims=True)
    _max = np.where(_max < epsilon, 1.0, _max)
    return (data / _max, _max)


def decompose_and_get_rmse(X):
    delay = 1
    svd_rank = compute_rank(X, svd_rank=0)

    dmd = DMD(
        svd_rank,
        exact=True,
        opt=True,
    )
    dmd = hankel_preprocessing(dmd, d=delay)
    dmd.fit(X)
    rmse = np.mean(get_rmse(X, dmd.reconstructed_data.real))
    # print(svd_rank)
    return rmse


num_orbits = 500

data_set_1 = ("data-set-1", "main_yz_20_prec_xy_0.3_add_xy5_-y-z3")
data_set_2 = ("data-set-2", "main_0.5xyz_2_prec_yz_0.5_add_-xyz0.1_-y-z0.2")
data_set_3 = ("data-set-3", "main_-0.5y-z_180_prec_-yz_1_add_-xy-z32_0.1x-0.2y0.3z20")
data_set_4 = ("data-set-4", "main_-0.5y-z_50_prec_-yz_10_add_-xy-z120_0.1x-0.2y0.3z20")

used_set_title, used_set_settings = data_set_1

data_folder = "sim_data"
data_filename = f"{used_set_title}_{num_orbits}-orbits.npy"
data_path = f"{data_folder}/{data_filename}"
data = np.load(data_path)


#############################
save = False
show = True
#############################

# remove titles
data = data[:, 1:].astype(float)

# number of sensors (rows) and time steps (columns) from data
total_num_sensors, total_num_snapshots = data.shape

epsilon = 1e-12

normal_data_rmse = []
max_abs_data_rmse = []
min_max_data_rmse = []
starting_snapshot = 10000

for sampling_interval in range(1, config.orbital_period):
    # print("------------------")
    # print(f"current interval: {sampling_interval}\n")

    number_of_snapshots = 16
    end_snapshot = (
        starting_snapshot + number_of_snapshots * sampling_interval - sampling_interval
    )
    # duration from "total_time"
    duration = data[0][end_snapshot] - data[0][starting_snapshot]

    # remove total_time
    data_subset = np.delete(data, 0, axis=0)

    # subset for decomposition
    data_subset = data_subset[
        :,
        starting_snapshot : end_snapshot + sampling_interval : (sampling_interval),
    ]

    x, t = data_subset.shape

    data_subset_min_max, _, _ = get_min_max_scaling(data_subset)
    data_subset_max_abs, _ = get_max_absolute_scaling(data_subset)

    # unscaled data
    try:
        normal_rmse = decompose_and_get_rmse(data_subset)
    except Exception:
        normal_rmse = -1
    normal_data_rmse.append((sampling_interval, normal_rmse.real))

    # min_max
    try:
        min_max_rmse = decompose_and_get_rmse(data_subset_min_max)
    except Exception:
        min_max_rmse = -1
    min_max_data_rmse.append((sampling_interval, min_max_rmse.real))

    # max_abs
    try:
        max_abs_rmse = decompose_and_get_rmse(data_subset_max_abs)
    except Exception:
        max_abs_rmse = -1
    max_abs_data_rmse.append((sampling_interval, max_abs_rmse.real))


print(f"normal min: {min(normal_data_rmse, key=lambda x: x[1])}")
print(f"min_max min: {min(min_max_data_rmse, key=lambda x: x[1])}")
print(f"max_abs min: {min(max_abs_data_rmse, key=lambda x: x[1])}")

normal_sum_rmse = np.sum(v for (_, v) in normal_data_rmse)
min_max_sum_rmse = np.sum(v for (_, v) in min_max_data_rmse)
max_abs_sum_rmse = np.sum(v for (_, v) in max_abs_data_rmse)
print(f"sum of normal rmse:{normal_sum_rmse:.8f}")
print(f"sum of min_max rmse:{min_max_sum_rmse:.8f}")
print(f"sum of max_abs rmse:{max_abs_sum_rmse:.8f}")

keys1, values1 = zip(*normal_data_rmse)
keys2, values2 = zip(*min_max_data_rmse)
keys3, values3 = zip(*max_abs_data_rmse)


figsize = (12, 8)

tick_font_size = 25
marker_size = 8
legend_font_size = 20
axes_label_font_size = 28
line_width = 2
line_alpha = 0.5

colors = ["navy", "chocolate", "magenta", "lawngreen"]
markers = ["<", ">", "^", "v"]

plots_path = "plots/scaling_comparison/"

# unscaled plot
plt.figure(figsize=figsize)
plt.plot(
    keys1,
    values1,
    color=colors[2],
    marker=markers[2],
    markersize=marker_size,
    linestyle="dotted",
    linewidth=line_width,
    label=r"\textsf{no scaling, cumulated RMSE }$\approx$" f" {normal_sum_rmse:.2f}",
)
plt.tick_params(axis="both", labelsize=tick_font_size)
plt.xlabel("Sampling interval", fontsize=axes_label_font_size)
plt.ylabel("RMSE(reconstructed data, actual data)", fontsize=axes_label_font_size)
plt.grid(True)
plt.legend(loc="upper right", fontsize=legend_font_size)
plt.tight_layout()

if save:
    plt.savefig(f"{plots_path}scaling-comparison_no-scaling.pdf", format="pdf")
if show:
    plt.show()
plt.clf()

# comparison plot
plt.figure(figsize=figsize)
plt.plot(
    keys3,
    values3,
    color=colors[0],
    marker=markers[0],
    markersize=marker_size,
    alpha=line_alpha,
    linestyle="dotted",
    linewidth=line_width,
    label=r"\textsf{maximum absolute scaling, cumulated RMSE }$\approx$"
    f" {max_abs_sum_rmse:.2f}",
)
plt.plot(
    keys2,
    values2,
    color=colors[1],
    marker=markers[1],
    markersize=marker_size,
    alpha=line_alpha,
    linestyle="dotted",
    linewidth=line_width,
    label=r"\textsf{min-max scaling, cumulated RMSE }$\approx$"
    f" {min_max_sum_rmse:.2f}",
)
plt.tick_params(axis="both", labelsize=tick_font_size)
plt.xlabel("Sampling interval", fontsize=axes_label_font_size)
plt.ylabel("RMSE(reconstructed data, actual data)", fontsize=axes_label_font_size)
plt.grid(True)
plt.legend(loc="upper right", fontsize=legend_font_size)
plt.tight_layout()
if save:
    plt.savefig(f"{plots_path}scaling-comparison_mms-mas.pdf", format="pdf")
if show:
    plt.show()
plt.clf()
