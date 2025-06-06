import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
import statistics
from collections import defaultdict

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


def decompose_and_get_rmse(X, exact=True, opt=True):
    delay = 1
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
    {"title": "data-set-1", "data": None, "results": []},
    {"title": "data-set-2", "data": None, "results": []},
    {"title": "data-set-3", "data": None, "results": []},
    {"title": "data-set-4", "data": None, "results": []},
]

data_folder = "sim_data"
filename_end = f"_{num_orbits}-orbits.npy"

for set in data_sets:
    path = f"{data_folder}/{set['title']}{filename_end}"
    set["data"] = np.load(path)[:, 1:].astype(float)
    print(f"loading {set['title']} done")

print("\n")

epsilon = 1e-12

starting_snapshot = 500

#############################
max_sampling_frequency = config.orbital_period

exact = True
opt = True

save = False
show = True
#############################

figsize = (12, 8)

tick_font_size = 25
marker_size = 6
legend_font_size = 20
axes_label_font_size = 28
line_width = 2
line_alpha = 0.5

# comparison plot
plt.figure(figsize=figsize)

colors = ["navy", "chocolate", "magenta", "lawngreen"]
markers = ["<", ">", "^", "v"]

for i, data_set in enumerate(data_sets):
    data_set_title = data_set["title"]
    data = data_set["data"]
    results = data_set["results"]

    print(f"\n{data_set_title}\n")

    for sampling_interval in range(1, max_sampling_frequency + 1):
        number_of_snapshots = 16

        end_snapshot = (
            starting_snapshot
            + number_of_snapshots * sampling_interval
            - sampling_interval
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

        data_subset_min_max, _, _ = get_min_max_scaling(data_subset)

        try:
            min_max_rmse = decompose_and_get_rmse(data_subset_min_max, exact, opt)
        except Exception:
            min_max_rmse = -1
        results.append((sampling_interval, min_max_rmse.real))

    min_rmse_key, min_rmse_value = min(
        (x for x in results if x[1] > 0), key=lambda x: x[1]
    )
    max_rmse_key, max_rmse_value = max(results, key=lambda x: x[1])

    print(f"min rmse: key={min_rmse_key}, value={min_rmse_value}")
    print(f"max rmse: key={max_rmse_key}, value={max_rmse_value}")

    sum_rmse = sum(v for (_, v) in results)
    mean_rmse = sum_rmse / len(results)
    median_rmse = statistics.median(v for (_, v) in results)
    print("sum & mean & median rmse:")
    print(f"{sum_rmse:.4f} & {mean_rmse:.4f} & {median_rmse:.4f}")

    print("--------------")

    keys, values = zip(*results)
    trendline_rmse = Polynomial(Polynomial.fit(x=keys, y=values, deg=12).convert().coef)
    color = colors[i]
    marker = markers[i]

    # plt.plot(
    #     keys,
    #     values,
    #     marker=marker,
    #     markersize=marker_size,
    #     alpha=line_alpha,
    #     linestyle="dotted",
    #     color=color,
    # label=f"{data_set_title}, median RMSE "
    #     r"$\approx$"
    #     f" {median_rmse:.4f}",
    # )
    plt.plot(
        keys,
        trendline_rmse(keys),
        "-",
        color=color,
        label=f"Trendline {data_set_title}, median RMSE "
        r"$\approx$"
        f" {median_rmse:.4f}",
        linewidth=line_width,
    )

rmse_limits = [0.001, 0.002, 0.003]
best_sampling_interval = []

value_sums = defaultdict(float)
for set in data_sets:
    key_value_list = set["results"]
    key_list = []
    for k, v in key_value_list:
        value_sums[k] += v
        for x in rmse_limits:
            if v <= x:
                key_list.append((x, k))
                break
    best_sampling_interval.append(key_list)

for i, best_in_set in enumerate(best_sampling_interval):
    best_sampling_interval[i] = sorted(best_in_set, key=lambda x: (x[0], x[1]))

    print_best_sampling_interval = True
    if print_best_sampling_interval:
        print(f"best sampling_interval in data set {i + 1}:")
        k = 0
        for key, value in best_sampling_interval[i]:
            if key > k:
                print(f"\nrmse <= {key}:")
                k = key
            print(f"{value}", end=", ")
        print("\n--------\n")


min_key = min(value_sums.items(), key=lambda x: x[1])
print(f"sampling_interval with lowest overall RMSE: {min_key[0]} (sum: {min_key[1]})")

plt.tick_params(axis="both", labelsize=tick_font_size)
plt.xlabel("Sampling interval", fontsize=axes_label_font_size)
plt.ylabel("Trend RMSE(reconstructed data, actual data)", fontsize=axes_label_font_size)
plt.grid(True)
plt.legend(loc="upper right", fontsize=legend_font_size)
plt.tight_layout()

if save:
    plots_path = "plots/accuracy_over_sampling_frequency/"
    plt.savefig(
        f"{plots_path}acc-sampling-frequency_max-fs-{max_sampling_frequency}_"
        f"exact-{('yes' if exact else 'no')}_"
        f"opt-{('yes' if opt else 'no')}.pdf",
        format="pdf",
    )
if show:
    plt.show()
plt.clf()
