import numpy as np
import matplotlib.pyplot as plt
import bisect
from numpy.linalg import svd

import warnings

warnings.filterwarnings("ignore")
plt.rcParams["text.usetex"] = True


def scale_min_max(data, epsilon=1e-12):
    _min = data.min(axis=1, keepdims=True)
    _max = data.max(axis=1, keepdims=True)
    denom = _max - _min
    denom = np.where(denom < epsilon, 1.0, denom)
    return ((data - _min) / denom, _min, _max)


def descale_min_max(data, min, max):
    return data * (max - min) + min


colors = ["chocolate", "magenta", "lawngreen", "navy", "darkorange"]
markers = ["<", ">", "^", "v"]

num_orbits = 500

data_sets = [
    {
        "title": "data-set-1",
        "data": None,
        "T": [
            1,
            2,
            3,
            4,
            7,
            8,
            16,
            75,
            80,
            87,
            90,
            92,
            93,
            105,
            111,
            152,
            174,
            186,
            241,
            243,
            264,
        ],
        "n": [
            15,
            19,
            21,
            21,
            22,
            23,
            20,
            23,
            22,
            22,
            24,
            21,
            21,
            24,
            24,
            23,
            23,
            21,
            23,
            24,
            23,
        ],
        "results": [],
    },
    {
        "title": "data-set-2",
        "data": None,
        "T": [
            2,
            3,
            4,
            5,
            6,
            11,
            12,
            13,
            50,
            57,
            74,
            85,
            93,
            94,
            126,
            136,
            141,
            147,
            148,
            149,
            151,
            152,
            153,
            155,
            185,
            186,
            187,
            248,
            249,
            251,
            252,
        ],
        "n": [
            13,
            16,
            17,
            17,
            17,
            21,
            21,
            23,
            23,
            24,
            21,
            22,
            20,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            23,
            22,
            21,
            20,
            19,
            22,
            22,
            23,
            22,
        ],
        "results": [],
    },
    {
        "title": "data-set-3",
        "data": None,
        "T": [1, 2, 167, 180],
        "n": [19, 20, 22, 16],
        "results": [],
    },
    {
        "title": "data-set-4",
        "data": None,
        "T": [
            1,
            5,
            9,
            12,
            15,
            17,
            18,
            22,
            23,
            39,
            40,
            43,
            44,
            50,
            53,
            54,
            55,
            56,
            58,
            70,
            78,
            82,
            92,
            99,
            103,
            111,
            117,
            123,
            125,
            126,
            131,
            132,
            136,
            138,
            141,
            144,
            145,
            153,
            156,
            158,
            163,
            164,
            170,
            171,
            180,
            181,
            187,
            190,
            198,
            199,
            201,
            204,
            209,
            210,
            211,
            214,
            217,
            222,
            226,
            227,
            234,
            240,
            242,
            247,
            248,
            252,
            258,
            262,
            270,
            272,
            276,
            277,
            279,
        ],
        "n": [
            21,
            22,
            21,
            20,
            23,
            22,
            17,
            23,
            22,
            23,
            23,
            23,
            23,
            23,
            22,
            17,
            22,
            23,
            23,
            22,
            22,
            23,
            21,
            20,
            22,
            22,
            21,
            22,
            22,
            17,
            23,
            20,
            22,
            22,
            23,
            15,
            23,
            21,
            18,
            23,
            22,
            23,
            23,
            20,
            15,
            23,
            21,
            23,
            17,
            23,
            23,
            20,
            22,
            22,
            23,
            23,
            23,
            23,
            22,
            22,
            17,
            19,
            23,
            23,
            23,
            15,
            23,
            23,
            17,
            22,
            20,
            23,
            19,
        ],
        "results": [],
    },
]

for data_set in data_sets:
    t = len(data_set["T"])
    n = len(data_set["n"])
    assert t == n, f"{data_set['title']}, T={t}, n={n}"

data_folder = "sim_data"
filename_end = f"_{num_orbits}-orbits.npy"

epsilon = 1e-12

starting_snapshot = 500
delay = 1


#############################
save = False
show = True

scale = True

print_svd_rank_truncation = True

# set to negative to calculate all data sets
only_this_data_set_index = 3

threshold = 0.99

# delete sensors (starting from, how many)
dimensions_missing = (26, 0)
#############################

print(f"threshold={threshold}\n")

for data_set in data_sets:
    if only_this_data_set_index >= 0:
        data_set = data_sets[only_this_data_set_index]

    path = f"{data_folder}/{data_set['title']}{filename_end}"
    data = np.load(path)[:, 1:].astype(float)

    data_set_title = data_set["title"]
    print(f"loading {data_set_title} done\n")

    sampling_interval_list = data_set["T"]
    number_of_snapshots_list = data_set["n"]
    results = data_set["results"]

    print(f"\n{data_set_title}\n")

    for sampling_interval, number_of_snapshots in zip(
        sampling_interval_list, number_of_snapshots_list
    ):
        print(f"T={sampling_interval}, n={number_of_snapshots}")

        end_snapshot = (
            starting_snapshot
            + number_of_snapshots * sampling_interval
            - sampling_interval
        )

        # remove the dimensions
        start, end = dimensions_missing
        end += start
        data_subset = np.delete(data, np.s_[start:end], axis=0)

        # remove total_time
        data_subset = np.delete(data_subset, 0, axis=0)

        # subset for decomposition
        data_subset = data_subset[
            :,
            starting_snapshot : end_snapshot + sampling_interval : (sampling_interval),
        ]

        if scale:
            data_subset, data_subset_min, data_subset_max = scale_min_max(data_subset)

        singular_values = svd(data_subset, full_matrices=False, compute_uv=False)

        energies = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        indices = np.arange(len(energies), dtype=int)

        above_thresh = np.arange(len(energies)) > bisect.bisect_right(
            energies, threshold
        )
        below_thresh = ~above_thresh

        results.append(len(indices[below_thresh]))

        color_below_threshold = colors[-1]
        color_above_threshold = colors[-2]

        if save or show:
            fig_size = (12, 6.5)
            tick_font_size = 29
            label_font_size = 31
            line_width = 4

            # cumulative energy
            plt.figure(figsize=fig_size)
            plt.plot(  # dotted line
                indices,
                energies,
                alpha=0.7,
                linestyle="dotted",
                linewidth=line_width,
                label="Cumulative Energy",
            )
            plt.scatter(  # below threshold dots
                indices[below_thresh],
                energies[below_thresh],
                color=color_below_threshold,
                marker="o",
                s=line_width * 60,
                zorder=3,
            )
            plt.scatter(  # above threshold dots
                indices[above_thresh],
                energies[above_thresh],
                color=colors[3],
                marker="o",
                s=line_width * 40,
                zorder=3,
            )
            # horizontal threshold line
            plt.axhline(
                y=threshold,
                color=color_below_threshold,
                linestyle="--",
                linewidth=line_width,
            )
            plt.xticks(indices[::4])
            plt.tick_params(axis="both", labelsize=tick_font_size)
            plt.xlabel("Mode Index", fontsize=label_font_size)
            plt.ylabel("Cumulative Energy", fontsize=label_font_size)
            plt.grid(True)
            plt.tight_layout()

            if save:
                plots_path = f"plots/dmd_analysis/{data_set_title}/cumulative_energy/"
                title = "cumulative-energy"
                plt.savefig(
                    f"{plots_path}{data_set_title}_{title}_thold-{threshold}_T-{sampling_interval}_n-{number_of_snapshots}_t0-{starting_snapshot}{'' if scale else '_non-scaled'}.pdf",
                    format="pdf",
                )
            if show:
                plt.show()
            plt.clf()

            # singular values plot
            plt.figure(figsize=fig_size)
            plt.tick_params(axis="both", labelsize=tick_font_size)

            plt.semilogy(
                indices,
                singular_values,
                alpha=0.7,
                linestyle="dotted",
                linewidth=line_width,
            )
            plt.scatter(
                indices[below_thresh],
                singular_values[below_thresh],
                marker="o",
                s=line_width * 60,
                color=color_below_threshold,
                zorder=3,
            )
            plt.scatter(
                indices[above_thresh],
                singular_values[above_thresh],
                marker="o",
                s=line_width * 40,
                color=color_above_threshold,
                zorder=3,
            )
            plt.xticks(indices[::4])
            plt.xlabel("Mode Index", fontsize=label_font_size)
            plt.ylabel(
                "Singular Value (" r"$\sigma$" ") (log scale)", fontsize=label_font_size
            )
            plt.grid(True)
            plt.tight_layout()

            if save:
                plots_path = f"plots/dmd_analysis/{data_set_title}/singular_values/"
                title = "singular_values"
                plt.savefig(
                    f"{plots_path}{data_set_title}_{title}_thold-{threshold}_T-{sampling_interval}_n-{number_of_snapshots}_t0-{starting_snapshot}{'' if scale else '_non-scaled'}.pdf",
                    format="pdf",
                )
            if show:
                plt.show()
            plt.clf()

    if print_svd_rank_truncation:
        print(results)

    if only_this_data_set_index >= 0:
        break

    print(
        "__________________________________________________________________________\n\n"
    )
