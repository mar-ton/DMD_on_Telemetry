import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from pydmd.utils import compute_rank
from pydmd import DMD
from pydmd.preprocessing import hankel_preprocessing

import warnings

warnings.filterwarnings("ignore")
plt.rcParams["text.usetex"] = True


def get_rmse(X_actual, X_predicted):
    return np.sqrt(np.mean((X_actual - X_predicted) ** 2, axis=1))


def scale_min_max(data, epsilon=1e-12):
    _min = data.min(axis=1, keepdims=True)
    _max = data.max(axis=1, keepdims=True)
    denom = _max - _min
    denom = np.where(denom < epsilon, 1.0, denom)
    return ((data - _min) / denom, _min, _max)


def descale_min_max(data, min, max):
    return data * (max - min) + min


def compute_dmd(
    X,
    Y=None,
    svd_rank=0,
    tlsq_rank=0,
    exact=True,
    opt=True,
    pre_proc_delay=1,
):
    if svd_rank == 0:
        svd_rank = compute_rank(X, svd_rank=0)

    # print(f"svd={svd_rank}")
    # print(f"tlsq={tlsq_rank}")
    # print(f"exact?={exact}")
    # print(f"opt?={opt}")
    # print(f"delay={pre_proc_delay}")

    dmd = DMD(
        svd_rank=svd_rank,
        tlsq_rank=tlsq_rank,
        exact=exact,
        opt=opt,
    )
    dmd = hankel_preprocessing(dmd, d=pre_proc_delay)
    dmd.fit(X, Y)

    return dmd, svd_rank, tlsq_rank


colors = ["navy", "chocolate", "magenta", "lawngreen"]
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
        "svd": [4, 5, 6, 6, 7, 8, 6, 9, 8, 7, 9, 8, 8, 9, 9, 8, 8, 8, 9, 9, 9],
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
        "svd": [
            3,
            4,
            4,
            4,
            5,
            7,
            7,
            8,
            9,
            9,
            8,
            9,
            6,
            8,
            8,
            8,
            8,
            7,
            5,
            6,
            8,
            9,
            9,
            8,
            5,
            5,
            5,
            5,
            5,
            6,
            5,
        ],
    },
    {
        "title": "data-set-3",
        "data": None,
        "T": [1, 2, 167, 180],
        "n": [19, 20, 22, 16],
        "svd": [4, 4, 9, 7],
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
        "svd": [
            6,
            7,
            7,
            8,
            7,
            7,
            6,
            8,
            7,
            7,
            7,
            8,
            8,
            8,
            7,
            7,
            8,
            8,
            7,
            7,
            6,
            7,
            6,
            7,
            7,
            7,
            6,
            7,
            7,
            7,
            7,
            8,
            7,
            7,
            7,
            6,
            6,
            7,
            5,
            7,
            7,
            8,
            7,
            7,
            6,
            7,
            7,
            8,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            7,
            8,
            7,
            7,
            7,
            5,
            7,
            8,
            8,
            6,
            6,
            8,
            8,
            6,
            7,
            8,
            7,
            6,
        ],
    },
]

for data_set in data_sets:
    t = len(data_set["T"])
    n = len(data_set["n"])
    s = len(data_set["svd"])
    assert t == n == s, f"{data_set['title']}, T={t}, n={n}, svd={s}"

data_folder = "sim_data"
filename_end = f"_{num_orbits}-orbits.npy"

epsilon = 1e-12

starting_snapshot = 500

exact = True
opt = True
delay = 1


#############################
save = False
show = True

scale = True

print_eigs = False
print_latex_table = False

# set to negative to calculate all data sets
only_this_data_set_index = 3

# delete sensors (starting index, how many to delete)
delete_dimensions_from = (26, 0)
#############################


for data_set in data_sets:
    if only_this_data_set_index >= 0:
        data_set = data_sets[only_this_data_set_index]

    path = f"{data_folder}/{data_set['title']}{filename_end}"
    data = np.load(path)[:, 1:].astype(float)

    data_set_title = data_set["title"]
    print(f"loading {data_set_title} done\n")

    sampling_interval_list = data_set["T"]
    number_of_snapshots_list = data_set["n"]
    svd_list = data_set["svd"]

    print(f"\n{data_set_title}\n")

    if print_eigs:
        print("eigs, |eigs|, growth rate\n")

    if print_latex_table:
        print(
            "T, n, svd, rmse, true mean f, dominant f, relative error, true mean steps per one cycle, steps per one cycle"
        )

    for sampling_selector in range(len(sampling_interval_list)):
        sampling_interval = sampling_interval_list[sampling_selector]

        # if sampling_interval > 39:
        #     break

        n = number_of_snapshots_list[sampling_selector]
        svd = svd_list[sampling_selector]

        end_snapshot = starting_snapshot + n * sampling_interval - sampling_interval

        # duration from "total_time"
        # duration = (
        #     data[0][end_snapshot] - data[0][starting_snapshot] + sampling_interval
        # )

        # mean angle per step over complete time frame
        mean_angular_velocity = np.mean(
            data[16, starting_snapshot : end_snapshot + sampling_interval]
        )

        true_mean_steps_per_one_cycle = 360 / mean_angular_velocity
        true_mean_frequency = 1 / true_mean_steps_per_one_cycle

        # remove the dimensions
        start, end = delete_dimensions_from
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

        dmd, svd, tlsq = compute_dmd(
            data_subset,
            exact=exact,
            opt=opt,
            pre_proc_delay=delay,
            svd_rank=svd,
        )

        rmse = np.mean(get_rmse(data_subset, dmd.reconstructed_data.real))

        steps_on_x_axis = list(
            range(starting_snapshot, end_snapshot + 1, sampling_interval)
        )

        ### dmd.modes.shape = (n_features, n_modes)
        ### dmd.dynamics.shape = (n_modes, n_snapshots)
        ### dmd.eigs.shape = (n_modes,)
        ### dmd.growth_rate.shape = (n_modes,)

        # sort amplitudes like pydmd.plotter.plot_summary()
        sorted_indices = np.argsort(-np.abs(dmd.amplitudes))
        eigs = dmd.eigs[sorted_indices]

        modes = dmd.modes[:, sorted_indices]
        dynamics = dmd.dynamics[sorted_indices, :]
        amplitudes = np.abs(dmd.amplitudes[sorted_indices])

        # original using wrong (???) PyDMD implementation
        # growth_rate = dmd.growth_rate[sorted_indices]

        # correct way to compute continuous growth rate, already sorted
        # see 2016KutzEtAl (doi: 10.1137/1.9781611974508)
        growth_rate = (np.log(eigs) / sampling_interval).real

        if print_eigs:
            print(f"T={sampling_interval}, n={n}")
            print(f"{np.round(eigs, decimals=3)}\n")
            print(f"{np.round(np.abs(eigs), decimals=3)}\n")
            print(f"{np.round(growth_rate, decimals=5)}\n\n")

        eigs_oscillatory = eigs[np.abs(np.imag(eigs)) > epsilon]

        # frequency in cycles per unit time
        frequencies = np.angle(eigs_oscillatory) / (2 * np.pi * sampling_interval)
        dmd_frequencies = dmd.frequency
        dominant_frequencies = frequencies[frequencies > 0]

        relative_error_frequencies = np.abs(
            true_mean_frequency - dominant_frequencies
        ) / np.abs(true_mean_frequency)

        steps_per_one_cycle = 1 / dominant_frequencies
        decimal_rounding = 5

        if print_latex_table:
            print(
                f"{sampling_interval} & {n} & {svd} & {np.round(rmse, decimals=decimal_rounding)} "
                f"& {np.round(true_mean_frequency, decimals=decimal_rounding)} & {',\\newline'.join([f'{x}' for x in np.round(dominant_frequencies, decimals=decimal_rounding)])} "
                f"& {',\\newline'.join([f'{x}' for x in np.round(relative_error_frequencies, decimals=decimal_rounding)])} & {np.round(true_mean_steps_per_one_cycle, decimals=2)} "
                f"& {',\\newline'.join([f'{x}' for x in np.round(steps_per_one_cycle, decimals=2)])} \\\\"
            )

        if save or show:
            tick_font_size = 20
            axes_label_font_size = 25
            line_width = 3
            max_plots_per_row = 4
            number_modes = modes.shape[1]
            rows_to_plot = ceil(number_modes / max_plots_per_row)
            cols_to_plot = min(number_modes, max_plots_per_row)
            size_plot_mode_x = 4.5
            size_plot_mode_y = 6

            mode_indices = None
            if mode_indices is None:
                mode_indices = range(0, number_modes)

            # dynamics of modes over time heatmap
            figsize = (size_plot_mode_x * cols_to_plot, size_plot_mode_y * rows_to_plot)
            fig = plt.figure(figsize=figsize)
            for i, k in enumerate(mode_indices):
                row = i // max_plots_per_row
                col = i % max_plots_per_row
                ax = plt.subplot2grid((rows_to_plot, cols_to_plot), (row, col))
                contribution_k = np.outer(modes[:, k], dynamics[k, :])
                cax = ax.imshow(
                    contribution_k.real,
                    aspect="auto",
                    cmap="coolwarm",
                    interpolation="none",
                )
                cbar = plt.colorbar(cax, ax=ax, shrink=0.9)
                cbar.set_label(
                    f"Contribution to mode {k} (Real)", fontsize=tick_font_size
                )
                cbar.ax.tick_params(labelsize=tick_font_size)

                ax.set_title(f"Mode {k}", fontsize=axes_label_font_size)
                ax.set_xlabel("Time", fontsize=axes_label_font_size)

                tick_interval = ceil(n / 5)
                x_tick_indices = np.arange(0, len(steps_on_x_axis), tick_interval)
                ax.set_xticks(x_tick_indices)
                ax.set_xticklabels([steps_on_x_axis[i] for i in x_tick_indices])

                if col == 0:
                    ax.set_ylabel("Sensor Index", fontsize=axes_label_font_size)
                ax.tick_params(labelsize=tick_font_size)
            plt.tight_layout()

            if save:
                plots_path = f"plots/dmd_analysis/{data_set_title}/modes/"
                title = "modes"
                plt.savefig(
                    f"{plots_path}{data_set_title}_{title}_T-{sampling_interval}_n-{n}_t0-{starting_snapshot}_"
                    f"svd-{svd}{('_exact' if exact else '')}{('_opt' if opt else '')}{'' if scale else '_non-scaled'}.pdf",
                    format="pdf",
                )
            if show:
                plt.show()
            plt.clf()

            # eigenvalues with unit circle
            figsize = (12, 9.5)
            tick_font_size = 27
            legend_font_size = 27
            axes_label_font_size = 29
            line_width = 3
            marker_size = 6
            fig, ax = plt.subplots(figsize=figsize)
            sc = ax.scatter(
                eigs.real,
                eigs.imag,
                c=growth_rate,
                cmap="viridis",
                s=marker_size * 40,
                edgecolor="k",
                zorder=3,
            )
            cbar = plt.colorbar(sc, ax=ax, shrink=1)
            cbar.set_label(
                r"Continuous growth rate ($Re(\omega)$)", fontsize=axes_label_font_size
            )
            cbar.ax.tick_params(labelsize=tick_font_size)

            theta = np.linspace(0, 2 * np.pi, 300)
            ax.plot(np.cos(theta), np.sin(theta), "r--", linewidth=line_width)
            ax.set_xlabel("Re(" r"$\lambda$" ")", fontsize=axes_label_font_size)
            ax.set_ylabel("Im(" r"$\lambda$" ")", fontsize=axes_label_font_size)
            ax.set_aspect("equal")
            ax.grid(True)
            plt.tick_params(axis="both", labelsize=tick_font_size)
            plt.tight_layout()

            if save:
                plots_path = f"plots/dmd_analysis/{data_set_title}/eigs_unit_circle/"
                title = "eigs-unit-circle"
                plt.savefig(
                    f"{plots_path}{data_set_title}_{title}_T-{sampling_interval}_n-{n}_t0-{starting_snapshot}_"
                    f"svd-{svd}{('_exact' if exact else '')}{('_opt' if opt else '')}{'' if scale else '_non-scaled'}.pdf",
                    format="pdf",
                )
            if show:
                plt.show()
            plt.clf()

            ###################  old mode and dynamics plots, unused in thesis
            # # modes
            # tick_font_size = 17
            # axes_label_font_size = 20
            # line_width = 3
            # size_plot_mode_x = 5
            # size_plot_mode_y = 5
            # figsize = (size_plot_mode_x * cols_to_plot, size_plot_mode_y * rows_to_plot)
            # fig = plt.figure(figsize=figsize)
            # for i, k in enumerate(mode_indices):
            #     row = i // max_plots_per_row
            #     col = i % max_plots_per_row
            #     ax_mode = plt.subplot2grid((rows_to_plot, cols_to_plot), (row, col))
            #     ax_mode.plot(
            #         np.arange(modes.shape[0]), modes[:, k].real, linewidth=line_width
            #     )
            #     ax_mode.set_title(f"Mode {k}", fontsize=axes_label_font_size)
            #     ax_mode.set_xlabel("Sensor Index", fontsize=axes_label_font_size)
            #     if col == 0:
            #         ax_mode.set_ylabel(
            #             "Mode Amplitude (Real)", fontsize=axes_label_font_size + 2
            #         )
            #     ax_mode.tick_params(labelsize=tick_font_size)
            #     ax_mode.grid(True)
            # plt.tight_layout()

            # if save:
            #     plots_path = f"plots/dmd_analysis/{data_set_title}/modes/"
            #     title = "modes"
            #     plt.savefig(
            #         f"{plots_path}{data_set_title}_{title}_T-{sampling_interval}_n-{n}_t0-{starting_snapshot}_"
            #         f"svd-{svd}{('_exact' if exact else '')}{('_opt' if opt else '')}{'' if scale else '_non-scaled'}.pdf",
            #         format="pdf",
            #     )
            # if show:
            #     plt.show()
            # plt.clf()

            # # dynamics per mode
            # tick_font_size = 17
            # axes_label_font_size = 20
            # line_width = 3
            # size_plot_mode_x = 5
            # size_plot_mode_y = 5
            # figsize = (size_plot_mode_x * cols_to_plot, size_plot_mode_y * rows_to_plot)

            # fig = plt.figure(figsize=figsize)
            # for i, k in enumerate(mode_indices):
            #     row = i // max_plots_per_row
            #     col = i % max_plots_per_row
            #     ax_dyn = plt.subplot2grid((rows_to_plot, cols_to_plot), (row, col))
            #     ax_dyn.plot(steps_on_x_axis, dynamics[k, :].real, linewidth=line_width)
            #     ax_dyn.set_title(f"Dynamics of Mode {k}", fontsize=axes_label_font_size)
            #     ax_dyn.set_xlabel("Time", fontsize=axes_label_font_size)
            #     if col == 0:
            #         ax_dyn.set_ylabel(
            #             "Amplitude (Real)", fontsize=axes_label_font_size + 2
            #         )
            #     ax_dyn.tick_params(labelsize=tick_font_size)
            #     ax_dyn.grid(True)
            # plt.tight_layout()

            # if save:
            #     plots_path = f"plots/dmd_analysis/{data_set_title}/dynamics/"
            #     title = "dynamics"
            #     plt.savefig(
            #         f"{plots_path}{data_set_title}_{title}_T-{sampling_interval}_n-{n}_t0-{starting_snapshot}_"
            #         f"svd-{svd}{('_exact' if exact else '')}{('_opt' if opt else '')}{'' if scale else '_non-scaled'}.pdf",
            #         format="pdf",
            #     )
            # if show:
            #     plt.show()
            # plt.clf()

        print("\n---------------\n")

    if only_this_data_set_index >= 0:
        break

    print(
        "____________________________________________________________________________________________________________________________________________________\n\n"
    )
