import numpy as np
from scipy.linalg import norm

from pydmd.utils import compute_rank
from pydmd import DMD
from pydmd.preprocessing import hankel_preprocessing

import warnings

warnings.filterwarnings("ignore")


colors = ["navy", "chocolate", "magenta", "lawngreen"]
markers = ["<", ">", "^", "v"]

num_orbits = 500

data_folder = "sim_data"
filename_end = f"_{num_orbits}-orbits.npy"

epsilon = 1e-12


def get_rmse(X_actual, X_predicted):
    return np.sqrt(np.mean((X_actual - X_predicted) ** 2, axis=1))


def get_relative_error(X_actual, X_predicted):
    return norm(X_predicted - X_actual) / norm(X_actual)


def scale_min_max(data, epsilon=1e-12):
    _min = data.min(axis=1, keepdims=True)
    _max = data.max(axis=1, keepdims=True)
    denom = _max - _min
    denom = np.where(denom < epsilon, 1.0, denom)
    return ((data - _min) / denom, _min, _max)


def descale_min_max(data, min, max):
    return data * (max - min) + min


def get_optimal_svd_rank(X, Y=None, exact=True, opt=True, pre_proc_delay=1):
    errors_for_svd = []
    for rank in range(0, X.shape[1]):
        dmd = DMD(
            svd_rank=rank,
            exact=exact,
            opt=opt,
        )
        dmd = hankel_preprocessing(dmd, d=pre_proc_delay)
        dmd.fit(X, Y)
        rmse = np.mean(get_rmse(X, dmd.reconstructed_data.real))
        errors_for_svd.append((rank, rmse))
    errors_for_svd = np.array(errors_for_svd)
    optimal_svd_rank, _ = errors_for_svd[np.argmin(errors_for_svd[:, 1])]
    return int(optimal_svd_rank)


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

    return dmd, svd_rank


data_sets = [
    {
        "title": "data-set-1",
        "results": [],
        "T": [1, 2, 3, 4, 7, 8, 16, 75],
        "n": [15, 19, 21, 21, 22, 23, 20, 23],
        "svd": [4, 5, 6, 6, 7, 8, 6, 9],
    },
    {
        "title": "data-set-2",
        "results": [],
        "T": [2, 3, 4, 5, 6, 11, 12, 13],
        "n": [13, 16, 17, 17, 17, 21, 21, 23],
        "svd": [3, 4, 4, 4, 5, 7, 7, 8],
    },
    {
        "title": "data-set-3",
        "results": [],
        "T": [1, 2, 167, 180],
        "n": [19, 20, 22, 16],
        "svd": [4, 4, 9, 7],
    },
    {
        "title": "data-set-4",
        "results": [],
        "T": [1, 5, 9, 12, 15, 17, 18, 22],
        "n": [21, 22, 21, 20, 23, 22, 17, 23],
        "svd": [6, 7, 7, 8, 7, 7, 6, 8],
    },
]

for data_set in data_sets:
    t = len(data_set["T"])
    n = len(data_set["n"])
    s = len(data_set["svd"])
    assert t == n == s, f"{data_set['title']}, T={t}, n={n}, svd={s}"


starting_snapshot = 500


#############################
exact = True
opt = True
delay = 1

print_latex_table = True


# set to negative to calculate all data sets
only_this_data_set_index = 3

predicted_snapshots_max = 5
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
    results = data_set["results"]

    print(f"\n{data_set_title}\n")

    if print_latex_table:
        print(
            "snapshots predicted & relative error $\\eta$ & relative error $\\eta$ (scaled data)\\\\"
        )
        print("\\hline")

    for sampling_selector in range(len(sampling_interval_list)):
        sampling_interval = sampling_interval_list[sampling_selector]
        n = number_of_snapshots_list[sampling_selector]
        svd = svd_list[sampling_selector]
        this_sample_result = []

        # subsampled data, end inclusive
        end_snapshot = (
            starting_snapshot
            + (n + predicted_snapshots_max) * sampling_interval
            - sampling_interval
        )
        data_subset = data[
            1:,  # omit total time
            starting_snapshot : end_snapshot + sampling_interval : sampling_interval,
        ]

        training_data = data_subset[:, :n]
        # testing_data = data_subset[:, 1:]

        # if scale:
        training_data_scaled, training_data_min, training_data_max = scale_min_max(
            training_data
        )
        dmd_trained_base, _ = compute_dmd(
            training_data_scaled, exact=exact, opt=opt, pre_proc_delay=delay
        )
        reconstructed = descale_min_max(
            dmd_trained_base.reconstructed_data.real,
            training_data_min,
            training_data_max,
        )

        # else:
        dmd_trained_base, _ = compute_dmd(
            training_data, exact=exact, opt=opt, pre_proc_delay=delay
        )

        # (n predicted snapshots, eta non scaled, eta scaled)
        this_sample_result.append(
            (
                0,
                get_relative_error(
                    X_actual=training_data,
                    X_predicted=dmd_trained_base.reconstructed_data.real,
                ),
                get_relative_error(X_actual=training_data, X_predicted=reconstructed),
            )
        )

        # append predictions to copy of training data
        data_and_predictions_cumulated = np.copy(training_data)
        data_and_predictions_cumulated_scaled = np.copy(training_data)

        for i in range(predicted_snapshots_max):
            current_data = np.copy(data_and_predictions_cumulated)
            # if scale:
            current_data_scaled, current_data_min, current_data_max = scale_min_max(
                current_data
            )
            new_prediction_scaled = dmd_trained_base.predict(current_data_scaled).real
            new_prediction_scaled = descale_min_max(
                new_prediction_scaled, current_data_min, current_data_max
            )
            # else:
            new_prediction = dmd_trained_base.predict(current_data).real

            predicted_column = new_prediction[:, -1:]
            predicted_column_scaled = new_prediction_scaled[:, -1:]
            data_and_predictions_cumulated = np.hstack(
                (data_and_predictions_cumulated, predicted_column)
            )
            data_and_predictions_cumulated_scaled = np.hstack(
                (data_and_predictions_cumulated_scaled, predicted_column_scaled)
            )

            relative_error = get_relative_error(
                X_actual=data_subset[:, : n + i + 1],
                X_predicted=data_and_predictions_cumulated,
            )
            relative_error_scaled = get_relative_error(
                X_actual=data_subset[:, : n + i + 1],
                X_predicted=data_and_predictions_cumulated_scaled,
            )

            this_sample_result.append((i + 1, relative_error, relative_error_scaled))

        if print_latex_table:
            print(f"T={sampling_interval}, n={n} & & \\\\")
            for i, noscale, scale in this_sample_result:
                print(f"{i} & {noscale:.2e} & {scale:.2e} \\\\")
            print("\\hline")

    print("---------set done-------------\n\n")
    if only_this_data_set_index >= 0:
        break

    print(
        "____________________________________________________________________________________________________________________________________________________\n\n"
    )
