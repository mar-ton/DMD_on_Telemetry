import numpy as np

from pydmd.utils import compute_rank
from pydmd import DMD
from pydmd.preprocessing import hankel_preprocessing

import warnings

warnings.filterwarnings("ignore")


def get_rmse(X_true, X_pred):
    return np.sqrt(np.mean((X_true - X_pred) ** 2, axis=1))


def get_min_max_scaling(data):
    _min = data.min(axis=1, keepdims=True)
    _max = data.max(axis=1, keepdims=True)
    denom = _max - _min
    denom = np.where(denom < epsilon, 1.0, denom)
    return ((data - _min) / denom, _max, _min)


def decompose_and_get_rmse(X, exact=True, opt=True, delay=1):
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
    {
        "title": "data-set-1",
        "data": None,
        "results": [],
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
        "n": [],
    },
    {
        "title": "data-set-2",
        "data": None,
        "results": [],
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
        "n": [],
    },
    {
        "title": "data-set-3",
        "data": None,
        "results": [],
        "T": [1, 2, 167, 180],
        "n": [],
    },
    {
        "title": "data-set-4",
        "data": None,
        "results": [],
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
        "n": [],
    },
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

max_snapshot_count = 100

#############################
exact = True
opt = True
delay = 1

# set to negative to calculate all data sets
only_this_data_set_index = 2
#############################

for i, data_set in enumerate(data_sets):
    if only_this_data_set_index >= 0:
        data_set = data_sets[only_this_data_set_index]
    data_set_title = data_set["title"]
    data = data_set["data"]
    results = data_set["results"]
    sampling_interval_list = data_set["T"]
    n_list = data_set["n"]

    print(f"\n{data_set_title}\n")

    for sampling_interval in sampling_interval_list:
        results = []
        for number_of_snapshots in range(max((delay + 1), 3), max_snapshot_count + 1):
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
                starting_snapshot : end_snapshot + sampling_interval : (
                    sampling_interval
                ),
            ]

            data_subset_min_max, _, _ = get_min_max_scaling(data_subset)

            try:
                min_max_rmse = decompose_and_get_rmse(data_subset_min_max, exact, opt)
            except Exception:
                min_max_rmse = -1
            results.append((number_of_snapshots, min_max_rmse.real))

        min_tuple = min((v for v in results if v[1] > 0), key=lambda x: x[1])
        n_list.append(min_tuple)

    print([k for k, v in n_list])
    print("\n")
    print([v for k, v in n_list])
    print("\n")
    print(
        "__________________________________________________________________________\n"
    )

    if only_this_data_set_index >= 0:
        break
