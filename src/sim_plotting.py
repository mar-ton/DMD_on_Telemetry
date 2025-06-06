import matplotlib.pyplot as plt
import numpy as np
import os.path
import global_simulation_config as config
from sim_modularized import get_simulated_data

plt.rcParams["text.usetex"] = True


def scale_min_max(data, epsilon=1e-12):
    _min = data.min(axis=1, keepdims=True)
    _max = data.max(axis=1, keepdims=True)
    denom = _max - _min
    denom = np.where(denom < epsilon, 1.0, denom)
    return ((data - _min) / denom, _min, _max)


selected_sensors = [
    "temperature_front",  # 3
    "temperature_back",  # 4
    "temperature_right",  # 5
    "temperature_left",  # 6
    "temperature_top",  # 7
    "temperature_bottom",  # 8
]

selected_sensors_2 = [
    "battery_level",  # 2
    "temperature_center",  # 9
]


def get_index_from_title(title) -> int:
    try:
        return config.titles.index(title)
    except Exception:
        raise ValueError(f"could not find {title}")


num_orbits = 500

#############################
start_snapshot = 0

save = False
show = True

data_set_1 = ("data-set-1", "main_yz_20_prec_xy_0.3_add_xy5_-y-z3")
data_set_2 = ("data-set-2", "main_0.5xyz_2_prec_yz_0.5_add_-xyz0.1_-y-z0.2")
data_set_3 = ("data-set-3", "main_-0.5y-z_180_prec_-yz_1_add_-xy-z32_0.1x-0.2y0.3z20")
data_set_4 = ("data-set-4", "main_-0.5y-z_50_prec_-yz_10_add_-xy-z120_0.1x-0.2y0.3z20")


#############################
# make sure you really want to update the data (existing ones will be overwritten)
update_data = False

used_set_title, used_set_settings = data_set_4

scale = False

# set T to None to use orbits_to_plot
T = None
n = 23
steps_to_plot = None if T is None else T * n

orbits_to_plot = 0

data_folder = "sim_data"
data_filename = f"{used_set_title}_{num_orbits}-orbits.npy"
data_path = f"{data_folder}/{data_filename}"


if update_data:
    data = get_simulated_data(num_orbits)
    np.save(data_path, data)
else:
    if not os.path.isfile(data_path):
        data = get_simulated_data(num_orbits)
        np.save(data_path, data)
    else:
        data = np.load(data_path)

titles = data[:, 0]
assert np.all(titles == config.titles)

if steps_to_plot is None:
    steps_to_plot = int(
        (orbits_to_plot if orbits_to_plot != 0 else num_orbits) * config.orbital_period
    )
end_snapshot = start_snapshot + steps_to_plot

# +1 to delete titles
data = data[:, start_snapshot + 1 : end_snapshot + 1].astype(float)
print(data.shape)

# mean_angular_velocity = np.round(
#     np.mean(data[get_index_from_title("combined_angular_velocity"), :]), decimals=4
# )
# print(f"mean angular velo:\n{mean_angular_velocity}\n")

if scale:
    data, _, _ = scale_min_max(data)

# time vector
t = np.arange(start_snapshot, end_snapshot, dtype=int)

fig_size = (12, 9.4)
tick_font_size = 27
axes_label_font_size = 29
line_width = 3
line_alpha = 0.7
legend_font_size = 27

if show or save:
    # first plot
    plt.figure(figsize=fig_size)
    for title in selected_sensors:
        plt.plot(
            t,
            data[get_index_from_title(title), :],
            label=title,
            alpha=line_alpha,
            linewidth=line_width,
        )
    plt.ylabel(
        f"{('Sensor value' if len(selected_sensors) == 1 else 'Sensor values')}",
        fontsize=axes_label_font_size,
    )
    plt.xlabel("Simulated Steps", fontsize=axes_label_font_size)

    # ax = plt.gca()
    # # ax.set_xlim(left=t[0])
    # tick_interval = ceil(n / 6)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1, offset=tick_interval))
    plt.tick_params(axis="both", labelsize=tick_font_size)

    plt.grid(True)
    plt.tight_layout()
    # legend = plt.legend(loc="upper left", fontsize=legend_font_size)
    # for line in legend.get_lines():
    #     line.set_linewidth(6)
    plot_folder = "plots/sim_data_plots"
    title = "comparison-for-dmd"
    plot_filename = (
        f"{used_set_title}_{title}_start-{start_snapshot}_"
        f"{f'orbits-{orbits_to_plot}' if T is None else f'T-{T}_n-{n}'}"
        f"{'' if scale else '_non-scaled'}"
        ".pdf"
    )
    if save:
        plt.savefig(f"{plot_folder}/{plot_filename}", format="pdf")
    if show:
        plt.show()
    plt.clf()

    # second plot
    plt.figure(figsize=fig_size)
    for title in selected_sensors_2:
        plt.plot(
            t,
            data[get_index_from_title(title), :],
            label=title,
            alpha=line_alpha,
            linewidth=line_width,
        )
    plt.xlabel("Simulated Steps", fontsize=axes_label_font_size)
    plt.ylabel(
        f"{('Sensor value' if len(selected_sensors_2) == 1 else 'Sensor values')}",
        fontsize=axes_label_font_size,
    )
    plt.tick_params(axis="both", labelsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()
    legend = plt.legend(loc="lower right", fontsize=legend_font_size)
    for line in legend.get_lines():
        line.set_linewidth(6)
    plot_folder = "plots/sim_data_plots"
    title = "center"
    plot_filename = (
        f"{used_set_title}_{title}_start-{start_snapshot}_"
        f"{f'orbits-{orbits_to_plot}' if T is None else f'T-{T}_n-{n}'}"
        ".pdf"
    )
    if save:
        plt.savefig(f"{plot_folder}/{plot_filename}", format="pdf")
    if show:
        plt.show()
    plt.clf()
