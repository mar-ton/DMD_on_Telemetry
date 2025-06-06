import matplotlib.pyplot as plt
import numpy as np
import global_simulation_config as config

plt.rcParams["text.usetex"] = True


def scale_min_max(data, epsilon=1e-12):
    _min = data.min(axis=1, keepdims=True)
    _max = data.max(axis=1, keepdims=True)
    denom = _max - _min
    denom = np.where(denom < epsilon, 1.0, denom)
    return ((data - _min) / denom, _min, _max)


selected_sensors = [
    "battery_level",  # 2
    "heater_running",  # 28
]


def get_index_from_title(title) -> int:
    try:
        return config.titles.index(title)
    except Exception:
        raise ValueError(f"could not find {title}")


num_orbits = 100
start_snapshot = 1

#############################
save = False
show = True

scale = True
#############################

steps_to_plot = 140

used_set_title = "power-data-validation"

data_folder = "sim_data"
data_filename = f"{used_set_title}_{num_orbits}-orbits_main_z_20.npy"
data_path = f"{data_folder}/{data_filename}"

data = np.load(data_path)

titles = data[:, 0]


end_snapshot = start_snapshot + steps_to_plot

# +1 to delete titles
data = data[:, start_snapshot + 1 : end_snapshot + 1].astype(float)

if scale:
    data, _, _ = scale_min_max(data)

# time vector
t = np.arange(start_snapshot, end_snapshot, dtype=int)

fig_size = (12, 9.4)
tick_font_size = 27
axes_label_font_size = 29
line_width = 2
line_alpha = 0.7
legend_font_size = 27

selected_sensors = [(3, "battery_level"), (28, "heater_running")]

if show or save:
    plt.figure(figsize=fig_size)
    for index, title in selected_sensors:
        plt.plot(
            t,
            data[index, :],
            label=title,
            alpha=line_alpha,
            linewidth=line_width,
        )
    plt.xlim(left=t[0], right=t[-1])
    plt.ylabel(
        "Sensor values (scaled)",
        fontsize=axes_label_font_size,
    )
    plt.xlabel("Simulated Steps", fontsize=axes_label_font_size)

    ax = plt.gca()
    xticks = ax.get_xticks()

    highlights = [62, 76, 93, 104]
    for highlight in highlights:
        plt.axvline(
            x=highlight,
            color="blue",
            linestyle="--",
            alpha=line_alpha,
            linewidth=line_width,
        )
        if highlight not in xticks:
            up_down = 5
            for i in range(highlight - up_down, highlight + up_down + 1):
                if i in xticks:
                    xticks = [x for x in xticks if x != i]
            xticks.append(highlight)

    # ax.set_xlim(left=t[0], right=t[-1])
    ax.set_xticks(np.sort(xticks))

    plt.tick_params(axis="both", labelsize=tick_font_size)
    plt.grid(True)
    plt.tight_layout()
    legend = plt.legend(loc="upper right", fontsize=legend_font_size)
    for line in legend.get_lines():
        line.set_linewidth(line_width + 2)

    if save:
        plot_folder = "plots"
        plot_filename = "battery-and-heater_2-orbits_main_z_20_new.pdf"
        plt.savefig(f"{plot_folder}/{plot_filename}", format="pdf")
    if show:
        plt.show()
    plt.clf()
