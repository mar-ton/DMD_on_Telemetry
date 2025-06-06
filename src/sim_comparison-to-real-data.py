import matplotlib.pyplot as plt
import numpy as np
import global_simulation_config as config

selected_sensors_swiss_cube = [
    "temperature_left",  # 6
    "temperature_bottom",  # 8
    "temperature_back",  # 4
]

selected_sensors_move = [
    "temperature_center",  # 9
]


def get_index_from_title(title) -> int:
    try:
        return config.titles.index(title)
    except Exception:
        raise ValueError(f"could not find {title}")


num_orbits = 500
orbits_to_plot = 1
steps_to_plot = orbits_to_plot * config.orbital_period

start_snapshot_move_1 = 5840
start_snapshot_move_2 = 5560
start_snapshot_swiss_cube = 2070

end_snapshot_move_1 = start_snapshot_move_1 + steps_to_plot
end_snapshot_move_2 = start_snapshot_move_2 + steps_to_plot
end_snapshot_swiss_cube = start_snapshot_swiss_cube + steps_to_plot

#############################
save = False
show = True
#############################

data_set_1 = ("data-set-1", "main_yz_20_prec_xy_0.3_add_xy5_-y-z3")
data_set_2 = ("data-set-2", "main_0.5xyz_2_prec_yz_0.5_add_-xyz0.1_-y-z0.2")

used_set_title1, used_set_settings1 = data_set_1
used_set_title2, used_set_settings12 = data_set_2

data_folder = "sim_data"
data_filename1 = f"{used_set_title1}_{num_orbits}-orbits.npy"
data_path1 = f"{data_folder}/{data_filename1}"

data_filename2 = f"{used_set_title2}_{num_orbits}-orbits.npy"
data_path2 = f"{data_folder}/{data_filename2}"

data1 = np.load(data_path1)
data2 = np.load(data_path2)

titles = data1[:, 0]
assert np.all(titles == config.titles)
titles = data2[:, 0]
assert np.all(titles == config.titles)


# delete titles
data_move_1 = data1[:, start_snapshot_move_1 + 1 : end_snapshot_move_1 + 1].astype(
    float
)
data_move_2 = data2[:, start_snapshot_move_2 + 1 : end_snapshot_move_2 + 1].astype(
    float
)
data_swiss = data2[
    :, start_snapshot_swiss_cube + 1 : end_snapshot_swiss_cube + 1
].astype(float)

# time vector
x_move_1 = np.arange(start_snapshot_move_1, end_snapshot_move_1)
x_move_2 = np.arange(start_snapshot_move_2, end_snapshot_move_2)
x_swiss_cube = np.arange(start_snapshot_swiss_cube, end_snapshot_swiss_cube)

fig_size_swiss = (11, 9)
fig_size_move = (12, 6)

tick_font_size = 17
marker_size = 8
legend_font_size = 20
axes_label_font_size = 28
line_width = 2
line_alpha = 0.7


# swiss_cube
plt.figure(figsize=fig_size_swiss)
plt.tick_params(axis="both", labelsize=tick_font_size)
for title in selected_sensors_swiss_cube:
    plt.plot(
        x_swiss_cube,
        data_swiss[get_index_from_title(title), :],
        label=title,
        alpha=line_alpha,
        linewidth=line_width,
    )
plt.ylabel(
    "Sensor values",
    fontsize=axes_label_font_size,
)
plt.xlabel("Simulated Steps", fontsize=axes_label_font_size)
plt.grid(True)
legend = plt.legend(loc="upper right", fontsize=legend_font_size)
for line in legend.get_lines():
    line.set_linewidth(6)
plt.tight_layout()
plot_folder = "plots/real_data_comparison/"
plot_filename = f"{used_set_title2}_swiss-cube_temperatures_start-{start_snapshot_swiss_cube}_steps-{steps_to_plot}.pdf"
if save:
    plt.savefig(f"{plot_folder}/{plot_filename}", format="pdf")
if show:
    plt.show()
plt.clf()


# move 1
plt.figure(figsize=fig_size_move)
plt.tick_params(axis="both", labelsize=tick_font_size)
for title in selected_sensors_move:
    plt.plot(
        x_move_1,
        data_move_1[get_index_from_title(title), :],
        label=title,
        linewidth=line_width,
    )
plt.xlabel("Simulated Steps", fontsize=axes_label_font_size)
plt.ylabel(
    "Sensor value",
    fontsize=axes_label_font_size,
)
plt.grid(True)
legend = plt.legend(loc="lower right", fontsize=legend_font_size)
for line in legend.get_lines():
    line.set_linewidth(6)
plt.tight_layout()
plot_folder = "plots/real_data_comparison/"
plot_filename = f"{used_set_title1}_move1_center-temp_start-{start_snapshot_move_1}_steps-{steps_to_plot}.pdf"
if save:
    plt.savefig(f"{plot_folder}{plot_filename}", format="pdf")
if show:
    plt.show()
plt.clf()


# move 2
plt.figure(figsize=fig_size_move)
plt.tick_params(axis="both", labelsize=tick_font_size)
for title in selected_sensors_move:
    plt.plot(
        x_move_2,
        data_move_2[get_index_from_title(title), :],
        label=title,
        linewidth=line_width,
    )
plt.xlabel("Simulated Steps", fontsize=axes_label_font_size)
plt.ylabel(
    "Sensor value",
    fontsize=axes_label_font_size,
)
plt.grid(True)
legend = plt.legend(loc="lower right", fontsize=legend_font_size)
for line in legend.get_lines():
    line.set_linewidth(6)
plt.tight_layout()
plot_folder = "plots/real_data_comparison/"
plot_filename = f"{used_set_title2}_move2_center-temp_start-{start_snapshot_move_2}_steps-{steps_to_plot}.pdf"
if save:
    plt.savefig(f"{plot_folder}{plot_filename}", format="pdf")
if show:
    plt.show()
plt.clf()
