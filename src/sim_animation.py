import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation

# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sim_modularized import CubeSat, _normalize
import global_simulation_config as config


cube_sat = CubeSat()
# cube_sat.main_rotation.axis = _normalize(np.array([1, 0, 0]))
# cube_sat.main_rotation.angle = 0
# cube_sat.precession.axis = _normalize(np.array([0, 0, 1]))
# cube_sat.precession.angle = 1
# cube_sat.additional_rotations = []
# cube_sat.additional_rotations = [
#     _Rotation(axis=np.array([0, 0, 1]), angle=5),
#     _Rotation(axis=np.array([0, 1, 0]), angle=5),
# ]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

counter_text = ax.text2D(
    0.05,
    0.95,
    f"Orbit: {cube_sat.simulation.orbit} | Current Step: {cube_sat.simulation.current_time}\nTotal Step: {cube_sat.simulation.total_time}",
    transform=ax.transAxes,
    fontsize=14,
    color="black",
)

arrow_main_rotation = None
arrow_top_orientation = None
arrow_front_orientation = None
# arrow_right_orientation = None
arrow_sum_rotations = None

text_main_axis = None
text_sum_rotations = None

text_top = None
text_front = None
# text_right = None

offset = np.array([2, 0, 0])
ax.quiver(
    *offset,
    *-config.sun_direction,
    color="orange",
    length=1.0,
    normalize=True,
    linewidth=3,
)
label_pos = offset + (-config.sun_direction) * 1.1  # Push label beyond the arrow tip
ax.text(*label_pos, "Sun", color="orange", fontsize=14, fontweight="bold")

if cube_sat.additional_rotations:
    additional_rotations = Rotation.identity()
    for rotation in reversed(cube_sat.additional_rotations):
        additional_rotations *= rotation.get_rotation()

    ax.text(
        *(_normalize(additional_rotations.as_rotvec()) * 0.5),
        "Additional rotations axis",
        color="green",
        fontsize=10,
    )

    ax.quiver(
        0,
        0,
        0,
        *_normalize(additional_rotations.as_rotvec()),
        color="green",
        length=1.0,
        normalize=True,
        pivot="middle",
    )
else:
    additional_rotations = Rotation.identity()


def update(frame):
    global arrow_main_rotation, arrow_top_orientation, arrow_front_orientation, arrow_sum_rotations, text_main_axis, text_front, text_top, text_sum_rotations
    # arrow_right_orientation, \
    # text_right, \

    if arrow_main_rotation:
        arrow_main_rotation.remove()
    if arrow_top_orientation:
        arrow_top_orientation.remove()
    if arrow_front_orientation:
        arrow_front_orientation.remove()
    # if arrow_right_orientation:
    #     arrow_right_orientation.remove()
    if arrow_sum_rotations:
        arrow_sum_rotations.remove()
    if text_main_axis:
        text_main_axis.remove()
    if text_sum_rotations:
        text_sum_rotations.remove()
    if text_front:
        text_front.remove()
    if text_top:
        text_top.remove()
    # if text_right:
    #     text_right.remove()

    # Step the simulation
    vector_main_rotation = cube_sat.main_rotation.axis
    vector_top_orientation = cube_sat.face_orientations["top"]
    vector_front_orientation = cube_sat.face_orientations["front"]
    # vector_right_orientation = cube_sat.face_orientations["right"]
    vector_sum_rotations = _normalize(cube_sat.current_combined_rotation.as_rotvec())

    cube_sat.step()

    # Color depending on orientation
    dot = np.dot(vector_main_rotation, config.sun_direction)
    color_main_rotation = "red" if dot > 0 else "blue"

    dot = np.dot(vector_top_orientation, config.sun_direction)
    color_top_orientation = "pink" if dot > 0 else "darkred"

    dot = np.dot(vector_front_orientation, config.sun_direction)
    color_front_orientation = "pink" if dot > 0 else "darkred"

    # dot = np.dot(vector_right_orientation, config.sun_direction)
    # color_right_orientation = "pink" if dot >= 0 else "darkred"

    dot = np.dot(vector_sum_rotations, config.sun_direction)
    color_sum_rotations = "magenta" if dot > 0 else "indigo"

    # Draw the main vector arrow
    arrow_main_rotation = ax.quiver(
        0,
        0,
        0,
        *vector_main_rotation,
        color=color_main_rotation,
        length=1.0,
        normalize=True,
        pivot="middle",
    )

    text_main_axis = ax.text(
        *(vector_main_rotation * 0.5),
        "Main axis",
        color=color_main_rotation,
        fontsize=10,
    )

    text_sum_rotations = ax.text(
        *(vector_sum_rotations * 0.5),
        "Sum all rotations",
        color=color_sum_rotations,
        fontsize=10,
    )

    text_front = ax.text(
        *(vector_front_orientation * 0.5),
        "front",
        color=color_front_orientation,
        fontsize=10,
    )

    # text_right = ax.text(
    #     *(vector_right_orientation * 0.5),
    #     "right",
    #     color=color_front_orientation,
    #     fontsize=10,
    # )

    text_top = ax.text(
        *(vector_top_orientation * 0.5),
        "top",
        color=color_top_orientation,
        fontsize=10,
    )

    arrow_top_orientation = ax.quiver(
        0,
        0,
        0,
        *vector_top_orientation,
        color=color_top_orientation,
        length=1.0,
        normalize=True,
        pivot="middle",
    )

    arrow_front_orientation = ax.quiver(
        0,
        0,
        0,
        *vector_front_orientation,
        color=color_front_orientation,
        length=1.0,
        normalize=True,
        pivot="middle",
    )

    # arrow_right_orientation = ax.quiver(
    #     0,
    #     0,
    #     0,
    #     *vector_right_orientation,
    #     color=color_right_orientation,
    #     length=1.0,
    #     normalize=True,
    #     pivot="middle",
    # )

    arrow_sum_rotations = ax.quiver(
        0,
        0,
        0,
        *vector_sum_rotations,
        color=color_sum_rotations,
        length=1.0,
        normalize=True,
        pivot="middle",
    )

    # Update the text
    counter_text.set_text(
        f"Orbit: {cube_sat.simulation.orbit} | Current Step: {cube_sat.simulation.current_time}\nTotal Step: {cube_sat.simulation.total_time}"
    )

    return (
        arrow_main_rotation,
        counter_text,
        arrow_sum_rotations,
        arrow_top_orientation,
        arrow_front_orientation,
        # arrow_right_orientation,
        text_main_axis,
        text_sum_rotations,
        text_front,
        text_top,
        # text_right,
    )


ani = FuncAnimation(fig, update, frames=range(200), interval=50, blit=False)
plt.show()
