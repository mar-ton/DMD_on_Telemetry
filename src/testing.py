import numpy as np
import copy
import io
import contextlib
import pytest
import global_simulation_config as config

from sim_modularized import (
    CubeSat,
    AxisAngle,
    _get_np_array_from_dict,
    _normalize,
    get_simulated_data,
)


def dicts_equal(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not np.allclose(dict1[key], dict2[key]):
            print(f"vector1={dict1[key]}\nvector2={dict2[key]}")
            return False
    return True


def check_angles(faces, test_ident):
    ninety_degree_pairs = [
        ("front", "right"),
        ("right", "top"),
        ("top", "front"),
    ]

    for face1, face2 in ninety_degree_pairs:
        assert np.isclose(np.dot(faces[face1], faces[face2]), 0), (
            f"expected 90 degree between {face1} and {face2} in {test_ident}"
        )

    opposing_pairs = [
        ("front", "back"),
        ("right", "left"),
        ("top", "bottom"),
    ]

    for face1, face2 in opposing_pairs:
        assert np.isclose(np.dot(faces[face1], faces[face2]), -1), (
            f"expected 180 degree between {face1} and {face2} in {test_ident}"
        )


##################################################
##################################################
### test helper functions
##################################################
##################################################

### test _get_np_array_from_dict()
dictionary_to_check = {
    "front": 0.1,
    "back": 1.23,
    "right": 2.345,
    "left": 3.4567,
    "top": 4.5678,
    "bottom": 5.6789,
}
list_check = np.array([0.1, 1.23, 2.345, 3.4567, 4.5678, 5.6789])
assert np.allclose(list_check, _get_np_array_from_dict(dictionary_to_check))


### test _normalize()
vector_before = np.array([1.5, -2.35, 8.7])
vector_check = np.array([0.16418957, -0.25723033, 0.9522995])
assert np.allclose(vector_check, _normalize(vector_before))


### test _in_shadow_of_planet()
cube_sat = CubeSat()

# test default is correct
assert cube_sat.simulation.current_time == 0
assert not cube_sat._in_shadow_of_planet()

# test off by one is correctly handled
cube_sat.simulation.current_time = config.sun_illumination_time - 1
assert not cube_sat._in_shadow_of_planet()
cube_sat.simulation.current_time = config.sun_illumination_time
assert cube_sat._in_shadow_of_planet()

# test last steps are correct as well
cube_sat.simulation.current_time = config.orbital_period - 1
assert cube_sat._in_shadow_of_planet()
cube_sat.simulation.current_time = config.orbital_period
# logic is true but current time is now bigger than orbital period
assert cube_sat._in_shadow_of_planet()
# check when modulus applied
cube_sat.simulation.current_time %= config.orbital_period
assert not cube_sat._in_shadow_of_planet()


print("All helper functions tests passed.")


##################################################
##################################################
### rotational tests
##################################################
##################################################


##################################################
### orientation of faces
##################################################
cube_sat = CubeSat()
cube_sat.main_rotation.axis = np.array([1, 0, 0])
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []

### test 1 ### no rotation
cube_sat.main_rotation.angle = 0
orientations_check = copy.deepcopy(cube_sat.face_orientations)
cube_sat.step()

assert dicts_equal(orientations_check, cube_sat.face_orientations), (
    f"should be: {orientations_check}\nwas: {cube_sat.face_orientations}"
)
check_angles(cube_sat.face_orientations, "test 1")


### test 2 ### 90 degrees along +x
cube_sat.main_rotation.angle = 90
cube_sat.step()

# make sure face_vectors are still unit vectors
for name, vec in cube_sat.face_orientations.items():
    assert np.isclose(np.linalg.norm(vec), 1.0), f"{name} not unit vector"
orientations_check = {
    "front": np.array([1, 0, 0]),
    "back": np.array([-1, 0, 0]),
    "right": np.array([0, 0, 1]),  # -> top
    "left": np.array([0, 0, -1]),  # -> bottom
    "top": np.array([0, -1, 0]),  # -> left
    "bottom": np.array([0, 1, 0]),  # -> right
}
assert dicts_equal(orientations_check, cube_sat.face_orientations), (
    f"should be:\n{orientations_check}\n\n is:\n{cube_sat.face_orientations}"
)
check_angles(cube_sat.face_orientations, "test 2")


### test 3 ### no rotation after one rotation
cube_sat.main_rotation.angle = 0
orientations_check = copy.deepcopy(cube_sat.face_orientations)
cube_sat.step()

# make sure face_vectors are still unit vectors
for name, vec in cube_sat.face_orientations.items():
    assert np.isclose(np.linalg.norm(vec), 1.0), f"{name} not unit vector"
assert dicts_equal(orientations_check, cube_sat.face_orientations), (
    f"should be: {orientations_check}\nwas: {cube_sat.face_orientations}"
)
check_angles(cube_sat.face_orientations, "test 3")


### test 4 ### 90 degrees along +y
cube_sat = CubeSat()
cube_sat.main_rotation.axis = np.array([0, 1, 0])
cube_sat.main_rotation.angle = 90
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []
cube_sat.step()

# make sure face_vectors are still unit vectors
for name, vec in cube_sat.face_orientations.items():
    assert np.isclose(np.linalg.norm(vec), 1.0), f"{name} not unit vector"
orientations_check = {
    "front": np.array([0, 0, -1]),  # -> bottom
    "back": np.array([0, 0, 1]),  # -> top
    "right": np.array([0, 1, 0]),
    "left": np.array([0, -1, 0]),
    "top": np.array([1, 0, 0]),  # -> front
    "bottom": np.array([-1, 0, 0]),  # -> back
}
assert dicts_equal(orientations_check, cube_sat.face_orientations), (
    f"should be:\n{orientations_check}\n\n is:\n{cube_sat.face_orientations}"
)
check_angles(cube_sat.face_orientations, "test 4")


### test 5 ### 90 degrees along +z with negative angle
cube_sat = CubeSat()
cube_sat.main_rotation.axis = np.array([0, 0, 1])
cube_sat.main_rotation.angle = -90
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []
cube_sat.step()

# make sure face_vectors are still unit vectors
for name, vec in cube_sat.face_orientations.items():
    assert np.isclose(np.linalg.norm(vec), 1.0), f"{name} not unit vector"
orientations_check = {
    "front": np.array([0, -1, 0]),  # -> left
    "back": np.array([0, 1, 0]),  # -> right
    "right": np.array([1, 0, 0]),  # -> front
    "left": np.array([-1, 0, 0]),  # -> back
    "top": np.array([0, 0, 1]),
    "bottom": np.array([0, 0, -1]),
}
assert dicts_equal(orientations_check, cube_sat.face_orientations), (
    f"should be:\n{orientations_check}\n\n is:\n{cube_sat.face_orientations}"
)
check_angles(cube_sat.face_orientations, "test 5")


##################################################
### precessing the main axis
##################################################

### test precession 90 degrees along +x with main axis +x
cube_sat = CubeSat()
cube_sat.main_rotation.axis = np.array([1, 0, 0])
cube_sat.main_rotation.angle = 0
cube_sat.precession.axis = np.array([1, 0, 0])
cube_sat.precession.angle = 90

# main axis should stay the same
orientations_check = copy.deepcopy(cube_sat.main_rotation.axis)
cube_sat.step()
assert np.allclose(orientations_check, cube_sat.main_rotation.axis)
check_angles(cube_sat.face_orientations, "test 6")


### test precession 90 degrees along +x with main axis +y
cube_sat.main_rotation.axis = np.array([0, 1, 0])
cube_sat.main_rotation.angle = 0
cube_sat.precession.axis = np.array([1, 0, 0])
cube_sat.precession.angle = 90
cube_sat.step()
orientations_check = np.array([0, 0, 1])  # +y -> +z
assert np.allclose(orientations_check, cube_sat.main_rotation.axis)
check_angles(cube_sat.face_orientations, "test 7")


### test precession 90 degrees along +x with main axis -y
cube_sat.main_rotation.axis = np.array([0, -1, 0])
cube_sat.main_rotation.angle = 0
cube_sat.precession.axis = np.array([1, 0, 0])
cube_sat.precession.angle = 90
cube_sat.step()
orientations_check = np.array([0, 0, -1])  # -y -> -z
assert np.allclose(orientations_check, cube_sat.main_rotation.axis)
check_angles(cube_sat.face_orientations, "test 8")


### test precession 90 degrees along -y with main axis -x
cube_sat.main_rotation.axis = np.array([-1, 0, 0])
cube_sat.main_rotation.angle = 0
cube_sat.precession.axis = np.array([0, -1, 0])
cube_sat.precession.angle = 90
cube_sat.step()
orientations_check = np.array([0, 0, -1])  # -x -> -z
assert np.allclose(orientations_check, cube_sat.main_rotation.axis)
check_angles(cube_sat.face_orientations, "test 9")


### test precession 90 degrees along +z with main axis -x
cube_sat.main_rotation.axis = np.array([-1, 0, 0])
cube_sat.main_rotation.angle = 0
cube_sat.precession.axis = np.array([0, 0, 1])
cube_sat.precession.angle = 90
cube_sat.step()
orientations_check = np.array([0, -1, 0])  # -x -> -y
assert np.allclose(orientations_check, cube_sat.main_rotation.axis)
check_angles(cube_sat.face_orientations, "test 10")


##################################################
### additional axes
##################################################

### test 70+20 degrees along +x with identical axes
cube_sat = CubeSat()
cube_sat.main_rotation.angle = 0
cube_sat.precession.angle = 0
cube_sat.additional_rotations = [
    AxisAngle(np.array([1, 0, 0]), 70),
    AxisAngle(np.array([1, 0, 0]), 20),
]
cube_sat.step()
orientations_check = {
    "front": np.array([1, 0, 0]),
    "back": np.array([-1, 0, 0]),
    "right": np.array([0, 0, 1]),  # -> top
    "left": np.array([0, 0, -1]),  # -> bottom
    "top": np.array([0, -1, 0]),  # -> left
    "bottom": np.array([0, 1, 0]),  # -> right
}
assert dicts_equal(orientations_check, cube_sat.face_orientations)
check_angles(cube_sat.face_orientations, "test 11")


### test 90 degrees along -x and +z
cube_sat = CubeSat()
cube_sat.main_rotation.angle = 0
cube_sat.precession.angle = 0
cube_sat.additional_rotations = [
    AxisAngle(np.array([-1, 0, 0]), 90),
    AxisAngle(np.array([0, 0, 1]), 90),
]
cube_sat.step()
orientations_check = {
    "front": np.array([0, 1, 0]),  # -> front -> right
    "back": np.array([0, -1, 0]),  # -> back -> left
    "right": np.array([0, 0, -1]),  # -> bottom -> bottom
    "left": np.array([0, 0, 1]),  # -> top -> top
    "top": np.array([-1, 0, 0]),  # -> right -> back
    "bottom": np.array([1, 0, 0]),  # -> left -> front
}
assert dicts_equal(orientations_check, cube_sat.face_orientations), (
    f"should be:\n{orientations_check}\nwas:\n{cube_sat.face_orientations}"
)
check_angles(cube_sat.face_orientations, "test 12")


##################################################
### combine all
##################################################

### precess -z around +y 90 degrees to get -x main axis
### then rotate +y 90, -z 90 and -y 90 to get back to the original position
cube_sat = CubeSat()
cube_sat.main_rotation.angle = 90
cube_sat.main_rotation.axis = np.array([0, 0, -1])
cube_sat.precession.angle = 90
cube_sat.precession.axis = np.array([0, 1, 0])
cube_sat.additional_rotations = [
    AxisAngle(np.array([0, 1, 0]), 90),
    AxisAngle(np.array([0, 0, -1]), 90),
    AxisAngle(np.array([0, -1, 0]), 90),
]

cube_sat.step()
orientations_check = np.array([-1, 0, 0])  # -z -> -x
assert np.allclose(orientations_check, cube_sat.main_rotation.axis)

# -x, +y , -z, -y
orientations_check = {
    "front": np.array([1, 0, 0]),
    "back": np.array([-1, 0, 0]),
    "right": np.array([0, 1, 0]),
    "left": np.array([0, -1, 0]),
    "top": np.array([0, 0, 1]),
    "bottom": np.array([0, 0, -1]),
}
assert dicts_equal(orientations_check, cube_sat.face_orientations)
check_angles(cube_sat.face_orientations, "test 13")


print("All rotational tests passed.")


##################################################
##################################################
### thermal system tests
##################################################
##################################################

### test temps +sun are always higher than -sun, sides are not the same as other faces, and check symmetric behavior
# rotate so front and bottom are facing sun, back and top never
cube_sat = CubeSat()
cube_sat.main_rotation.axis = np.array([0, -1, 0])
cube_sat.main_rotation.angle = 45
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []
cube_sat.step()
cube_sat.main_rotation.angle = 0

cube_sat.temperatures = {face: 20.0 for face in config.base_face_orientations}

while cube_sat.simulation.orbit < 50:
    # keep heater from running
    cube_sat.battery_charge = 0
    cube_sat.step()
    assert cube_sat.temperatures["front"] > cube_sat.temperatures["back"], (
        f"iteration {cube_sat.simulation.orbit * config.orbital_period + cube_sat.simulation.current_time}, temp1={cube_sat.temperatures['front']}, temp2={cube_sat.temperatures['front']}"
    )
    assert cube_sat.temperatures["bottom"] > cube_sat.temperatures["top"], (
        f"iteration {cube_sat.simulation.orbit * config.orbital_period + cube_sat.simulation.current_time}, temp1={cube_sat.temperatures['bottom']}, temp2={cube_sat.temperatures['bottom']}"
    )
    assert np.isclose(
        cube_sat.temperatures["front"], cube_sat.temperatures["bottom"]
    ), (
        f"iteration {cube_sat.simulation.orbit * config.orbital_period + cube_sat.simulation.current_time}, temp1={cube_sat.temperatures['front']}, temp2={cube_sat.temperatures['front']}"
    )
    assert np.isclose(cube_sat.temperatures["back"], cube_sat.temperatures["top"]), (
        f"iteration {cube_sat.simulation.orbit * config.orbital_period + cube_sat.simulation.current_time}, temp1={cube_sat.temperatures['back']}, temp2={cube_sat.temperatures['top']}"
    )
    assert np.isclose(cube_sat.temperatures["left"], cube_sat.temperatures["right"]), (
        f"iteration {cube_sat.simulation.orbit * config.orbital_period + cube_sat.simulation.current_time}, temp1={cube_sat.temperatures['left']}, temp2={cube_sat.temperatures['right']}"
    )


### test temperature indicators when temps above limit
cube_sat = CubeSat()
cube_sat.temperatures = {
    face: (config.temp_face_high_limit + 50) for face in config.base_face_orientations
}
cube_sat.center_temperature = config.temp_center_high_limit + 50
cube_sat.step()

assert cube_sat.indicator.center_temp_range == 1
assert cube_sat.indicator.faces_temp_range == 1


### test temperature indicators when temps below limit
cube_sat = CubeSat()
cube_sat.temperatures = {
    face: (config.temp_face_low_limit - 50) for face in config.base_face_orientations
}
cube_sat.center_temperature = config.temp_center_low_limit - 50
cube_sat.step()

assert cube_sat.indicator.center_temp_range == -1
assert cube_sat.indicator.faces_temp_range == -1


print("All thermal system tests passed.")


##################################################
##################################################
### power system tests
##################################################
##################################################


### test battery stays 0 and heater never runs if pointing away from the sun
cube_sat = CubeSat()
# rotate to look away from the sun
cube_sat.main_rotation.axis = np.array([0, -1, 0])
cube_sat.main_rotation.angle = 90
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []
cube_sat.step()

assert np.allclose(cube_sat.face_orientations["top"], np.array([-1, 0, 0]))
cube_sat.battery_charge = 0
cube_sat.heater_runtime = 0
cube_sat.main_rotation.angle = 0

for i in range(20 * config.orbital_period):
    cube_sat.step()
    assert cube_sat.battery_charge == 0, (
        f"i={i}, battery charge={cube_sat.battery_charge}"
    )
    assert cube_sat.heater_runtime == 0


### test battery stays at 100 when always (no orbital change) facing the sun
cube_sat = CubeSat()
# rotate to look towards the sun
cube_sat.main_rotation.axis = np.array([0, 1, 0])
cube_sat.main_rotation.angle = 90
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []
cube_sat.step()

assert np.allclose(cube_sat.face_orientations["top"], np.array([1, 0, 0]))
cube_sat.battery_charge = 0
cube_sat.main_rotation.angle = 0

reset = config.heater_allowed
config.heater_allowed = False
for i in range(20 * config.orbital_period):
    battery_before = cube_sat.battery_charge
    cube_sat.step()
    cube_sat.simulation.current_time = 0
    assert cube_sat.battery_charge != 0
    assert cube_sat.battery_charge > battery_before or cube_sat.battery_charge == 100, (
        f"iteration={i}, battery was: {battery_before}, is now:{cube_sat.battery_charge}"
    )

config.heater_allowed = reset
print("All power system tests passed.")


##################################################
##################################################
### heater tests
##################################################
##################################################
reset = config.heater_allowed
config.heater_allowed = True


### test heater works when below temp limit
cube_sat = CubeSat()
cube_sat.temperatures = {
    face: (config.heater_limit - 100) for face in config.base_face_orientations
}
cube_sat.center_temperature = config.heater_limit - 100
# put satellite in shadow so no power gain
cube_sat.simulation.current_time = config.sun_illumination_time + 1
battery_before = cube_sat.battery_charge
heater_runtime_before = cube_sat.heater_runtime

for i in range(10):
    cube_sat.step()

    assert heater_runtime_before + 1 == cube_sat.heater_runtime
    assert cube_sat.indicator.heater_running == 1
    assert np.isclose(
        battery_before - config.power_drawn_low - config.heater_power,
        cube_sat.battery_charge,
    )

    battery_before = cube_sat.battery_charge
    heater_runtime_before = cube_sat.heater_runtime


### test heater does nothing, followup with same satellite instance when no temp is below temp limit
cube_sat.temperatures = {
    face: (config.heater_limit + 30) for face in config.base_face_orientations
}
cube_sat.center_temperature = config.heater_limit + 30
battery_before = cube_sat.battery_charge
heater_runtime_before = cube_sat.heater_runtime
for i in range(10):
    cube_sat.step()

    assert heater_runtime_before == cube_sat.heater_runtime, (
        f"i={i}, before={heater_runtime_before}, now={cube_sat.heater_runtime}"
    )
    assert cube_sat.indicator.heater_running == 0
    assert np.isclose(
        battery_before - config.power_drawn_low,
        cube_sat.battery_charge,
    )

    battery_before = cube_sat.battery_charge

config.heater_allowed = reset
print("All heater tests passed.")


##################################################
##################################################
### sun angles tests
##################################################
##################################################

### test sun angles in default position
cube_sat = CubeSat()
cube_sat.main_rotation.angle = 0
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []
angles_check = {
    "front": 1,
    "back": -1,
    "right": 0,
    "left": 0,
    "top": 0,
    "bottom": 0,
}
for i in range(10):
    cube_sat.step()
    assert dicts_equal(angles_check, cube_sat.sun_angles), (
        f"failed in iteration {i}\nexpected={angles_check}\ngot={cube_sat.sun_angles}"
    )


### test sun angles with four non-zero faces
# rotate so front and bottom are facing sun, back and top never
cube_sat = CubeSat()
cube_sat.main_rotation.angle = 45
cube_sat.main_rotation.axis = np.array([0, -1, 0])
cube_sat.precession.angle = 0
cube_sat.additional_rotations = []
angles_check = {
    "front": 0.70710678,
    "back": -0.70710678,
    "right": 0,
    "left": 0,
    "top": -0.70710678,
    "bottom": 0.70710678,
}
cube_sat.step()
assert dicts_equal(angles_check, cube_sat.sun_angles), (
    f"failed before iteration\nexpected={angles_check}\ngot={cube_sat.sun_angles}"
)

cube_sat.main_rotation.angle = 0

for i in range(10):
    cube_sat.step()
    assert dicts_equal(angles_check, cube_sat.sun_angles), (
        f"failed in iteration {i}\nexpected={angles_check}\ngot={cube_sat.sun_angles}"
    )


### test sun angles with all non-zero faces
# first rotate -y so front and bottom are facing sun, then rotate around +z -> three facing sun, other three not
cube_sat = CubeSat()
cube_sat.main_rotation.angle = 0
cube_sat.precession.angle = 0
cube_sat.additional_rotations = [
    AxisAngle(axis=np.array([0, -1, 0]), angle=45),
    AxisAngle(axis=np.array([0, 0, 1]), angle=45),
]
angles_check = {
    "front": 0.5,
    "back": -0.5,
    "right": -0.70710678,
    "left": 0.70710678,
    "top": -0.5,
    "bottom": 0.5,
}
cube_sat.step()
assert dicts_equal(angles_check, cube_sat.sun_angles), (
    f"failed before iteration\nexpected={angles_check}\ngot={cube_sat.sun_angles}"
)

cube_sat.additional_rotations = []

for i in range(10):
    cube_sat.step()
    assert dicts_equal(angles_check, cube_sat.sun_angles), (
        f"failed in iteration {i}\nexpected={angles_check}\ngot={cube_sat.sun_angles}"
    )


print("All sun angles tests passed.")


##################################################
##################################################
### data set tests
##################################################
##################################################

### test input checker
with pytest.raises(ValueError):
    data = get_simulated_data(0)

with pytest.raises(TypeError):
    data = get_simulated_data(1.0)


### test correct amount of rows with 1 orbit
repetitions = 1
check_data = (len(config.titles), config.orbital_period * repetitions + 1)
with contextlib.redirect_stdout(io.StringIO()):
    data = get_simulated_data(repetitions)
assert isinstance(data, np.ndarray)
assert check_data == np.shape(data), f"expected: {check_data}\ngot:{np.shape(data)}"

### test correct amount of rows with 9 orbits
repetitions = 9
check_data = (len(config.titles), config.orbital_period * repetitions + 1)
with contextlib.redirect_stdout(io.StringIO()):
    data = get_simulated_data(repetitions)
assert isinstance(data, np.ndarray)
assert check_data == np.shape(data), f"expected: {check_data}\ngot:{np.shape(data)}"


print("All data set tests passed.")


print("\nAll tests passed.")
