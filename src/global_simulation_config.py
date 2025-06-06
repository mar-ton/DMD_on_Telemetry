import numpy as np

orbital_period: int = 93

sun_direction = np.array([1, 0, 0])
sun_illumination_time: int = 62

# 2021: Morsch Filho et al, Simulation of a CubeSat...
# 2015: \Cengel et al, Heat and Mass Transfer...
solar_flux = 1367.0
earth_flux = 237.0
absorptivity = 0.6
emissivity = 0.6
deep_space_temp = -270.45
thermal_conductance = 100.0
thermal_conductance_center = 23.0

thermal_capacitance = 960 * 0.9
thermal_capacitance_center = 1110 * 0.073

area_per_side = 0.01

power_per_top_solar_panel = 4.0
power_per_side_solar_panel = 2.8

power_drawn_high = 4.6
power_drawn_low = 0.5
power_efficiency = 0.7

heater_allowed = True
heater_heat = 10
heater_power = 2
heater_limit = 0

temp_face_low_limit = -40
temp_face_high_limit = 75

temp_center_low_limit = -10
temp_center_high_limit = 40

dt = 1

base_face_orientations = {
    "front": np.array([1, 0, 0]),
    "back": np.array([-1, 0, 0]),
    "right": np.array([0, 1, 0]),
    "left": np.array([0, -1, 0]),
    "top": np.array([0, 0, 1]),
    "bottom": np.array([0, 0, -1]),
}

titles = [
    "total_time",  # 0
    "current_time_in_orbit",  # 1
    "battery_level",  # 2
    "temperature_front",  # 3
    "temperature_back",  # 4
    "temperature_right",  # 5
    "temperature_left",  # 6
    "temperature_top",  # 7
    "temperature_bottom",  # 8
    "temperature_center",  # 9
    "top_orientation_x",  # 10
    "top_orientation_y",  # 11
    "top_orientation_z",  # 12
    "front_orientation_x",  # 13
    "front_orientation_y",  # 14
    "front_orientation_z",  # 15
    "combined_angular_velocity",  # 16
    "sum_rotations_x",  # 17
    "sum_rotations_y",  # 18
    "sum_rotations_z",  # 19
    "sun_angle_front",  # 20
    "sun_angle_back",  # 21
    "sun_angle_right",  # 22
    "sun_angle_left",  # 23
    "sun_angle_top",  # 24
    "sun_angle_bottom",  # 25
    "center_temp_range",  # 26
    "face_temp_range",  # 27
    "heater_running",  # 28
]



# for legacy models
radiation_absorbed = 9.0
radiation_emitted = 0.1
radiation_emitted_to_deep_space = 0.4
