import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp
from scipy.constants import Stefan_Boltzmann
import global_simulation_config as config


@dataclass
class SimulationState:
    total_time: int = 0
    current_time: int = 0
    orbit: int = 0


@dataclass
class AxisAngle:
    axis: np.ndarray
    angle: float

    def get_rotation(self) -> Rotation:
        return Rotation.from_rotvec(np.deg2rad(self.angle) * self.axis)


@dataclass
class Indicator:
    center_temp_range: int = 0
    faces_temp_range: int = 0
    heater_running: int = 0


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _get_np_array_from_dict(dict: dict) -> np.ndarray:
    return np.array(list(dict.values()))


class CubeSat:
    def __init__(self):
        self.orientation = Rotation.identity()
        self.face_orientations = config.base_face_orientations

        self.main_rotation = AxisAngle(_normalize(np.array([0, -0.5, -1])), 50)
        self.precession = AxisAngle(_normalize(np.array([0, -1, 1])), 10)

        self.additional_rotations = [
            AxisAngle(_normalize(np.array([-1, 1, -1])), 120),
            AxisAngle(_normalize(np.array([0.1, -0.2, 0.3])), 20),
        ]

        self.current_combined_rotation = None
        self._update_current_combined_rotation()

        self.temperatures = {face: 0.0 for face in config.base_face_orientations}
        self.center_temperature = 0.0

        self.sun_angles = {face: 0.0 for face in config.base_face_orientations}
        self._update_sun_angles()

        self.solar_panels_per_face = {
            face: (1, config.power_per_side_solar_panel)
            for face in config.base_face_orientations
        }
        self.solar_panels_per_face["top"] = (4, config.power_per_top_solar_panel)
        self.solar_panels_per_face["bottom"] = (0, 0)

        self.battery_charge = 100.0

        self.heater_runtime: int = 0

        self.indicator = Indicator()

        self.simulation = SimulationState()

    def step(self):
        self._rotate()
        self._update_face_orientations()

        self._update_sun_angles()
        self._update_power_state()
        self._update_thermal_state()
        self._update_thermal_indicators()

        self._update_simulation()

    def _rotate(self):
        self._update_main_rotation_axis()
        self._update_current_combined_rotation()
        self.orientation = self.current_combined_rotation * self.orientation

    def _update_main_rotation_axis(self):
        self.main_rotation.axis = self.precession.get_rotation().apply(
            self.main_rotation.axis
        )

    def _update_current_combined_rotation(self):
        combined = Rotation.identity()

        for rotation in reversed(self.additional_rotations):
            combined *= rotation.get_rotation()

        self.current_combined_rotation = combined * self.main_rotation.get_rotation()

    def _update_face_orientations(self):
        self.face_orientations = {
            key: self.orientation.apply(vec)
            for key, vec in config.base_face_orientations.items()
        }

    def _update_sun_angles(self):
        for face_name, face_vector in self.face_orientations.items():
            self.sun_angles[face_name] = np.dot(face_vector, config.sun_direction)

    def _update_power_state(self):
        # in shadow
        if self._in_shadow_of_planet():
            self.battery_charge -= config.power_drawn_low

        # in sunlight
        else:
            self.battery_charge -= config.power_drawn_high

            for face_name, face_orientation in self.face_orientations.items():
                (num_panels, power_output) = self.solar_panels_per_face[face_name]
                if num_panels != 0:
                    angle = self.sun_angles[face_name]
                    if angle > 0:
                        self.battery_charge += angle * num_panels * power_output

        # keep between 0 and 100
        self.battery_charge = min(100, max(0, self.battery_charge))

    def _update_thermal_state(self):
        faces_emission = self._get_emission()
        faces_absorption = self._get_absorption()
        faces_radiative_delta = faces_emission + faces_absorption

        center_delta = self._get_internal_heat_delta()

        self._apply_thermal_deltas(
            np.concatenate((faces_radiative_delta, [center_delta]))
        )

    def _get_emission(self) -> np.ndarray:
        return (
            config.emissivity
            * Stefan_Boltzmann
            * config.area_per_side
            * (
                _get_np_array_from_dict(self.temperatures) ** 4
                - config.deep_space_temp**4
            )
        )

    def _get_absorption(self) -> np.ndarray:
        # in shadow -> only heat source is Earth
        if self._in_shadow_of_planet():
            return (
                config.earth_flux
                * config.area_per_side
                * config.absorptivity
                * np.maximum(_get_np_array_from_dict(self.sun_angles), 0)
            )
        # in sunlight -> heat source is Sun for +sun direction faces and Earth for -sun direction faces
        else:
            return (
                config.solar_flux
                * config.area_per_side
                * config.absorptivity
                * np.maximum(_get_np_array_from_dict(self.sun_angles), 0)
            ) + (
                config.earth_flux
                * config.area_per_side
                * config.absorptivity
                * np.abs(np.minimum(_get_np_array_from_dict(self.sun_angles), 0))
            )

    def _get_internal_heat_delta(self) -> float:
        # model heat from electronics
        if self._in_shadow_of_planet():
            center_delta = config.power_drawn_low * (1 - config.power_efficiency)
        else:
            center_delta = config.power_drawn_high * (1 - config.power_efficiency)

        self.indicator.heater_running = 0

        # model heater
        if (
            config.heater_allowed
            and self.center_temperature < config.heater_limit
            and self.battery_charge > config.heater_power
        ):
            center_delta += config.heater_heat
            self.heater_runtime += 1
            self.battery_charge -= config.heater_power
            self.indicator.heater_running = 1

        return center_delta

    def _apply_thermal_deltas(self, temps_delta: np.ndarray):
        # since we integrate over one step, _ is added only to adhere to scipy module
        def thm_deriv(_, temps: np.ndarray) -> np.ndarray:
            T_faces = temps[:6]
            T_center = temps[6]
            dT_faces = temps_delta[:6] + (
                (T_center - T_faces)
                * config.thermal_conductance
                / config.thermal_capacitance
            )
            dT_center = temps_delta[6] + (
                (np.sum(T_faces - T_center))
                * config.thermal_conductance_center
                / config.thermal_capacitance_center
            )
            return np.concatenate((dT_faces, [dT_center]))

        # use Runge-Kutta RK45
        t_span = (0, config.dt)
        y0 = np.concatenate(
            (_get_np_array_from_dict(self.temperatures), [self.center_temperature])
        )
        solution = solve_ivp(thm_deriv, t_span, y0, method="RK45")

        self.temperatures = dict(zip(self.temperatures.keys(), (solution.y[:, -1])[:6]))
        self.center_temperature = (solution.y[:, -1])[6]

    def _update_thermal_indicators(self):
        if np.any(
            _get_np_array_from_dict(self.temperatures) > config.temp_face_high_limit
        ):
            self.indicator.faces_temp_range = 1
        elif np.any(
            _get_np_array_from_dict(self.temperatures) < config.temp_face_low_limit
        ):
            self.indicator.faces_temp_range = -1
        else:
            self.indicator.faces_temp_range = 0

        if self.center_temperature > config.temp_center_high_limit:
            self.indicator.center_temp_range = 1
        elif self.center_temperature < config.temp_center_low_limit:
            self.indicator.center_temp_range = -1
        else:
            self.indicator.center_temp_range = 0

    def _update_simulation(self):
        self.simulation.total_time += 1
        self.simulation.current_time += 1
        if self.simulation.current_time >= config.orbital_period:
            self.simulation.current_time = 0
            self.simulation.orbit += 1

    def _in_shadow_of_planet(self) -> bool:
        return self.simulation.current_time >= config.sun_illumination_time


def get_simulated_data(simulated_orbits: int) -> np.ndarray:
    if not isinstance(simulated_orbits, int):
        raise TypeError(f"must enter an integer, but entered: {simulated_orbits}")
    if simulated_orbits < 1:
        raise ValueError(f"orbit number can't be smaller than 1: {simulated_orbits}")

    cube_sat = CubeSat()
    data = []

    while cube_sat.simulation.orbit < simulated_orbits:
        data.append(
            [
                cube_sat.simulation.total_time,
                cube_sat.simulation.current_time,
                cube_sat.battery_charge,
                *list(cube_sat.temperatures.values()),
                cube_sat.center_temperature,
                *cube_sat.face_orientations["top"],
                *cube_sat.face_orientations["front"],
                (
                    np.linalg.norm(
                        cube_sat.current_combined_rotation.as_rotvec(degrees=True)
                    )
                ),
                *_normalize(cube_sat.current_combined_rotation.as_rotvec()),
                *list(cube_sat.sun_angles.values()),
                cube_sat.indicator.center_temp_range,
                cube_sat.indicator.faces_temp_range,
                cube_sat.indicator.heater_running,
            ]
        )
        cube_sat.step()

    data = np.array(data).T

    print(f"Orbit {cube_sat.simulation.orbit} done")
    print(f"Orientation: {cube_sat.face_orientations['top']}")
    print(
        "Temperature:",
        ", ".join(f"{k}: {v:.2f}" for k, v in cube_sat.temperatures.items()),
    )
    print(f"Battery Charge: {cube_sat.battery_charge}%")
    print(f"Heater Runtime: {cube_sat.heater_runtime}")
    print(f"max angle={np.max(data[16]):.2f}")
    print(f"avg angle={np.mean(data[16]):.2f}")
    print(f"min angle={np.min(data[16]):.2f}")

    data = np.concatenate((np.array(config.titles).reshape(-1, 1), data), axis=1)
    return data
