This is the repository for my Bachelors Thesis "Time Series Analysis and Prediction of CubeSat Telemetry Data Using Dynamic Mode Decomposition" at the Technical University of Munich (TUM).

It contains not only the plots used in the thesis but all plots that were created and the raw results. Of course, all scripts are included as well (simulation-related starting with `sim` and analysis or plotting-related with `dmd-analysis`). You can also find the defined data sets as numpy files (`sim_data.zip`).

To use the scripts, just execute the following commands in the root of the repository. This will set up a `.venv` environment and install the necessary dependencies (can be found in `pyproject.toml`) to run the scripts.

`python -m venv .venv`

`source .venv/bin/activate`

`pip install -e .`

This README assumes you are running the scripts from within VSCode. If not, then you might have to adapt the structure accordingly.

Using VSCode, just open the repo folder and verify the correct `.venv`: Open Command Palette -> “Python: Select Interpreter” -> pick the .venv created in the repo from just before

If you want to run the scripts without creating new data sets, just unzip the data-set files (currently, all scripts look for those in `sim_data/`). If you produce your own data-sets, you need to adapt the analysis scripts accordingly. Currently, only `sim_plotting.py` is configured to create or update the numpy files.

The simulation can be found in `sim_modularized.py` and offers the function `get_simulated_data(simulated_orbits: int) -> np.ndarray` if you would want to create your own data-sets from other scripts.
