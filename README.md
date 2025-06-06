This is the repository for my Bachlors Thesis "Time Series Analysis and Prediction of CubeSat Telemetry Data Using Dynamic Mode Decomposition".

It contains not only the used plots in the thesis but all different plots and raw data. Of course all scipts (simulation starting with `sim` and analysis and plotting with `dmd-analysis`) are also included. You can also find the defined data sets as numpy files (`sim_data.zip`).

To use the scripts, set up a `.venv` environment in the root of the repository. The following depencies are necesseary to run the scripts and are installed into `.venv` when following the rest of this README:
- numpy
- scipy
- matplotlib
- pydmd
- pytest
- pyqt5 (to show the plots)

* `python -m venv .venv`
* `source .venv/bin/activate`
* `pip install -e .`

This README assumes you are running the scripts from within VSCode. If not, then you have to adapt the structure accordingly.

Using VSCode, just open the repo folder and verify the correct `.venv`: Open Command Palette -> “Python: Select Interpreter” -> pick the .venv created in the repo from just before

If you want to run the scripts without creating new data sets, just unzip the data-set files (currently, all scripts look for those in `sim_data/`). If you produce your own data sets, you need to adapt the analysis scripts accordingly.
