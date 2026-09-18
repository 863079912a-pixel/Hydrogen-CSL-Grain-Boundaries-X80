# Code inventory

This file summarizes the executable and post-processing files currently organized in the repository.

| Repository file | Role | Coverage / purpose |
|---|---|---|
| `01_equilibrium_tensile/run_equilibrium_tensile_all_cases.py` | Equilibrium tensile simulation driver | Σ3, Σ5, Σ11, Σ17; 0H, 1H, 5%H, 10%H, 15%H, 20%H, 25%H, 50%H, 100%H |
| `01_equilibrium_tensile/run_equilibrium_tensile.bat` | Windows launcher | Launches the organized equilibrium Python driver |
| `02_nonequilibrium_tensile/run_nonequilibrium_tensile_0H_1H.py` | Non-equilibrium tensile/fracture simulation | 0H and 1H; four grain boundaries |
| `02_nonequilibrium_tensile/run_nonequilibrium_tensile_5_20H.py` | Non-equilibrium tensile/fracture simulation | 5%H, 10%H, 15%H, 20%H; four grain boundaries |
| `02_nonequilibrium_tensile/run_nonequilibrium_tensile_25_100H.py` | Non-equilibrium tensile/fracture simulation | 25%H, 50%H, 100%H; four grain boundaries |
| `03_postprocessing/equilibrium/collect_equilibrium_strain_stress_csv.m` | Equilibrium CSV collection | Collects `strain_stress.csv` files from case directories |
| `03_postprocessing/nonequilibrium/plot_dynamic_stress_strain_all_conditions.m` | Dynamic stress–strain plotting | Reads the 36 repository non-equilibrium CSVs and plots grain-boundary / hydrogen-condition comparisons |
| `04_fracture_criterion/run_interpretable_threshold_ml_analysis.py` | Interpretable fracture-threshold analysis | Feature construction, single-indicator threshold scanning, classification metrics, LOOCV, optional bootstrap/logistic analysis, and plotting |
| `06_tools/collect_dynamic_stress_strain.m` | Dynamic CSV collection utility | Collects and renames the 36 dynamic stress–strain CSV files from workstation result directories |

## Runtime configuration

The LAMMPS simulation drivers and collection utilities retain workstation-specific path blocks so that their original execution settings remain traceable. These path variables should be edited before rerunning on another computer.

The dynamic plotting script is repository-relative and reads the compact non-equilibrium CSV files from `05_stress_strain_data/nonequilibrium/` using the columns `strain_nominal` and `stress`.

## Case-selection settings

The organized equilibrium driver enumerates all four grain boundaries and all nine manuscript hydrogen conditions. The organized 5–20%H non-equilibrium driver is configured for all four grain boundaries with a fresh-run mode. These public-facing case selections preserve the underlying loading and analysis algorithms while exposing the complete manuscript matrix.

## Python syntax check

All four Python scripts in the repository pass Python byte-code compilation. The equilibrium driver may emit a documentation-string warning on some Python versions because a Windows path appears in its module docstring; this does not prevent compilation or execution.
