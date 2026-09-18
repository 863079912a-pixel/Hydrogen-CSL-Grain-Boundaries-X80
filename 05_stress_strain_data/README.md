# Stress–strain data

This directory contains the stress–strain datasets used for the manuscript figures and quantitative comparisons.

## Coverage

Both loading modes contain 36 CSV files corresponding to four grain boundaries (Σ3, Σ5, Σ11, and Σ17) and nine hydrogen conditions (0H, 1H, 5%H, 10%H, 15%H, 20%H, 25%H, 50%H, and 100%H).

## Equilibrium data

Directory: `equilibrium/`

File naming example:

`sigma3_GB_025cov_strain_stress.csv`

Columns:

- `step`: strain-step index;
- `strain`: engineering strain;
- `stress_raw_gpa`: stress series retained for the manuscript equilibrium stress–strain curves, in GPa;
- `pe_eV`: potential energy, in eV;
- `ly_A`: simulation-cell length in the loading direction, in Å;
- `pyy_bar`: y-direction pressure/stress quantity written by the simulation workflow, in bar.

## Non-equilibrium data

Directory: `nonequilibrium/`

File naming example:

`Sigma3_25H_stress_strain_dynamic.csv`

Columns:

- `chunk`: dynamic loading chunk index;
- `time_ps`: simulation time, in ps;
- `opening_A`: imposed grip opening, in Å;
- `strain_nominal`: nominal tensile strain;
- `stress`: finalized tensile-stress series used for the manuscript dynamic stress–strain curves, in GPa.

The files are compact manuscript-ready tables intended for direct plotting and comparison. The MATLAB script `03_postprocessing/nonequilibrium/plot_dynamic_stress_strain_all_conditions.m` reads `strain_nominal` and `stress` directly from these files.
