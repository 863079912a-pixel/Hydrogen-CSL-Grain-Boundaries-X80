# Hydrogen-Induced Degradation and Fracture Localization in α-Fe CSL Grain Boundaries

Data and computational code associated with the manuscript:

**Quantitative Characterization of Hydrogen-Induced Mechanical Degradation and Fracture Localization in α-Fe CSL Grain Boundaries Relevant to X80 Pipeline Steel**

Authors: An Li, Guangyuan Weng, Zhaoyang Han, Zhuosong Zhao, Wenhong Tang, Lei Zhu, and Tianhao Wang.

## Public repository contents

This repository provides:

- equilibrium (quasi-static) tensile simulation scripts;
- non-equilibrium (dynamic) tensile simulation scripts;
- MATLAB post-processing utilities;
- the Python workflow used for the interpretable fracture-threshold analysis;
- equilibrium stress–strain data;
- non-equilibrium stress–strain data.

The tensile data cover four representative grain boundaries:

- Σ3(112)[110]
- Σ5(210)[001]
- Σ11(332)[110]
- Σ17(410)[001]

and nine hydrogen conditions:

- 0H
- 1H
- 5%H
- 10%H
- 15%H
- 20%H
- 25%H
- 50%H
- 100%H

Thus, 36 stress–strain datasets are provided for each loading mode.

## Repository structure

```text
01_equilibrium_tensile/
02_nonequilibrium_tensile/
03_postprocessing/
04_fracture_criterion/
05_stress_strain_data/
    equilibrium/
    nonequilibrium/
06_tools/
07_docs/
README.md
DATA_AVAILABILITY.md
CITATION.cff
requirements.txt
```

## Stress–strain data

The manuscript stress–strain datasets are stored in:

- `05_stress_strain_data/equilibrium/`
- `05_stress_strain_data/nonequilibrium/`

Each folder contains 36 CSV files covering the four grain boundaries and nine hydrogen conditions listed above. Column definitions and naming conventions are described in `05_stress_strain_data/README.md`.

## Fracture-threshold analysis

The `04_fracture_criterion/` directory contains the interpretable threshold / machine-learning analysis script used to identify the grain-boundary-localized cracking discriminator discussed in the manuscript, together with its output directory structure.

## Software and path configuration

The simulation drivers are written in Python and launch LAMMPS calculations. The post-processing utilities are written in MATLAB. The fracture-threshold workflow uses Python packages listed in `requirements.txt`.

Some simulation and collection scripts retain workstation-specific Windows path variables from the calculation environment. Before rerunning a calculation on another computer, edit the path/settings block near the beginning of the corresponding script. The dynamic plotting script in `03_postprocessing/nonequilibrium/` reads the repository CSV files directly.

## Large simulation outputs

Large molecular-dynamics trajectory, dump, and restart files are omitted from the repository because of their size.

## Corresponding author

Prof. Guangyuan Weng  
School of Mechanical Engineering, Xi'an Shiyou University  
Xi'an 710065, Shaanxi, China  
E-mail: weng_guangyuan@163.com
