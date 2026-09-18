# Interpretable fracture-criterion analysis

This directory contains the interpretable threshold / machine-learning workflow used in the manuscript to identify a quantitative discriminator for grain-boundary-localized cracking.

## Contents

- `run_interpretable_threshold_ml_analysis.py`: main analysis script.
- `outputs/`: directory reserved for generated tables, summaries, and figures.

## Analysis workflow

The main script implements the following analysis steps:

1. construction and merging of per-condition indicators and fracture-mode labels;
2. single-feature threshold scanning in both `greater` and `less` directions;
3. calculation of classification metrics including F1 score, accuracy, recall, specificity, and AUC;
4. leave-one-out cross-validation (LOOCV) for threshold stability;
5. optional bootstrap resampling and logistic-regression analysis;
6. export of analysis tables and publication-oriented figures.

The principal physical indicator emphasized in the manuscript is `delta_gamma_excess_pre_drop`, representing the pre-instability grain-boundary shear-localization excess under non-equilibrium loading.

The analysis covers the four grain boundaries Σ3, Σ5, Σ11, and Σ17 across the nine hydrogen conditions 0H, 1H, 5%H, 10%H, 15%H, 20%H, 25%H, 50%H, and 100%H, corresponding to 36 grain-boundary / hydrogen-coverage conditions.
