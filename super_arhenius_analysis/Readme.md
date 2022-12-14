## This folder contains codes for super-Arrhenius analysis.

* `disentanglement`: estimate disentanglement time of structure `gain-of-non-native-entanglement`
* `unfold`: estimate unfolding time of structure `loss-of-native-entanglement`

## How to run:
1) `python bootstrapping_disentanglement.py`

This command generates `datapoints.dat` and `survival_{TEMP}.dat`.
`datapoints.dat` is disentanglement/unfolding rate and 95% confidence interval.
For testing, please change variable `nboots` to a small value. In the papers, we use `nboots=10000` iterations.

3) `python curve_fit.py`

This command requires survival probability data from the previous command to plot data using Matplotlib. 

`survival_{TEMP}.dat` is survival probability at each temperature, used to plot survival probability