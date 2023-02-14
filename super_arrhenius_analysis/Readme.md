## This folder contains codes for super-Arrhenius analysis.

* `disentanglement`: estimate disentanglement time of structure `gain-of-non-native-entanglement`
* `unfold`: estimate unfolding time of structure `loss-of-native-entanglement`

## How to run:

1) To get the disentangled time: `python bootstrapping_disentanglement.py`

2) To get the unfolding time: `python bootrapping_unfold_Q_XXX.py` (where _XXX is the theshold to defined unfolded state)

This command generates `datapoints.dat` and `survival_{TEMP}.dat`.

`datapoints.dat` contains disentanglement/unfolding rate and 95% confidence interval.
3 last rows contain information about the extrapolated time at 298 K
For testing, please change variable `nboots` to a small value. In the papers, we use `nboots=10000` iterations.



2) `python curve_fit.py`

This command requires survival probability data from the previous command to plot data using Matplotlib. 

`survival_{TEMP}.dat` is survival probability at each temperature, used to plot survival probability.

The first col is the time in unit of ns, the second col is the probability of entangled state.
