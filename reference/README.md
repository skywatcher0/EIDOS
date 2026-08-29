# Table 5 reference

`table5_eidos_reference.csv` contains the three EIDOS rows transcribed from the paper source used to build Table 5. It is read only after a fresh run, when `reproduce.py` checks identifiers, formulas, and displayed semantic NRMSE values.

The experiment never reads this file while generating observations, constructing the candidate bank, selecting a candidate, or fitting coefficients.
