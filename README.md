# tripm

Standalone Octave/Matlab programs for Linear Transportation Problems (LTPs). Can handle quite large dense LTPs. An example application is its usage for computation of the Earth-movers distance.

The LTP is $$\min_{X} \sum_{i,j} c_{i,j}x_{i,j}$$ subject to $$x_{i,j}\geq 0$$ $$\sum_i x_{i,j} = d_j$$ and $$\sum_j x_{i,j} = s_i$$ The problem can be rectangular, and it is expected that $$\sum_j d_j = \sum_i s_i$$.

Call `test_pdipmltp_randlap` in `Octave` for a quick health-check.
