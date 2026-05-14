# Adaptive Estimation of Aggregated Values of Conditional Linear Programs
This repository includes the replication package of the empirical section and simulation exercise of the paper "Adaptive Estimation of Aggregated Values of Conditional Linear Programs"

We use the Jobs First dataset in line with Kline and Tartari (2016). 
KT use a specific cleaning
procedure that results in a **4,461-person** sample and their Table 4
reports the proportions in each latent type. `distribution_over_states.py`
replicates their data pipeline in Python and arrives at the same sample,
and the same file also replicates KT's Table 4 (state distributions and
bootstrap SEs). `replicating5.py` then reproduces KT's Table 5 (analytical
linprog bounds on flow probabilities) using `Table4_mat_python.txt` as
input. These steps were necessary to make sure we were working on the
same sample.

The remaining files implement the CLP estimator, first on synthetic data
where the ground truth is known, then on the real JF sample at the coarse
5×9 design (Table 5 of the paper), then at increasingly granular designs
(Appendix), and finally for welfare-gain bounds.


