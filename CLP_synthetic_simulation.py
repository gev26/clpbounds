"""
CLP_synthetic_simulation.py
===========================
Run CLP_final_group's full pipeline on SYNTHETIC data and compare to the
KT-style analytical bounds.

Why this file exists
--------------------
Validation/sanity check.  Real JF data has the truth hidden, so we cannot
verify the CLP bounds against ground truth.  Here we generate synthetic data
with KNOWN latent flows beta_0 and verify that:
  1. The CLP plug-in (cross-fit LASSO + per-i LP + multiplier bootstrap)
     produces bounds that bracket the true beta_0.
  2. The CLP bounds are tighter than the analytical bounds (per the CLP
     paper's Jensen-inequality argument: aggregating per-i LP bounds gives
     tighter than population-level LP).
  3. The first stage is informative (b_hat(X) tracks the conditional
     b_0(X) above sampling noise) and the binding-vertex distribution has
     reasonable spread.

Diagnoses corrected in this revision
------------------------------------
(a) TRUE-pi for pi(2u,1r) was reported as 1.2608 (impossible: pi is a
    conditional probability, must be in [0, 1]).  Cause: the displayed
    "true pi" was computed as
        beta_true_kt[j] / src_pop_kt[j],
    where beta_true_kt is the FULL-population fraction of the relevant
    latent type but src_pop_kt[j] = p_dict['p21_c'] was estimated from
    the D=0 sub-sample only.  In finite samples the two differ by O(1/sqrtN)
    and the ratio can exceed 1 when the D=0 arm under-samples a small type
    such as 2u (~ 2% of the population).  Fix: compute true source
    marginals from REALISED TYPES (full population), since under random
    treatment P^A(s^A) is a population quantity and is NOT estimated from
    D=0 here -- it is known from the simulation ground truth.

(b) The analytical LP exactly mirrors KT's bound.m / replicating5.py 5x9
    pi-parameterization (see analytical_lp_bounds for row-by-row notes).

(c) The CLP per-i LP from CLP_final_group picks
        argmin_v  v . b_hat[i]
    out of (up to) C(9,5) = 126 candidate dual vertices.  If all i pick
    the SAME vertex v* then E[v* . B] = v* . E[B] = the analytical bound,
    i.e. the CLP bound coincides with the analytical bound (no Jensen
    tightening).  The unique-vertex pathology happens when b_hat(X) is
    nearly constant across X (covariates have negligible power) OR when
    the variation lies entirely inside one vertex's optimality cone.  We
    therefore (i) print binding-vertex histograms for BOTH directions
    and the cross-fit R^2 per B-component, and (ii) redesign the DGP
    below so that two X-clusters have CONSERVATION MOMENTS THAT FLIP
    SIGNS, forcing different vertices to bind in different X-cells.

Synthetic DGP
-------------
- N = 2500 persons, T = 5 quarters per person (12,500 person-quarters).
- 28 base covariates per person matching CLP_final_group's COV_VARS.
- Each person draws ONE latent behavioural type from 14 KT-feasible types.
- Two-cluster mixture (welfare-attached vs work-oriented) keyed on
  (earnings history, AFDC history) via a logistic.  Cluster type
  distributions are designed so the conservation moments p^J(r) - p^A(r)
  flip signs across clusters at multiple rows -- this is what gives
  diverse binding vertices and a meaningful first-stage R^2.
- Random treatment D ~ Bernoulli(0.5) per person.
- IPW weight = 2 (true propensity 0.5 is known).

First-stage configurations compared
-----------------------------------
The CLP block runs TWELVE first-stage configurations -- two feature
regimes times six (estimator, mode) combinations:
    feature regime in {base, econ}
    estimator in    {OLS, Ridge, LASSO}
    mode in         {full, cv}

  * BASE regime: 28 base covariates only (matches CLP_final_group COV_VARS).
  * ECON regime: 28 base + ~400 squared/interaction features (mirrors
    the empirical ECON spec in CLP_granular_correct_econ.py / 255-feature
    set; the simulation's exact count is closer to 400 because we include
    all pairwise interactions, but the methodology is identical).

The DGP's cluster logit has a LINEAR component recoverable from the 28
base features (controlled by X_MOD_SCALE) and a NONLINEAR component
(squares and pairwise interactions) that is ONLY recoverable from the
econ feature set (controlled by X_NONLIN_SCALE).  This makes the econ
regime genuinely informative on top of the base regime: the first-stage
R^2 should rise and CLP bounds should tighten in expectation.

mode='full' = in-sample fit on the whole sample (NO cross-fitting;
biased in general but kept as a diagnostic).
mode='cv'   = GroupKFold by person (regular CLP first stage required
for paper-style regularity).

EXPECTED 'full'-mode pathologies (these are NOT bugs):
  * OLS_full and Ridge_full often produce DEGENERATE widths -- LB equal
    to UB or even mildly LB > UB -- because the in-sample b_hat is so
    well-fit that the per-i LP collapses (b_hat[i] ~ B[i] makes the dual
    vertex selection trivial).  This is precisely WHY CLP requires
    cross-fitting: the analyst has to use OUT-OF-SAMPLE b_hat to
    preserve the regularity that justifies the bound formula.
  * LASSO_full often shrinks heavily, producing low vertex diversity
    (top-vertex share near 1.0), which makes CLP collapse to the
    analytical bound regardless of mode.
The 'cv' modes are the legitimate CLP runs; the 'full' modes serve as
a diagnostic to make the value of cross-fitting visible.

Comparison reported
-------------------
- True beta_0 (population averages over types).
- True pi_0 (computed from POPULATION marginals over realised types).
- Analytical (population LP) bounds in KT's pi-parameterization.
- CLP bounds for each of the 6 first-stage configurations, all in
  pi-units (clipped to [0, 1]) plus widths, "tighter than analytical"
  flags, and binding-vertex-diversity diagnostics.
- Final results saved to synthetic_simulation_summary.csv.

Run:  python3 CLP_synthetic_simulation.py
"""

import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# Reuse the CLP machinery
from CLP_final_group import (
    build_A, compute_B, clp_estimate, multiplier_bootstrap_ci,
    enumerate_dual_vertices,
    BETA_NAMES, PI_NAMES,
    S_0N, S_1N, S_2N, S_0P, S_1P, S_2P,
    K_FOLDS, RANDOM_SEED, N_BOOTSTRAP,
    COV_VARS,
)

# =============================================================================
# DGP CONSTANTS
# =============================================================================
N_PERSONS = 2500
N_QUARTERS = 5
SEED = 42

# Strength of LINEAR X-modulation in latent type probabilities.  Larger
# values concentrate person-types more around predictable X-clusters,
# which gives the first stage clear signal to learn (positive cross-fit
# R^2) and spreads binding vertices across the dual polytope.  The
# linear part is recoverable from the 28 base covariates alone.
X_MOD_SCALE = 4.0

# Strength of NONLINEAR X-modulation: cluster membership ALSO depends on
# squares and pairwise interactions of base covariates.  This part is
# only recoverable from the expanded "econ" feature set (28 base + ~200
# squared/interaction terms).  Setting this >0 makes the empirical
# regime (more features) genuinely informative on top of the base regime.
X_NONLIN_SCALE = 3.0

STATE_TO_CODE = {
    "0n": S_0N, "1n": S_1N, "2n": S_2N,
    "0p": S_0P, "1p": S_1P, "2p": S_2P,
}
STATE_NAMES = ["0n", "1n", "2n", "0p", "1p", "2p"]

# 14 latent behavioural types: (name, AFDC observable, JF observable, base prob)
TYPE_DEFS = [
    ("T01_stay_0n",      "0n", "0n", 0.05),   # neither work nor welfare, no change
    ("T02_0n_to_1r",     "0n", "1p", 0.05),   # opt onto welfare with low earnings
    ("T03_stay_0p",      "0p", "0p", 0.10),   # stay on welfare with no earnings
    ("T04_0r_to_0n",     "0p", "0n", 0.02),   # exit welfare, no earnings
    ("T05_0r_to_1r",     "0p", "1p", 0.15),   # stay on welfare, gain earnings
    ("T06_0r_to_1n",     "0p", "1n", 0.03),   # exit welfare, low earnings
    ("T07_0r_to_2n",     "0p", "2n", 0.02),   # exit welfare, high earnings
    ("T08_0r_to_2u",     "0p", "2p", 0.01),   # stay on welfare, hide high earnings
    ("T09_stay_1n",      "1n", "1n", 0.10),   # stay below-FPL working off welfare
    ("T10_1n_to_1r",     "1n", "1p", 0.15),   # join welfare from low earnings
    ("T11_stay_2n",      "2n", "2n", 0.10),   # stay above-FPL working off welfare
    ("T12_2n_to_1r",     "2n", "1p", 0.10),   # cut earnings to qualify for welfare
    ("T13_1u_to_1r",     "1p", "1p", 0.10),   # 1u stops underreporting (Lemma 2)
    ("T14_2u_to_1r",     "2p", "1p", 0.02),   # 2u stops underreporting
]
N_TYPES = len(TYPE_DEFS)

# Map each latent type to KT beta index in CLP_final_group's BETA_NAMES order.
# BETA_NAMES order:
#   0: beta(0n,1r)   1: beta(0r,0n)   2: beta(2n,1r)   3: beta(0r,2n)
#   4: beta(0r,1r)   5: beta(0r,1n)   6: beta(1n,1r)   7: beta(0r,2u)
#   8: beta(2u,1r)
# -1 => "stay" type (does not appear among the 9 free betas)
TYPE_TO_BETA_IDX = {
    0:  -1,  # T01 stay 0n
    1:   0,  # T02 (0n,1r)
    2:  -1,  # T03 stay 0p
    3:   1,  # T04 (0r,0n)
    4:   4,  # T05 (0r,1r)
    5:   5,  # T06 (0r,1n)
    6:   3,  # T07 (0r,2n)
    7:   7,  # T08 (0r,2u)
    8:  -1,  # T09 stay 1n
    9:   6,  # T10 (1n,1r)
    10: -1,  # T11 stay 2n
    11:  2,  # T12 (2n,1r)
    12: -1,  # T13 (1u,1r) -- pooled into observable 1p, not separable
    13:  8,  # T14 (2u,1r)
}


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================
def generate_person_covariates(N, rng):
    """Generate 28 base covariates per person matching CLP_final_group COV_VARS."""
    X = np.zeros((N, len(COV_VARS)))

    X[:,  0] = rng.binomial(1, 0.40, N)             # age2534
    race = rng.choice([1, 2, 3], size=N, p=[0.30, 0.50, 0.20])
    X[:,  1] = (race == 2).astype(float)            # black
    X[:,  2] = (race == 3).astype(float)            # hisp
    X[:,  3] = (race == 1).astype(float)            # white
    X[:,  4] = rng.binomial(1, 0.55, N)             # marnvr
    X[:,  5] = rng.binomial(1, 0.20, N)             # marapt
    X[:,  6] = rng.binomial(1, 0.50, N)             # hsged
    X[:,  7] = rng.binomial(1, 0.30, N)             # nohsged
    X[:,  8] = rng.uniform(1, 18, N)                # yngchtru
    X[:,  9] = rng.poisson(2, N) + 1                # kidcount

    # Earnings history (correlated; some persons have stable employment)
    person_earn_mean = rng.lognormal(mean=2.0, sigma=1.0, size=N)
    for k in range(8):
        noise = rng.normal(0, 0.5, N)
        X[:, 10 + k] = np.maximum(0, person_earn_mean + noise)

    # AFDC payment history (correlated; reflects welfare attachment)
    person_adc_mean = rng.uniform(0, 5, N)
    for k in range(8):
        noise = rng.normal(0, 0.5, N)
        X[:, 18 + k] = np.maximum(0, person_adc_mean + noise)

    X[:, 26] = rng.binomial(1, 0.40, N)             # applcant
    X[:, 27] = rng.uniform(0, 12, N)                # yremp

    return X


def compute_type_probs(X, rng):
    """
    Compute X-dependent latent type probabilities using a TWO-CLUSTER mixture.

    Cluster A (welfare-attached) and Cluster B (work-oriented) have very
    different type distributions.  Cluster membership is determined
    deterministically from a logistic of (earnings_history, AFDC_history),
    making X strongly predictive of the type.  This guarantees a non-trivial
    first-stage R^2.

    Returns (N, N_TYPES) array of per-person type probabilities (each row
    is a convex combination of the two cluster distributions).

    DESIGN NOTE (revised).  The conservation moment at observable cell r
    is
        p^J(r) - p^A(r)  =  (mass of types entering r under JF)
                           - (mass of types leaving r under AFDC).
    The CLP per-i LP only beats the analytical LP if these moments VARY
    across X (so that b_hat(X) is informative).  Even better, if some
    moments FLIP SIGNS across X-cells then different dual vertices are
    optimal in different cells, giving genuine Jensen tightening.

    Cluster membership is determined by a logit with TWO components:
      LINEAR     (controlled by X_MOD_SCALE):     work_score - welf_score
                  Recoverable from the 28 base covariates alone.
      NONLINEAR  (controlled by X_NONLIN_SCALE):  squares + interactions
                  Only recoverable from the expanded "econ" feature
                  set.  Adding this part makes the empirical-style
                  regime with engineered features genuinely tighter
                  than the base 28-feature regime.
    """
    N = X.shape[0]

    def _std(z):
        return (z - z.mean()) / (z.std() + 1e-8)

    earn_total = X[:, 10:18].sum(axis=1)
    adc_total  = X[:, 18:26].sum(axis=1)
    work_score = _std(earn_total)
    welf_score = _std(adc_total)

    # ---- LINEAR component (visible to base 28 features) ------------
    linear_score = work_score - welf_score

    # ---- NONLINEAR component (only visible to econ-engineered set) --
    # Use squares of standardised continuous covariates and interactions
    # among demographic dummies x earnings/AFDC indices.  These cannot
    # be approximated by linear combinations of the 28 base features.
    age_d  = X[:, 0]                 # age2534 dummy in {0, 1}
    hsged  = X[:, 6]                 # hs/GED dummy
    nohs   = X[:, 7]                 # no hs/GED dummy
    yng    = _std(X[:, 8])           # standardised yngchtru
    kids   = _std(X[:, 9])           # standardised kidcount
    yremp  = _std(X[:, 27])          # standardised yremp

    nonlin_score = (
        + 0.7 * (work_score * age_d)         # earner X age dummy
        + 0.6 * (welf_score * kids)          # welfare X kid count
        - 0.5 * (welf_score ** 2)            # quadratic in welfare attachment
        + 0.5 * (work_score * hsged)         # earner X education
        - 0.5 * (yng * welf_score)           # young child X welfare
        + 0.4 * (work_score * yremp)         # earner X years employed
        - 0.4 * (welf_score * nohs)          # welfare X low-education
        + 0.3 * (work_score ** 2)            # quadratic in earnings
    )

    # logistic membership: P(cluster = work-oriented | X)
    logit = X_MOD_SCALE * linear_score + X_NONLIN_SCALE * nonlin_score
    p_work = 1.0 / (1.0 + np.exp(-logit))                   # (N,)

    # ---------------------------------------------------------------
    # Cluster A: welfare-attached.
    # Designed so that conservation moments are POSITIVE at off-welfare
    # rows -- mass flows FROM 0r OUT to all four off-welfare cells (0n,
    # 1n, 2n, 2u) -- which forces the LP dual to reach into one cone of
    # vertex-space, plus T13 (1u->1r) for activity at the 1p cell that
    # KT drops.
    #
    # Resulting cluster-A moments (before mixture; numbers approximate):
    #   row 0n:  +T04 - T02 = +0.10 - 0.02 = +0.08
    #   row 1n:  +T06 - T10 = +0.10 - 0.02 = +0.08
    #   row 2n:  +T07 - T12 = +0.10 - 0.02 = +0.08
    #   row 0p:  -(T04+T05+T06+T07+T08)    = -0.50
    #   row 2p:  +T08 - T14 = +0.10 - 0.07 = +0.03
    # ---------------------------------------------------------------
    pi_A = np.array([
        0.02,  # T01 stay_0n
        0.02,  # T02 0n->1r
        0.10,  # T03 stay_0p
        0.10,  # T04 0r->0n
        0.10,  # T05 0r->1r
        0.10,  # T06 0r->1n
        0.10,  # T07 0r->2n
        0.10,  # T08 0r->2u
        0.05,  # T09 stay_1n
        0.02,  # T10 1n->1r
        0.05,  # T11 stay_2n
        0.02,  # T12 2n->1r
        0.15,  # T13 1u->1r
        0.07,  # T14 2u->1r
    ])
    pi_A /= pi_A.sum()

    # ---------------------------------------------------------------
    # Cluster B: work-oriented.
    # Designed so that conservation moments are NEGATIVE at off-welfare
    # rows -- mass flows OFF off-welfare cells INTO welfare (1r) -- so
    # the LP dual reaches into a DIFFERENT cone of vertex-space.  This
    # is what produces sign flips relative to Cluster A and forces the
    # CLP to use distinct vertices for distinct X-cells.
    #
    # Resulting cluster-B moments (before mixture):
    #   row 0n:  +0.01 - 0.15 = -0.14   (sign flip vs A)
    #   row 1n:  +0.01 - 0.20 = -0.19   (sign flip vs A)
    #   row 2n:  +0.01 - 0.15 = -0.14   (sign flip vs A)
    #   row 0p:  -(5*0.01)    = -0.05   (same sign, much smaller)
    #   row 2p:  +0.01 - 0.04 = -0.03   (sign flip vs A)
    # ---------------------------------------------------------------
    pi_B = np.array([
        0.10,  # T01 stay_0n
        0.15,  # T02 0n->1r
        0.02,  # T03 stay_0p
        0.01,  # T04 0r->0n
        0.01,  # T05 0r->1r
        0.01,  # T06 0r->1n
        0.01,  # T07 0r->2n
        0.01,  # T08 0r->2u
        0.15,  # T09 stay_1n
        0.20,  # T10 1n->1r
        0.10,  # T11 stay_2n
        0.15,  # T12 2n->1r
        0.04,  # T13 1u->1r
        0.04,  # T14 2u->1r
    ])
    pi_B /= pi_B.sum()

    # mixture
    probs = (1.0 - p_work[:, None]) * pi_A[None, :] + p_work[:, None] * pi_B[None, :]
    return probs


def engineer_econ_features(X_base: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Build an "econ"-style EXPANDED feature set on top of the 28 base
    covariates: squared continuous features and pairwise interactions
    among a curated subset of demographic and earnings/AFDC dummies and
    indices.  Returns the EXTRA features (not including the 28 base) and
    a list of feature names.

    This mirrors engineer_features_econ in CLP_granular_final_group.py
    in spirit (not in exact column-by-column composition).  The total
    'econ' feature set used by the CLP first stage is
        np.hstack([X_base, X_econ_extra]).

    By construction, the expanded set contains nonlinear functions of
    the base features that are NOT linearly recoverable.  In combination
    with X_NONLIN_SCALE > 0 in compute_type_probs, this makes the
    empirical-style regime informative on top of the 28-feature regime.
    """
    N, p = X_base.shape

    # Indices of continuous-ish base covariates whose squares we add.
    # (yngchtru, kidcount, 8 earnings cols, 8 AFDC cols, yremp)
    cont_idx = list(range(8, 26)) + [27]                         # 19 cols

    # All-pairs interactions among the 28 base features.
    pair_idx = [(i, j) for i in range(p) for j in range(i + 1, p)]  # 378 pairs

    feats: list = []
    names: list = []

    for i in cont_idx:
        feats.append(X_base[:, i] ** 2)
        names.append(f"x{i}_sq")

    for i, j in pair_idx:
        feats.append(X_base[:, i] * X_base[:, j])
        names.append(f"x{i}_x{j}")

    if not feats:
        return np.zeros((N, 0)), []
    return np.column_stack(feats), names


def generate_synthetic(N=N_PERSONS, T=N_QUARTERS, seed=SEED):
    """Generate full synthetic dataset: N persons x T quarters."""
    rng = np.random.default_rng(seed)

    # Person-level covariates
    X_person = generate_person_covariates(N, rng)

    # Type probabilities + sampled types
    type_probs = compute_type_probs(X_person, rng)
    cum = np.cumsum(type_probs, axis=1)
    u = rng.uniform(size=N)
    types = (cum < u[:, None]).sum(axis=1)
    types = np.clip(types, 0, N_TYPES - 1)

    # Map each person to their AFDC and JF observable states
    s_a_per_person = np.array(
        [STATE_TO_CODE[TYPE_DEFS[t][1]] for t in types], dtype=int
    )
    s_j_per_person = np.array(
        [STATE_TO_CODE[TYPE_DEFS[t][2]] for t in types], dtype=int
    )

    # Random treatment per person, then replicate to T quarters
    D_person = rng.binomial(1, 0.5, size=N)

    person_id = np.repeat(np.arange(N), T)
    D = np.repeat(D_person, T)
    s_a = np.repeat(s_a_per_person, T)
    s_j = np.repeat(s_j_per_person, T)
    state_obs = np.where(D == 0, s_a, s_j)
    X = np.repeat(X_person, T, axis=0)

    # IPW weight = 2 because the true propensity is 0.5 (random treatment).
    pscorewt = np.full(len(D), 2.0)

    return dict(
        D=D, state_obs=state_obs, X=X,
        person_id=person_id, pscorewt=pscorewt,
        X_person=X_person, types=types, type_probs=type_probs,
    )


# =============================================================================
# TRUE BETA AND POPULATION MARGINALS
# =============================================================================
def compute_true_beta(types: np.ndarray) -> np.ndarray:
    """Compute true beta_0 (length 9) from realised type frequencies."""
    N = len(types)
    beta_true = np.zeros(9)
    for t_idx in range(N_TYPES):
        beta_idx = TYPE_TO_BETA_IDX[t_idx]
        if beta_idx < 0:
            continue
        beta_true[beta_idx] = (types == t_idx).sum() / N
    return beta_true


def compute_marginals(D, state_obs):
    """
    Compute the 12 observable cell probabilities (KT naming convention).
    KT key format: pXY_arm where X = earnings bin (0,1,2),
                                 Y = participation (0=non, 1=part),
                                 arm in {c (control = AFDC), t (treated = JF)}.
    These are the SAMPLE marginals (one arm at a time) used by the
    analytical LP and the CLP beta -> pi conversion -- they are noisy
    estimators of population marginals.
    """
    p = {}
    state_codes_kt = {
        "00": S_0N, "10": S_1N, "20": S_2N,
        "01": S_0P, "11": S_1P, "21": S_2P,
    }
    for arm_label, arm_mask in [("c", D == 0), ("t", D == 1)]:
        n_arm = arm_mask.sum()
        for kt_xy, code in state_codes_kt.items():
            count = ((state_obs == code) & arm_mask).sum()
            p[f"p{kt_xy}_{arm_label}"] = count / n_arm if n_arm > 0 else 0.0
    return p


def compute_true_afdc_marginals(types: np.ndarray) -> Dict[str, float]:
    """
    POPULATION-LEVEL AFDC source marginals computed from realised latent
    types (full sample, NOT the D=0 sub-sample).  Used to compute the
    TRUE pi values displayed as ground truth.

    Why this is needed.  Under random treatment, P^A(s^A = s) is a
    population-level probability that does NOT depend on D.  In real-data
    estimation we only see D=0 individuals' AFDC observable, so we must
    estimate P^A(s) from the D=0 sub-sample (sampling noise O(1/sqrtN)).
    But in this simulation we KNOW each person's potential AFDC state
    (it is determined by their type), so we should use the FULL sample
    when we display the ground-truth pi values.  Otherwise pi_true =
    beta_true / src_pop_estimated can drift outside [0, 1] in finite
    samples whenever beta_true uses the full sample but src_pop uses
    only D=0 -- which is exactly how pi(2u,1r) was reported as 1.2608.

    Returns a dict {observable cell name -> true population mass}.
    """
    afdc_marg = {s: 0.0 for s in STATE_NAMES}
    for t_idx, (_, afdc_obs, _, _) in enumerate(TYPE_DEFS):
        afdc_marg[afdc_obs] += float((types == t_idx).mean())
    return afdc_marg


# =============================================================================
# ANALYTICAL LP BOUNDS  (KT pi-parameterization, matches Bounds.m lp_bounds)
# =============================================================================
def analytical_lp_bounds(p):
    """
    Population-level LP bounds in KT's pi-parameterization, identical in
    structure to KT's Bounds.m / replicating5.py.

    Returns (lb, ub, kt_pi_names) for KT's 9 free transition probabilities
    in KT's canonical ordering:
        pi(0r,0n), pi(0n,1r), pi(0r,2n), pi(2n,1r), pi(0r,1r),
        pi(0r,2u), pi(2u,1r), pi(1n,1r), pi(0r,1n).

    Row-by-row derivation (under random assignment so P^A(.|D=0) = P^A(.)
    and P^J(.|D=1) = P^J(.); state names in KT obs convention):

        Row 1 -- conservation at observable cell 0n:
            P^J(0n) - P^A(0n)  =  P^A(0r)*pi(0r,0n)  -  P^A(0n)*pi(0n,1r)
                =>  p00_t - p00_c  =  p01_c*pi(0r,0n) - p00_c*pi(0n,1r)

        Row 2 -- conservation at observable cell 1n:
            P^J(1n) - P^A(1n)  =  P^A(0r)*pi(0r,1n)  -  P^A(1n)*pi(1n,1r)
                =>  p10_t - p10_c  =  p01_c*pi(0r,1n) - p10_c*pi(1n,1r)

        Row 3 -- conservation at observable cell 2n:
            P^J(2n) - P^A(2n)  =  P^A(0r)*pi(0r,2n)  -  P^A(2n)*pi(2n,1r)
                =>  p20_t - p20_c  =  p01_c*pi(0r,2n) - p20_c*pi(2n,1r)

        Row 4 -- conservation at observable cell 0p
            (no inflows under JF; outflows are the 5 0r-source betas):
            P^J(0p)            =  P^A(0r) * (1 - sum 5 outflows)
                =>  sum_5_outflows = (p01_c - p01_t) / p01_c

        Row 5 -- conservation at observable cell 2p:
            P^J(2p) - P^A(2p)  =  P^A(0r)*pi(0r,2u)  -  P^A(2u)*pi(2u,1r)
                =>  p21_t - p21_c  =  p01_c*pi(0r,2u) - p21_c*pi(2u,1r)

    The 1p row is the redundant adding-up row and is dropped (KT
    convention).  Box bounds [0, 1] enforce that pi values are
    conditional probabilities.  These FIVE equality rows + 9 box
    constraints are exactly what Bounds.m solves.
    """
    p00_c = p['p00_c']; p00_t = p['p00_t']
    p10_c = p['p10_c']; p10_t = p['p10_t']
    p20_c = p['p20_c']; p20_t = p['p20_t']
    p01_c = p['p01_c']; p01_t = p['p01_t']
    p21_c = p['p21_c']; p21_t = p['p21_t']

    # KT 5x9 LP in pi-parameterization (matches Bounds.m and replicating5.py).
    # pi ordering: 0r0n, 0n1r, 0r2n, 2n1r, 0r1r, 0r2u, 2u1r, 1n1r, 0r1n
    A_kt = np.array([
        [p01_c, -p00_c,  0,      0,      0,      0,      0,      0,      0     ],
        [0,      0,      0,      0,      0,      0,      0,     -p10_c,  p01_c ],
        [0,      0,      p01_c, -p20_c,  0,      0,      0,      0,      0     ],
        [1,      0,      1,      0,      1,      1,      0,      0,      1     ],
        [0,      0,      0,      0,      0,      p01_c, -p21_c,  0,      0     ],
    ])
    b_kt = np.array([
        p00_t - p00_c,
        p10_t - p10_c,
        p20_t - p20_c,
        (p01_c - p01_t) / p01_c if p01_c > 1e-10 else 0.0,
        p21_t - p21_c,
    ])

    n_pi = 9
    lb = np.zeros(n_pi)
    ub = np.zeros(n_pi)
    for j in range(n_pi):
        c = np.zeros(n_pi); c[j] = 1.0
        res_lb = linprog(c,    A_eq=A_kt, b_eq=b_kt,
                          bounds=[(0, 1)] * n_pi, method='highs',
                          options={'disp': False})
        res_ub = linprog(-c,   A_eq=A_kt, b_eq=b_kt,
                          bounds=[(0, 1)] * n_pi, method='highs',
                          options={'disp': False})
        lb[j] = float(res_lb.fun) if res_lb.status == 0 else 0.0
        ub[j] = float(-res_ub.fun) if res_ub.status == 0 else 1.0

    kt_pi_names = [
        "pi(0r,0n)", "pi(0n,1r)", "pi(0r,2n)", "pi(2n,1r)", "pi(0r,1r)",
        "pi(0r,2u)", "pi(2u,1r)", "pi(1n,1r)", "pi(0r,1n)",
    ]
    return lb, ub, kt_pi_names


# =============================================================================
# FIRST-STAGE ESTIMATOR FACTORY  (LASSO / Ridge / OLS)
# =============================================================================
def _make_model(estimator_name: str, seed: int):
    """Return an unfitted sklearn estimator for the requested first stage."""
    if estimator_name == "LASSO":
        return LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, eps=1e-4,
                       max_iter=3000, random_state=seed, n_jobs=1)
    elif estimator_name == "Ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 40))
    elif estimator_name == "OLS":
        return LinearRegression(n_jobs=1)
    else:
        raise ValueError(f"Unknown estimator: {estimator_name!r}")


def estimate_b0(B_obs: np.ndarray, X_raw: np.ndarray,
                person_id: np.ndarray,
                estimator: str = "LASSO", mode: str = "cv",
                K: int = K_FOLDS, seed: int = RANDOM_SEED) -> np.ndarray:
    """
    First-stage estimator with two modes.

      mode='full'  : fit ONCE on the full sample, predict on the same sample
                     (in-sample fit, NO cross-fitting).  This is biased in
                     general -- CLP regularity requires cross-fitting -- but
                     is included as a diagnostic to show how much the
                     cross-fit step affects the LP downstream.
      mode='cv'    : GroupKFold by person with K folds; predict on held-out
                     fold each time.  This is the regular CLP first stage.

    StandardScaler is fit on the training portion only (fit on the full
    sample in 'full' mode, fit on each train fold in 'cv' mode).

    Returns b_hat (n, k) where k = number of B-components.
    """
    n, k = B_obs.shape
    b_hat = np.zeros((n, k))

    if mode == "full":
        sc = StandardScaler()
        Xs = sc.fit_transform(X_raw)
        for j in range(k):
            y = B_obs[:, j]
            if y.std() < 1e-10:
                b_hat[:, j] = y.mean()
                continue
            model = _make_model(estimator, seed)
            model.fit(Xs, y)
            b_hat[:, j] = model.predict(Xs)
        return b_hat

    if mode == "cv":
        gkf = GroupKFold(n_splits=K)
        folds = list(gkf.split(np.arange(n), groups=person_id))
        for tr, te in folds:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_raw[tr])
            X_te = sc.transform(X_raw[te])
            for j in range(k):
                y = B_obs[tr, j]
                if y.std() < 1e-10:
                    b_hat[te, j] = y.mean()
                    continue
                model = _make_model(estimator, seed)
                model.fit(X_tr, y)
                b_hat[te, j] = model.predict(X_te)
        return b_hat

    raise ValueError(f"Unknown mode: {mode!r}")


# =============================================================================
# KT TABLE 5 COMPOSITE BOUNDS  (Take-Up Work, Take-Up Welfare, Exit 0r)
# =============================================================================
# Reference: Bounds.m lines 527-678 of the KT replication package.  KT solve
# the LP min/max f' pi  s.t.  A pi = b,  0 <= pi <= 1, where the matrix A and
# vector b are the 5x9 conservation system in pi-parameterization.  Three
# composites differ only in the f vector:
#
#  TUW (line 533):
#    f = [0 p00_c p01_c 0 p01_c p01_c 0 0 p01_c]./(p01_c+p00_c)
#    => point-identified; equals (p00_c+p01_c-(p00_t+p01_t))/(p01_c+p00_c)
#       (KT line 552).
#
#  TUWelf (line 567):
#    f = [0 p00_c 0 p20_c zeros(1,3) p10_c 0]./(p00_c+p10_c+p20_c)
#
#  Exit Welfare 2 (line 666):
#    f = [1 0 1 zeros(1,5) 1]   -- no normalization in the LP (already
#                                    in pi-units after the LP).
#
# In CLP beta-parameterization the same composites become q_beta vectors.
# The mapping uses pi(s,d) = beta(s,d) / P^a(s) so that f_k * pi_k =
# (f_k / P^a(s_k)) * beta_k.  CLP BETA_NAMES order:
#     0: beta(0n,1r)   1: beta(0r,0n)   2: beta(2n,1r)   3: beta(0r,2n)
#     4: beta(0r,1r)   5: beta(0r,1n)   6: beta(1n,1r)   7: beta(0r,2u)
#     8: beta(2u,1r)
#
# The resulting q_beta vectors are (using the AFDC source marginal denominators):
#
#   TUW    : q_beta = [1, 0, 0, 1, 1, 1, 0, 1, 0] / (p00_c + p01_c)
#   TUWelf : q_beta = [1, 0, 1, 0, 0, 0, 1, 0, 0] / (p00_c + p10_c + p20_c)
#   Exit   : q_beta = [0, 1, 0, 1, 0, 1, 0, 0, 0] / p01_c
# =============================================================================
def composite_q_vectors_kt(p_dict: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Three KT-Table-5 composite q-vectors in CLP beta-parameterization.

    p_dict provides the AFDC-arm sample marginals (output of
    compute_marginals): p00_c, p01_c, p10_c, p20_c.  Returns a dict mapping
    composite name -> 9-vector q_beta to be passed to clp_estimate.
    """
    p00_c = p_dict['p00_c']; p01_c = p_dict['p01_c']
    p10_c = p_dict['p10_c']; p20_c = p_dict['p20_c']
    n_beta = 9
    q_tuw    = np.zeros(n_beta); q_tuw   [[0, 3, 4, 5, 7]] = 1.0
    q_tuwelf = np.zeros(n_beta); q_tuwelf[[0, 2, 6      ]] = 1.0
    q_exit   = np.zeros(n_beta); q_exit  [[1, 3, 5      ]] = 1.0
    q_tuw    /= max(1e-12, p00_c + p01_c)
    q_tuwelf /= max(1e-12, p00_c + p10_c + p20_c)
    q_exit   /= max(1e-12, p01_c)
    return {
        "TUW (Take-Up Work)":         q_tuw,
        "TUWelf (Take-Up Welfare)":   q_tuwelf,
        "Exit 0r (on-welfare zero earn -> off welfare)": q_exit,
    }


def true_composites_from_types(types: np.ndarray,
                                 type_defs: list) -> Dict[str, float]:
    """Compute the TRUE population values of the three KT composites
    directly from the realised type frequencies.  Each composite is a
    POPULATION conditional probability:
       TUW    = P[ working under JF | not working under AFDC ]
       TUWelf = P[ on welfare under JF | off welfare under AFDC ]
       Exit   = P[ off welfare under JF | 0r under AFDC ]
    """
    N = len(types)

    # Per-i AFDC and JF observable cells from realised type.
    afdc_cell = np.array([type_defs[t][1] for t in types])
    jf_cell   = np.array([type_defs[t][2] for t in types])

    # Helpers to classify cells.
    def _not_working(c): return c in ("0n", "0p")
    def _working(c):     return c in ("1n", "2n", "1p", "2p")
    def _off_welfare(c): return c in ("0n", "1n", "2n")
    def _on_welfare(c):  return c in ("0p", "1p", "2p")
    def _is_0r(c):       return c == "0p"   # 0r alias

    not_working_a = np.array([_not_working(c) for c in afdc_cell])
    working_j     = np.array([_working(c)     for c in jf_cell])
    off_a         = np.array([_off_welfare(c) for c in afdc_cell])
    on_j          = np.array([_on_welfare(c)  for c in jf_cell])
    is_0r_a       = np.array([_is_0r(c)       for c in afdc_cell])
    off_j         = np.array([_off_welfare(c) for c in jf_cell])

    return {
        "TUW (Take-Up Work)":         (
            float((not_working_a & working_j).sum())
            / max(1, int(not_working_a.sum()))
        ),
        "TUWelf (Take-Up Welfare)":   (
            float((off_a & on_j).sum()) / max(1, int(off_a.sum()))
        ),
        "Exit 0r (on-welfare zero earn -> off welfare)": (
            float((is_0r_a & off_j).sum()) / max(1, int(is_0r_a.sum()))
        ),
    }


def analytical_lp_composites(p: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """Population-LP composite bounds in KT pi-parameterization (matches
    Bounds.m lines 527-678).  Returns {name: (lb, ub)}."""
    p00_c = p['p00_c']; p00_t = p['p00_t']
    p10_c = p['p10_c']; p10_t = p['p10_t']
    p20_c = p['p20_c']; p20_t = p['p20_t']
    p01_c = p['p01_c']; p01_t = p['p01_t']
    p21_c = p['p21_c']; p21_t = p['p21_t']

    # Same 5x9 LP system as analytical_lp_bounds but with composite f-vectors.
    A_kt = np.array([
        [p01_c, -p00_c,  0,      0,      0,      0,      0,      0,      0     ],
        [0,      0,      0,      0,      0,      0,      0,     -p10_c,  p01_c ],
        [0,      0,      p01_c, -p20_c,  0,      0,      0,      0,      0     ],
        [1,      0,      1,      0,      1,      1,      0,      0,      1     ],
        [0,      0,      0,      0,      0,      p01_c, -p21_c,  0,      0     ],
    ])
    b_kt = np.array([
        p00_t - p00_c, p10_t - p10_c, p20_t - p20_c,
        (p01_c - p01_t) / p01_c if p01_c > 1e-10 else 0.0,
        p21_t - p21_c,
    ])

    # KT f vectors in pi-parameterization.  All three return values that
    # are ALREADY in pi-units (conditional probabilities); no further
    # division is required.
    #   TUW    : f^T pi = (β(0n,1r) + β(0r,2n) + β(0r,1r) + β(0r,1n)
    #                       + β(0r,2u)) / (p00 + p01)  (cf. Bounds.m line 533)
    #   TUWelf : f^T pi = (β(0n,1r) + β(2n,1r) + β(1n,1r))
    #                       / (p00 + p10 + p20)  (cf. line 567)
    #   Exit   : f^T pi = π(0r,0n) + π(0r,2n) + π(0r,1n)  (cf. line 666;
    #                       KT use NO normalization here -- the result is
    #                       already in pi-units, bounded in [0, 1] via the
    #                       row-4 constraint sum_j pi(0r,*) <= 1).
    composites_f = {
        "TUW (Take-Up Work)":
            np.array([0, p00_c, p01_c, 0, p01_c, p01_c, 0, 0, p01_c])
            / max(1e-12, p00_c + p01_c),
        "TUWelf (Take-Up Welfare)":
            np.array([0, p00_c, 0, p20_c, 0, 0, 0, p10_c, 0])
            / max(1e-12, p00_c + p10_c + p20_c),
        "Exit 0r (on-welfare zero earn -> off welfare)":
            np.array([1, 0, 1, 0, 0, 0, 0, 0, 1], dtype=float),
    }
    out = {}
    for name, f in composites_f.items():
        res_lb = linprog(f,    A_eq=A_kt, b_eq=b_kt,
                          bounds=[(0, 1)] * 9, method='highs',
                          options={'disp': False})
        res_ub = linprog(-f,   A_eq=A_kt, b_eq=b_kt,
                          bounds=[(0, 1)] * 9, method='highs',
                          options={'disp': False})
        lb = float(res_lb.fun) if res_lb.status == 0 else 0.0
        ub = float(-res_ub.fun) if res_ub.status == 0 else 1.0
        out[name] = (lb, ub)
    return out


# =============================================================================
# DIAGNOSTICS  (port of D2-D7 from CLP_final_group)
# =============================================================================
# These are the same diagnostics CLP_final_group prints per (feature_set,
# estimator) configuration.  We reuse them here per (estimator, mode,
# feature regime) configuration so that the synthetic simulation
# matches the structure of the real-data CLP output.
#
#   D2  vertex margin                : gap between 1st and 2nd smallest
#                                       LP scores; near-zero = fragile
#                                       vertex selection
#   D3  binding-vertex entropy       : Shannon entropy of v* across i,
#                                       plus distinct-vertex count and
#                                       top vertex's share
#   D4  first-stage R^2 per component: already printed by run_clp;
#                                       repeated in the diagnostic block
#                                       for completeness
#   D5  calibration plot              : per-component slope of (mean obs)
#                                       on (mean predicted) across deciles
#   D7  vertex stability              : fraction of i's whose binding
#                                       vertex flips when b_hat is
#                                       perturbed by 10% noise
# =============================================================================
def diag_vertex_margin(b_hat: np.ndarray, A_mat: np.ndarray,
                        q: np.ndarray) -> np.ndarray:
    """Per-i margin m_i = s_{i,(2)} - s_{i,(1)} where s_{i,j} = v_j^T b_hat[i]
    and the v_j are the candidate dual vertices.  Small m_i => the binding
    vertex is on a near-tie with the second-best, so a small first-stage
    perturbation can flip it (fragile).  Returns array of length n."""
    vertices = enumerate_dual_vertices(A_mat, q)
    if len(vertices) < 2:
        return np.zeros(len(b_hat))
    n = len(b_hat)
    margins = np.zeros(n)
    V = np.array(vertices)               # (#vertices, k)
    scores = b_hat @ V.T                  # (n, #vertices)
    s_sorted = np.sort(scores, axis=1)    # ascending; we picked argmin
    margins = s_sorted[:, 1] - s_sorted[:, 0]
    return margins


def diag_binding_vertex_entropy(vidx: np.ndarray) -> Tuple[float, int, float]:
    """Shannon entropy of the binding-vertex distribution across i,
    plus number of distinct vertices used and modal-vertex share.
    Higher entropy = more diverse (more Jensen tightening); entropy
    near 0 + top_share near 1 = LP collapsed to one vertex (CLP
    coincides with analytical bound)."""
    from collections import Counter
    cnt = Counter(int(v) for v in vidx)
    n = len(vidx)
    phat = np.array([c / n for c in cnt.values()])
    H = float(-np.sum(phat * np.log(phat + 1e-12)))
    n_distinct = len(cnt)
    top_share = float(max(cnt.values()) / n)
    return H, n_distinct, top_share


def diag_calibration(b_hat: np.ndarray, B_obs: np.ndarray,
                      n_bins: int = 10) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Per-component calibration: bin observations by predicted decile,
    return (pred_means, obs_means) per component.  Slope on the resulting
    decile-mean scatter is a calibration diagnostic (slope=1 well-calibrated;
    slope=0 collapsed to constant; slope>1 overfit)."""
    k = B_obs.shape[1]
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for j in range(k):
        idx = np.argsort(b_hat[:, j])
        bins = np.array_split(idx, n_bins)
        pred_means = np.array([b_hat[b, j].mean() for b in bins])
        obs_means  = np.array([B_obs[b, j].mean() for b in bins])
        out[j] = (pred_means, obs_means)
    return out


def _calibration_slope(pred_means: np.ndarray,
                        obs_means: np.ndarray) -> float:
    """OLS slope of obs_means on pred_means across the n_bins bin means.
    Returns NaN if pred_means has no variance (constant predictor)."""
    pm = np.asarray(pred_means, dtype=float)
    om = np.asarray(obs_means, dtype=float)
    if pm.std() < 1e-10:
        return float('nan')
    return float(np.cov(pm, om)[0, 1] / np.var(pm))


def diag_vertex_stability(b_hat: np.ndarray, A_mat: np.ndarray,
                           q: np.ndarray, noise_frac: float = 0.1,
                           seed: int = 7777) -> float:
    """Perturb each b_hat[i] by Gaussian noise with sd = noise_frac *
    component-std, recompute the binding vertex, and return the fraction
    of i's whose binding vertex changed.

      < 5%  : stable first stage
      5-20% : moderate
      > 20% : fragile (LP solution sensitive to small first-stage noise)
      ~100%: first stage essentially constant (any noise flips the vertex)
    """
    rng = np.random.default_rng(seed)
    noise_std = np.maximum(b_hat.std(axis=0) * noise_frac, 1e-10)
    b_perturbed = b_hat + rng.normal(0.0, noise_std, b_hat.shape)
    vertices = enumerate_dual_vertices(A_mat, q)
    if len(vertices) < 2:
        return float('nan')
    V = np.array(vertices)               # (#vertices, k)
    orig_idx = np.argmin(b_hat @ V.T,        axis=1)
    pert_idx = np.argmin(b_perturbed @ V.T,  axis=1)
    return float(np.mean(orig_idx != pert_idx))


def print_diagnostics(label: str,
                       b_hat: np.ndarray, B_obs: np.ndarray,
                       A_mat: np.ndarray,
                       all_vidx_ub: List[np.ndarray],
                       all_vidx_lb: List[np.ndarray]) -> Dict[str, dict]:
    """Print D2 / D3 / D5 / D7 diagnostics for one CLP configuration.
    `all_vidx_ub` and `all_vidx_lb` are lists of length n_pi (one per
    beta direction) of vertex-index arrays from clp_estimate."""
    n_pi = len(BETA_NAMES)
    state_labels = ["0n", "1n", "2n", "0p", "2p"][:B_obs.shape[1]]
    SEP = "-" * 80

    print(f"\n{SEP}")
    print(f"  DIAGNOSTICS: {label}")
    print(SEP)

    # --- D2: Vertex margin per beta direction (UB only; LB symmetric) ---
    print(f"\n  [D2] Vertex margin (gap between 1st and 2nd smallest LP "
          f"scores; UB direction)")
    print(f"  {'beta':<14} {'median':>9}  {'p10':>9}  {'%<1e-4':>8}  "
          f"{'%<1e-3':>8}")
    print(f"  {'-' * 56}")
    n_red_margin = 0
    for j, bname in enumerate(BETA_NAMES):
        q_up = np.zeros(n_pi); q_up[j] = 1.0
        m = diag_vertex_margin(b_hat, A_mat, q_up)
        med = float(np.median(m))
        p10 = float(np.percentile(m, 10))
        pct1 = 100.0 * float(np.mean(m < 1e-4))
        pct3 = 100.0 * float(np.mean(m < 1e-3))
        flag = "  <- fragile" if pct1 > 20 else ""
        if pct1 > 20:
            n_red_margin += 1
        print(f"  {bname:<14} {med:9.5f}  {p10:9.5f}  {pct1:7.1f}%  "
              f"{pct3:7.1f}%{flag}")

    # --- D3: Binding-vertex entropy / distribution ---------------------
    print(f"\n  [D3] Binding-vertex distribution (UB / LB)  "
          f"H = Shannon entropy, nv = #distinct, top = modal share")
    print(f"  {'beta':<14} {'UB H':>7} {'UB nv':>6} {'UB top':>7}    "
          f"{'LB H':>7} {'LB nv':>6} {'LB top':>7}")
    print(f"  {'-' * 70}")
    H_ub_arr, H_lb_arr = [], []
    for j, bname in enumerate(BETA_NAMES):
        H_u, nv_u, ts_u = diag_binding_vertex_entropy(all_vidx_ub[j])
        H_l, nv_l, ts_l = diag_binding_vertex_entropy(all_vidx_lb[j])
        H_ub_arr.append(H_u); H_lb_arr.append(H_l)
        flag = "  <- collapse" if ts_u > 0.95 and ts_l > 0.95 else ""
        print(f"  {bname:<14} {H_u:7.3f} {nv_u:6d} {ts_u:7.3f}    "
              f"{H_l:7.3f} {nv_l:6d} {ts_l:7.3f}{flag}")

    # --- D5: Calibration -----------------------------------------------
    print(f"\n  [D5] Calibration (decile-mean predicted vs observed; "
          f"slope=1 ok, ~0 collapsed, >1.3 overfit)")
    print(f"  {'comp':>6}  {'slope':>8}  {'min_pred':>10}  "
          f"{'max_pred':>10}  interpretation")
    print(f"  {'-' * 60}")
    calib = diag_calibration(b_hat, B_obs)
    for j, sl in enumerate(state_labels):
        pm, om = calib[j]
        slope = _calibration_slope(pm, om)
        if np.isnan(slope):
            interp = "n/a (no variance in pred)"
        elif abs(slope) < 0.2:
            interp = "COLLAPSED (slope~=0)"
        elif slope > 1.3:
            interp = "OVERFIT (slope>1.3)"
        elif 0.7 <= slope <= 1.3:
            interp = "calibrated"
        else:
            interp = f"moderate (slope={slope:.2f})"
        slope_str = f"{slope:8.3f}" if not np.isnan(slope) else "     nan"
        print(f"  B[{sl}]  {slope_str}  {pm.min():10.5f}  "
              f"{pm.max():10.5f}  {interp}")

    # --- D7: Vertex stability under perturbation ----------------------
    print(f"\n  [D7] Vertex stability under 10%-noise perturbation "
          f"(switch rate; UB direction)")
    print(f"  {'beta':<14} {'switch_rate':>12}  interpretation")
    print(f"  {'-' * 50}")
    n_unstable = 0
    for j, bname in enumerate(BETA_NAMES):
        q_up = np.zeros(n_pi); q_up[j] = 1.0
        sr = diag_vertex_stability(b_hat, A_mat, q_up)
        if np.isnan(sr):
            interp = "n/a"
        elif sr < 0.05:
            interp = "stable (<5%)"
        elif sr < 0.20:
            interp = "moderate (5-20%)"
        elif sr < 0.50:
            interp = "fragile (20-50%)"
        else:
            interp = "UNSTABLE (>50%)"
            n_unstable += 1
        sr_str = f"{sr:12.4f}" if not np.isnan(sr) else "         nan"
        print(f"  {bname:<14} {sr_str}  {interp}")

    # --- Combined summary ---------------------------------------------
    print(f"\n  Diagnostic summary for {label}:")
    print(f"    D2 fragile-margin betas (>20% with m<1e-4): "
          f"{n_red_margin}/{n_pi}")
    print(f"    D3 mean entropy (UB):  {np.mean(H_ub_arr):.3f}  "
          f"(higher = more diverse; 0 = collapsed)")
    print(f"    D3 mean entropy (LB):  {np.mean(H_lb_arr):.3f}")
    print(f"    D7 unstable betas (>50% switch): {n_unstable}/{n_pi}")
    return dict(
        n_red_margin=n_red_margin,
        H_ub_mean=float(np.mean(H_ub_arr)),
        H_lb_mean=float(np.mean(H_lb_arr)),
        n_unstable=n_unstable,
    )


# =============================================================================
# CLP BOUNDS  (per-i LP + multiplier bootstrap, parameterized first stage)
# =============================================================================
def run_clp(D, state_obs, X, person_id, pscorewt, p_dict,
             estimator: str = "LASSO", mode: str = "cv",
             n_bootstrap=N_BOOTSTRAP, K=K_FOLDS, seed=RANDOM_SEED):
    """
    Run CLP on the synthetic data with a parameterised first stage.

    Args:
      estimator: 'LASSO', 'Ridge', or 'OLS' -- choice of first-stage model.
      mode:      'full' (in-sample) or 'cv' (GroupKFold cross-fit by person).
    """
    A_mat = build_A()
    B_obs = compute_B(D, state_obs, pscorewt=pscorewt)
    if mode == "cv":
        print(f"\n  CLP first stage: {estimator}  (GroupKFold by person, "
              f"K={K}, {X.shape[1]} features, {len(np.unique(person_id)):,} "
              f"persons, {len(D):,} obs)")
    else:
        print(f"\n  CLP first stage: {estimator}  (FULL-SAMPLE in-sample fit, "
              f"{X.shape[1]} features, {len(D):,} obs) -- diagnostic only")
    b_hat = estimate_b0(B_obs, X, person_id, estimator=estimator, mode=mode,
                        K=K, seed=seed)

    # Sanity diagnostics on first stage
    b_emp = B_obs.mean(axis=0)
    b_fit = b_hat.mean(axis=0)
    print(f"    Mean B_obs (~ b_0):  {b_emp.round(4)}")
    print(f"    Mean b_hat:           {b_fit.round(4)}")
    print(f"    Max |bias|:           {np.max(np.abs(b_emp - b_fit)):.2e}")

    # First-stage R^2 per component.  Note the structural ceiling: even a
    # perfect prediction has R^2 = Var(E[B|X]) / Var(B), which for our
    # two-cluster DGP equals approximately
    #     0.25 * (mean_A - mean_B)^2 / Var(B)
    # By design this gives R^2 ceilings in the 4-8% range per row -- the
    # remaining 92-96% of Var(B) is the IRREDUCIBLE noise from D-flip and
    # cell-match (B is a 0 / +-2 indicator).  This matches the real-data
    # first-stage performance reported in the CLP paper (~5% out-of-fold).
    print(f"    First-stage R^2 per B-component (ceiling ~5% by construction):")
    for j, label in enumerate(["0n", "1n", "2n", "0p", "2p"]):
        ss_res = float(((B_obs[:, j] - b_hat[:, j]) ** 2).sum())
        ss_tot = float(((B_obs[:, j] - B_obs[:, j].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float('nan')
        print(f"      row {j} ({label}): R^2 = {r2:+.4f}")

    print(f"\n  CLP estimator for each beta (UB and LB) + multiplier bootstrap")
    n_pi = 9
    lb_b = np.zeros(n_pi); ub_b = np.zeros(n_pi)
    ci_lb_b = np.zeros((n_pi, 2)); ci_ub_b = np.zeros((n_pi, 2))
    n_verts_ub = []   # # distinct dual vertices used (UB direction)
    n_verts_lb = []   # # distinct dual vertices used (LB direction)
    top_share_ub = [] # share of i picking the modal vertex (UB)
    top_share_lb = [] # share of i picking the modal vertex (LB)
    all_vidx_ub: List[np.ndarray] = []   # full vertex-index arrays (per beta)
    all_vidx_lb: List[np.ndarray] = []   # used by print_diagnostics for D3

    for j in range(n_pi):
        q_up = np.zeros(n_pi); q_up[j] =  1.0
        q_dn = np.zeros(n_pi); q_dn[j] = -1.0

        ub_hat, c_up, _, verts_up, vidx_up = clp_estimate(
            q_up, b_hat, B_obs, A_mat,
        )
        ci_ub_b[j] = list(multiplier_bootstrap_ci(
            c_up, n_bs=n_bootstrap, person_id=person_id,
        ))
        n_verts_ub.append(np.unique(vidx_up).size)
        _, cnts_up = np.unique(vidx_up, return_counts=True)
        top_share_ub.append(cnts_up.max() / cnts_up.sum())
        all_vidx_ub.append(np.asarray(vidx_up))

        neg_lb, c_dn, _, verts_dn, vidx_dn = clp_estimate(
            q_dn, b_hat, B_obs, A_mat,
        )
        lb_b[j] = -neg_lb
        ci_lb_b[j] = [-x for x in
                      multiplier_bootstrap_ci(
                          c_dn, n_bs=n_bootstrap, person_id=person_id,
                      )[::-1]]
        ub_b[j] = ub_hat
        n_verts_lb.append(np.unique(vidx_dn).size)
        _, cnts_dn = np.unique(vidx_dn, return_counts=True)
        top_share_lb.append(cnts_dn.max() / cnts_dn.sum())
        all_vidx_lb.append(np.asarray(vidx_dn))

    # beta -> pi conversion using p_dict (sample marginals from D=0 arm)
    src_pop = np.array([
        p_dict['p00_c'],  # beta(0n,1r) source = 0n
        p_dict['p01_c'],  # beta(0r,0n) source = 0p
        p_dict['p20_c'],  # beta(2n,1r) source = 2n
        p_dict['p01_c'],  # beta(0r,2n) source = 0p
        p_dict['p01_c'],  # beta(0r,1r) source = 0p
        p_dict['p01_c'],  # beta(0r,1n) source = 0p
        p_dict['p10_c'],  # beta(1n,1r) source = 1n
        p_dict['p01_c'],  # beta(0r,2u) source = 0p
        p_dict['p21_c'],  # beta(2u,1r) source = 2u observable
    ])
    src_pop = np.maximum(src_pop, 1e-10)
    lb_pi = lb_b / src_pop
    ub_pi = ub_b / src_pop
    ci_lb_pi = ci_lb_b / src_pop[:, None]
    ci_ub_pi = ci_ub_b / src_pop[:, None]

    # ---- compact summary: vertex diversity per direction ---------------
    # The full diagnostic block (D2 / D3 / D5 / D7) is printed below; this
    # is just a one-line summary table for quick reading.
    print(f"\n  Binding-vertex diversity per direction "
          f"(# distinct vertices used; top vertex's share of obs):")
    print(f"    {'beta':<14}  {'UB#v':>4}  {'UB top%':>8}    "
          f"{'LB#v':>4}  {'LB top%':>8}")
    for j, nm in enumerate(BETA_NAMES):
        print(f"    {nm:<14}  {n_verts_ub[j]:4d}  {top_share_ub[j]:8.3f}    "
              f"{n_verts_lb[j]:4d}  {top_share_lb[j]:8.3f}")

    # ---- KT Table 5 composite bounds (CLP plug-in) ---------------------
    # Same LP machinery as the per-beta loop above, but with q = q_beta
    # for the composite (a 9-vector with weights).  Three composites:
    # TUW (Take-Up Work), TUWelf (Take-Up Welfare), Exit 0r.
    print(f"\n  CLP composite bounds (KT Table 5 in pi-units, clipped to [0,1])")
    print(f"  {'composite':<48}  {'LB':>8}  {'UB':>8}  {'width':>8}    "
          f"{'95% CI':>20}")
    print("  " + "-" * 100)
    composite_results = {}
    for cname, q_beta in composite_q_vectors_kt(p_dict).items():
        ub_hat, c_up, _, _, _ = clp_estimate(q_beta, b_hat, B_obs, A_mat)
        ci_ub_c = list(multiplier_bootstrap_ci(
            c_up, n_bs=n_bootstrap, person_id=person_id))
        neg_lb, c_dn, _, _, _ = clp_estimate(-q_beta, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb
        ci_lb_c = [-x for x in
                   multiplier_bootstrap_ci(
                       c_dn, n_bs=n_bootstrap, person_id=person_id)[::-1]]
        lb_clip = max(0.0, min(1.0, lb_hat))
        ub_clip = max(0.0, min(1.0, ub_hat))
        ci_lo = max(0.0, min(1.0, ci_lb_c[0]))
        ci_hi = max(0.0, min(1.0, ci_ub_c[1]))
        composite_results[cname] = dict(
            lb=lb_hat, ub=ub_hat, width=ub_hat - lb_hat,
            lb_clip=lb_clip, ub_clip=ub_clip,
            ci_lb=tuple(ci_lb_c), ci_ub=tuple(ci_ub_c),
            ci_outer=(ci_lo, ci_hi),
        )
        print(f"  {cname:<48}  {lb_clip:8.4f}  {ub_clip:8.4f}  "
              f"{ub_clip - lb_clip:8.4f}    [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ---- Full diagnostic block (D2 / D3 / D5 / D7) ---------------------
    label = (f"{estimator}_{mode} "
              f"({X.shape[1]} features, "
              f"{'cross-fit' if mode == 'cv' else 'in-sample'})")
    diag_summary = print_diagnostics(
        label=label,
        b_hat=b_hat, B_obs=B_obs, A_mat=A_mat,
        all_vidx_ub=all_vidx_ub, all_vidx_lb=all_vidx_lb,
    )

    return dict(
        lb_beta=lb_b, ub_beta=ub_b,
        ci_lb_beta=ci_lb_b, ci_ub_beta=ci_ub_b,
        lb_pi=lb_pi, ub_pi=ub_pi,
        ci_lb_pi=ci_lb_pi, ci_ub_pi=ci_ub_pi,
        b_hat=b_hat, B_obs=B_obs,
        n_vertices_ub=n_verts_ub,
        n_vertices_lb=n_verts_lb,
        top_share_ub=top_share_ub,
        top_share_lb=top_share_lb,
        all_vidx_ub=all_vidx_ub,
        all_vidx_lb=all_vidx_lb,
        diag_summary=diag_summary,
        composite=composite_results,
    )


# =============================================================================
# COMPARE ANALYTICAL vs CLP
# =============================================================================
# CLP (BETA_NAMES) order -> KT pi order:
# CLP idx 0 = beta(0n,1r) -> KT idx 1 = pi(0n,1r)
# CLP idx 1 = beta(0r,0n) -> KT idx 0 = pi(0r,0n)
# CLP idx 2 = beta(2n,1r) -> KT idx 3 = pi(2n,1r)
# CLP idx 3 = beta(0r,2n) -> KT idx 2 = pi(0r,2n)
# CLP idx 4 = beta(0r,1r) -> KT idx 4 = pi(0r,1r)
# CLP idx 5 = beta(0r,1n) -> KT idx 8 = pi(0r,1n)
# CLP idx 6 = beta(1n,1r) -> KT idx 7 = pi(1n,1r)
# CLP idx 7 = beta(0r,2u) -> KT idx 5 = pi(0r,2u)
# CLP idx 8 = beta(2u,1r) -> KT idx 6 = pi(2u,1r)
KT_ORDER_FROM_CLP = [1, 0, 3, 2, 4, 7, 8, 6, 5]  # KT[k] = CLP[KT_ORDER[k]]


def main():
    t0 = time.time()
    print("=" * 80)
    print("CLP_synthetic_simulation")
    print("=" * 80)

    print(f"\n[STEP 1] Generate synthetic data (N={N_PERSONS} persons, "
          f"T={N_QUARTERS} quarters)")
    data = generate_synthetic(N=N_PERSONS, T=N_QUARTERS, seed=SEED)
    D, state_obs, X = data['D'], data['state_obs'], data['X']
    person_id, pscorewt = data['person_id'], data['pscorewt']
    types = data['types']

    n_obs = len(D)
    n_persons = len(np.unique(person_id))
    print(f"  Observations: {n_obs:,}, persons: {n_persons:,}")
    print(f"  Treatment: D=1: {(D == 1).sum():,}  D=0: {(D == 0).sum():,}")
    print(f"  Observable state distribution:")
    for s_name, s_code in zip(STATE_NAMES,
                               [S_0N, S_1N, S_2N, S_0P, S_1P, S_2P]):
        print(f"    {s_name:>3}: {(state_obs == s_code).sum():5,} "
              f"({(state_obs == s_code).mean():.3f})")

    # Show realised type proportions vs base
    print(f"\n  Realised type frequencies vs base probs (X-modulated):")
    for t_idx, (name, _, _, base_p) in enumerate(TYPE_DEFS):
        freq = (types == t_idx).mean()
        print(f"    {name:<20} base={base_p:.3f}  realised={freq:.4f}")

    # True beta
    beta_true = compute_true_beta(types)
    print(f"\n[STEP 2] True beta_0 (population averages over realised types):")
    for j, nm in enumerate(BETA_NAMES):
        print(f"    {nm:<14}: {beta_true[j]:.4f}")

    print(f"\n[STEP 3] Compute population marginals")
    p_dict = compute_marginals(D, state_obs)
    for k in ('p00_c', 'p00_t', 'p10_c', 'p10_t', 'p20_c', 'p20_t',
              'p01_c', 'p01_t', 'p11_c', 'p11_t', 'p21_c', 'p21_t'):
        print(f"    {k} = {p_dict[k]:.4f}")

    print(f"\n[STEP 4] Analytical (population LP) bounds in pi-units")
    lb_an, ub_an, kt_names = analytical_lp_bounds(p_dict)

    # True pi computed by dividing true beta by *POPULATION* source-state
    # marginal P^A(s^A).  The population marginal is computed from realised
    # types over the whole sample (NOT the D=0 sub-sample), since under
    # random treatment P^A(s^A) is a population quantity that does not
    # depend on D.  The previous version of this file divided by the D=0
    # sample marginal, which produced pi(2u,1r) > 1 in finite samples
    # (the only 2u-source type T14 was under-sampled in the D=0 arm).
    true_afdc = compute_true_afdc_marginals(types)
    src_pop_kt_true = np.array([
        true_afdc['0p'],   # pi(0r,0n) source = 0r (observable 0p)
        true_afdc['0n'],   # pi(0n,1r) source = 0n
        true_afdc['0p'],   # pi(0r,2n) source = 0r
        true_afdc['2n'],   # pi(2n,1r) source = 2n
        true_afdc['0p'],   # pi(0r,1r) source = 0r
        true_afdc['0p'],   # pi(0r,2u) source = 0r
        true_afdc['2p'],   # pi(2u,1r) source = 2u (observable 2p)
        true_afdc['1n'],   # pi(1n,1r) source = 1n
        true_afdc['0p'],   # pi(0r,1n) source = 0r
    ])
    beta_true_kt = beta_true[KT_ORDER_FROM_CLP]
    pi_true_kt = beta_true_kt / np.maximum(src_pop_kt_true, 1e-10)

    # Cross-check: pi must lie in [0, 1] -- this is the Issue-1 invariant.
    bad_pi = [(kt_names[j], pi_true_kt[j])
              for j in range(len(kt_names))
              if not (-1e-9 <= pi_true_kt[j] <= 1 + 1e-9)]
    if bad_pi:
        print(f"    [WARN] pi_true outside [0,1] (should never happen now): {bad_pi}")
    else:
        print(f"    pi_true is in [0, 1] for all 9 parameters (Issue-1 fixed).")

    print(f"    {'parameter':<14}  {'true pi':>9}  {'LB_an':>9}  {'UB_an':>9}  "
          f"{'width':>8}  in_bnds")
    for j, nm in enumerate(kt_names):
        in_b = "yes" if (lb_an[j] - 1e-6 <= pi_true_kt[j] <= ub_an[j] + 1e-6) else "NO"
        print(f"    {nm:<14}  {pi_true_kt[j]:9.4f}  {lb_an[j]:9.4f}  "
              f"{ub_an[j]:9.4f}  {ub_an[j]-lb_an[j]:8.4f}  {in_b}")

    # Population vs sample AFDC marginal sanity check (so the user can see
    # how much sampling noise was being injected into the previous true-pi).
    print(f"\n  Population vs D=0 sample AFDC marginals (should agree as N -> inf):")
    sample_afdc = {
        '0n': p_dict['p00_c'], '1n': p_dict['p10_c'], '2n': p_dict['p20_c'],
        '0p': p_dict['p01_c'], '1p': p_dict['p11_c'], '2p': p_dict['p21_c'],
    }
    for s in STATE_NAMES:
        print(f"    {s:>3}: pop = {true_afdc[s]:.4f}   D=0 = {sample_afdc[s]:.4f}   "
              f"|diff| = {abs(true_afdc[s] - sample_afdc[s]):.4f}")

    # ---------------------------------------------------------------
    # KT Table 5 composite bounds: TRUE values + analytical LP bounds.
    # These are the population-level reference values that the per-config
    # CLP composite bounds (computed inside run_clp) should bracket.
    # ---------------------------------------------------------------
    print(f"\n[STEP 4b] KT Table 5 composite bounds: TRUE and analytical LP")
    true_comp = true_composites_from_types(types, TYPE_DEFS)
    ana_comp  = analytical_lp_composites(p_dict)
    print(f"    {'composite':<48}  {'true':>8}  {'AN_LB':>8}  "
          f"{'AN_UB':>8}  in_bounds")
    for cname in true_comp:
        tv = true_comp[cname]
        cmp_lb, cmp_ub = ana_comp[cname]
        in_b = "yes" if (cmp_lb - 1e-3 <= tv <= cmp_ub + 1e-3) else "NO"
        print(f"    {cname:<48}  {tv:8.4f}  {cmp_lb:8.4f}  "
              f"{cmp_ub:8.4f}  {in_b}")

    # ---------------------------------------------------------------
    # Build the BASE and ECON feature matrices.
    #   base = the 28 base covariates (matching CLP_final_group's COV_VARS)
    #   econ = base + ~400 squared/interaction features (engineer_econ_features)
    # The DGP carries explanatory power in BOTH the linear part of the base
    # features (X_MOD_SCALE * (work - welf)) and a NONLINEAR part of the
    # base features (X_NONLIN_SCALE * f(squares, interactions)).  Only the
    # 'econ' regime can recover the nonlinear part, so it should yield
    # higher first-stage R^2 and tighter CLP bounds in expectation.
    # ---------------------------------------------------------------
    print(f"\n[STEP 5] Build base and econ feature matrices")
    X_base = X
    X_econ_extra, econ_feat_names = engineer_econ_features(X_base)
    X_econ = np.hstack([X_base, X_econ_extra])
    print(f"  base regime: {X_base.shape[1]} features (28 base covariates)")
    print(f"  econ regime: {X_econ.shape[1]} features "
          f"(28 base + {X_econ_extra.shape[1]} squared/interaction terms)")

    # ---------------------------------------------------------------
    # STEP 6: run CLP for SIX first-stage configurations IN BOTH regimes.
    #   3 estimators x 2 modes x 2 regimes = 12 configurations total.
    # ---------------------------------------------------------------
    ESTIMATOR_CONFIGS: list = [
        ("OLS",   "full"),  ("OLS",   "cv"),
        ("Ridge", "full"),  ("Ridge", "cv"),
        ("LASSO", "full"),  ("LASSO", "cv"),
    ]
    REGIMES = [("base", X_base), ("econ", X_econ)]
    print(f"\n[STEP 6] CLP bounds across {len(REGIMES)} regimes "
          f"x {len(ESTIMATOR_CONFIGS)} estimator configs = "
          f"{len(REGIMES) * len(ESTIMATOR_CONFIGS)} runs total")
    print(f"  Note: 'full' modes are diagnostic -- CLP regularity needs cross-fit.")

    def _clip01(x):
        return max(0.0, min(1.0, float(x)))

    all_clp = {}     # keyed by f"{est}_{mode}_{regime}"
    for regime_name, X_used in REGIMES:
        for est, mode in ESTIMATOR_CONFIGS:
            label = f"{est}_{mode}_{regime_name}"
            print(f"\n  ============================================================")
            print(f"  Config {label}  ({X_used.shape[1]} features)")
            print(f"  ============================================================")
            all_clp[label] = run_clp(
                D, state_obs, X_used, person_id, pscorewt, p_dict,
                estimator=est, mode=mode,
                n_bootstrap=N_BOOTSTRAP, K=K_FOLDS, seed=RANDOM_SEED,
            )

    # ---------------------------------------------------------------
    # STEP 7: per-regime comparison tables (analytical + 6 CLP configs).
    # ---------------------------------------------------------------
    def _print_regime_tables(regime_name: str):
        print(f"\n\n{'='*120}")
        print(f"[STEP 7.{regime_name}] Per-regime comparison: regime = "
              f"{regime_name!r}  (pi-units, CLP clipped to [0,1])")
        print(f"{'='*120}")
        labels = [f"{e}_{m}_{regime_name}" for e, m in ESTIMATOR_CONFIGS]
        short = [f"{e}_{m}" for e, m in ESTIMATOR_CONFIGS]

        # 7a -- bounds
        print(f"\n  7a. Bounds [LB, UB]")
        header = f"  {'parameter':<14}  {'true':>7}  {'analytical':<14}"
        for s in short: header += f"  {s:<14}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for j, nm in enumerate(kt_names):
            truv = pi_true_kt[j]
            a_lb, a_ub = lb_an[j], ub_an[j]
            row = f"  {nm:<14}  {truv:7.4f}  [{a_lb:5.3f},{a_ub:5.3f}]"
            for lab in labels:
                r = all_clp[lab]
                c_lb = _clip01(r['lb_pi'][KT_ORDER_FROM_CLP][j])
                c_ub = _clip01(r['ub_pi'][KT_ORDER_FROM_CLP][j])
                row += f"  [{c_lb:5.3f},{c_ub:5.3f}]"
            print(row)

        # 7b -- widths
        print(f"\n  7b. Width comparison (smaller is tighter)")
        header = f"  {'parameter':<14}  {'analytical':>10}"
        for s in short: header += f"  {s:>11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for j, nm in enumerate(kt_names):
            a_w = ub_an[j] - lb_an[j]
            row = f"  {nm:<14}  {a_w:10.4f}"
            for lab in labels:
                r = all_clp[lab]
                c_lb = _clip01(r['lb_pi'][KT_ORDER_FROM_CLP][j])
                c_ub = _clip01(r['ub_pi'][KT_ORDER_FROM_CLP][j])
                row += f"  {c_ub - c_lb:11.4f}"
            print(row)

        # 7c -- vertex diversity (UB direction modal share)
        print(f"\n  7c. Vertex diversity (top vertex's share, UB direction)")
        header = f"  {'parameter':<14}"
        for s in short: header += f"  {s:>11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for j, nm in enumerate(BETA_NAMES):
            row = f"  {nm:<14}"
            for lab in labels:
                r = all_clp[lab]
                row += f"  {r['top_share_ub'][j]:>11.3f}"
            print(row)

    for regime_name, _ in REGIMES:
        _print_regime_tables(regime_name)

    # ---------------------------------------------------------------
    # STEP 8: BASE vs ECON head-to-head -- the core regime comparison.
    # ---------------------------------------------------------------
    print(f"\n\n{'='*120}")
    print(f"[STEP 8] BASE vs ECON head-to-head  "
          f"({X_base.shape[1]} features vs {X_econ.shape[1]} features)")
    print(f"{'='*120}")

    # 8a -- width side-by-side per estimator config
    print(f"\n  8a. CLP widths: base regime  vs  econ regime  (smaller is tighter)")
    header = (f"  {'parameter':<14}  {'analytical':>10}    " +
              "    ".join(f"{e+'_'+m+'  base / econ':<22}"
                          for e, m in ESTIMATOR_CONFIGS))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for j, nm in enumerate(kt_names):
        a_w = ub_an[j] - lb_an[j]
        row = f"  {nm:<14}  {a_w:10.4f}"
        for est, mode in ESTIMATOR_CONFIGS:
            r_b = all_clp[f"{est}_{mode}_base"]
            r_e = all_clp[f"{est}_{mode}_econ"]
            wb = (_clip01(r_b['ub_pi'][KT_ORDER_FROM_CLP][j])
                  - _clip01(r_b['lb_pi'][KT_ORDER_FROM_CLP][j]))
            we = (_clip01(r_e['ub_pi'][KT_ORDER_FROM_CLP][j])
                  - _clip01(r_e['lb_pi'][KT_ORDER_FROM_CLP][j]))
            row += f"    {wb:8.4f} / {we:8.4f}  "
        print(row)

    # 8b -- did econ TIGHTEN (yes), STAY same, or LOOSEN (no) vs base?
    print(f"\n  8b. Did econ regime TIGHTEN over base (smaller width)?  "
          f"(yes if econ < base by 1e-3)")
    header = f"  {'parameter':<14}"
    for est, mode in ESTIMATOR_CONFIGS:
        header += f"  {f'{est}_{mode}':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    tight_count = {f"{e}_{m}": 0 for e, m in ESTIMATOR_CONFIGS}
    for j, nm in enumerate(kt_names):
        row = f"  {nm:<14}"
        for est, mode in ESTIMATOR_CONFIGS:
            r_b = all_clp[f"{est}_{mode}_base"]
            r_e = all_clp[f"{est}_{mode}_econ"]
            wb = (_clip01(r_b['ub_pi'][KT_ORDER_FROM_CLP][j])
                  - _clip01(r_b['lb_pi'][KT_ORDER_FROM_CLP][j]))
            we = (_clip01(r_e['ub_pi'][KT_ORDER_FROM_CLP][j])
                  - _clip01(r_e['lb_pi'][KT_ORDER_FROM_CLP][j]))
            tag = ("yes" if we < wb - 1e-3
                   else ("same" if abs(we - wb) < 1e-3 else "no"))
            if tag == "yes":
                tight_count[f"{est}_{mode}"] += 1
            row += f"  {tag:>14}"
        print(row)
    print(f"\n  Total betas where econ tightened over base, by config:")
    for est, mode in ESTIMATOR_CONFIGS:
        print(f"    {est}_{mode}: {tight_count[f'{est}_{mode}']:>2} / "
              f"{len(kt_names)}")

    # 8c -- mean vertex-diversity by regime (rows = configs, cols = regimes)
    print(f"\n  8c. Mean top-vertex share (UB direction) by regime")
    print(f"     Lower means more vertex diversity = more Jensen tightening")
    print(f"  {'config':<13}  {'base':>8}  {'econ':>8}")
    print("  " + "-" * 31)
    for est, mode in ESTIMATOR_CONFIGS:
        b_share = float(np.mean(all_clp[f"{est}_{mode}_base"]['top_share_ub']))
        e_share = float(np.mean(all_clp[f"{est}_{mode}_econ"]['top_share_ub']))
        print(f"  {est+'_'+mode:<13}  {b_share:8.3f}  {e_share:8.3f}")

    # 8d -- Composite bounds across all configs vs analytical and truth.
    print(f"\n\n{'='*120}")
    print(f"[STEP 8d] KT Table 5 composite bounds: TRUE vs analytical LP "
          f"vs CLP per config")
    print(f"{'='*120}")
    composite_names = list(true_comp.keys())
    for cname in composite_names:
        print(f"\n  {cname}")
        print(f"    TRUE        : {true_comp[cname]:.4f}")
        cmp_lb, cmp_ub = ana_comp[cname]
        in_an = "yes" if (cmp_lb - 1e-3 <= true_comp[cname] <= cmp_ub + 1e-3) else "NO"
        print(f"    Analytical  : [{cmp_lb:.4f}, {cmp_ub:.4f}]  "
              f"width = {cmp_ub-cmp_lb:.4f}  contains true? {in_an}")
        print(f"    CLP per config (LB / UB / width / contains true?):")
        print(f"    {'config':<22}  {'LB':>8}  {'UB':>8}  {'width':>8}  in?")
        for regime_name, _ in REGIMES:
            for est, mode in ESTIMATOR_CONFIGS:
                lab = f"{est}_{mode}_{regime_name}"
                c = all_clp[lab].get('composite', {}).get(cname)
                if c is None:
                    print(f"    {lab:<22}  {'N/A':>8}")
                    continue
                lb = c['lb_clip']; ub = c['ub_clip']
                in_clp = "yes" if (lb - 1e-2 <= true_comp[cname] <= ub + 1e-2) else "NO"
                print(f"    {lab:<22}  {lb:8.4f}  {ub:8.4f}  "
                      f"{ub-lb:8.4f}  {in_clp}")

    # ---------------------------------------------------------------
    # STEP 9: save CSV summary including BOTH regimes.
    # ---------------------------------------------------------------
    summary_rows = []
    for j, nm in enumerate(kt_names):
        truv = pi_true_kt[j]
        a_lb, a_ub = lb_an[j], ub_an[j]
        rec = dict(parameter=nm, true_pi=truv,
                   an_lb=a_lb, an_ub=a_ub, an_width=a_ub - a_lb)
        clp_j = KT_ORDER_FROM_CLP[j]
        for regime_name, _ in REGIMES:
            for est, mode in ESTIMATOR_CONFIGS:
                r = all_clp[f"{est}_{mode}_{regime_name}"]
                c_lb = _clip01(r['lb_pi'][clp_j])
                c_ub = _clip01(r['ub_pi'][clp_j])
                rec[f"{est}_{mode}_{regime_name}_lb"] = c_lb
                rec[f"{est}_{mode}_{regime_name}_ub"] = c_ub
                rec[f"{est}_{mode}_{regime_name}_width"] = c_ub - c_lb
                rec[f"{est}_{mode}_{regime_name}_top_share_ub"] = (
                    r['top_share_ub'][clp_j])
                rec[f"{est}_{mode}_{regime_name}_n_vertices_ub"] = (
                    r['n_vertices_ub'][clp_j])
        summary_rows.append(rec)
    out_path = ("/Users/gevorgkhandamiryan/Desktop/cursorclp/check/"
                "synthetic_simulation_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    print(f"\n  Wrote summary CSV to {out_path}")

    print(f"\nTotal runtime: {(time.time() - t0):.1f}s")


if __name__ == "__main__":
    main()
