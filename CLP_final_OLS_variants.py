"""
CLP_final_OLS_variants.py
=========================
A version of CLP_final but with OLS-only first stage, comparing four
sample-splitting / covariate-subset specifications.

Specifications
--------------
1. spec1_full_panel
     OLS on the FULL panel (woman x quarter), in-sample prediction.
     No sample splitting at all.  ~29,000 person-quarter rows.

2. spec2_cross_section_Q{quarter}
     OLS on a single CROSS-SECTIONAL slice of women (one post-RA quarter,
     ~4,461 women).  No sample splitting.  Each woman appears once.

3. spec3_groupkfold_K{3,5,7}
     GroupKFold cross-fitting at the WOMAN level.
     Each woman's quarters all stay in the same fold, eliminating
     within-person leakage.  Run with K = 3, 5, 7.

4. spec4_cov_subset_{demo, earn, welf, all}
     OLS on the full panel with NO sample splitting, but using DIFFERENT
     covariate subsets:
       - demo : demographic only (age, race, marital, education, kids,
                missing flags) — 16 vars
       - earn : earnings history only (ernpq1..8 + anyernpq1..8) — 16 vars
       - welf : welfare history only (adcpq1..8 + fstpq1..7 + prevafdc) — 16 vars
       - all  : the full 28 baseline covariates (matches the default in
                CLP_final.py)

Common pipeline (all specs)
---------------------------
- A matrix      : the deterministic 5x9 +-1 conservation matrix from
                  CLP_final.build_A() (beta-parameterization, per CLP paper Sec. 4)
- B vector      : per-i 5-vector of IPW-weighted state indicators
                  (compute_B from CLP_final)
- Vertex enum   : C(9,5)=126 candidates via enumerate_dual_vertices
- Bound est.    : sigma_hat(q) = (1/N) sum_i nu_hat(X_i)' B_i
                  computed for both q = +e_j (UB) and q = -e_j (LB)
- Bootstrap CI  : person-clustered Exp(1) multiplier bootstrap, 200 draws
- beta -> pi    : post-hoc division by AFDC source-state marginal P^a(s^a)

Output
------
Prints per-spec bound tables and a final side-by-side comparison.
Saves nothing to disk by default.

Usage
-----
    python3 CLP_final_OLS_variants.py

Expected runtime: ~3-5 minutes on a modern laptop (with N_BOOTSTRAP=200).

Author/notes
------------
This file imports machinery (data prep, A, B, vertex enumeration, bootstrap)
from CLP_final.py to keep the comparison strictly controlled.  The only
change is the first-stage estimator and the splitting strategy.
"""

import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

import warnings
warnings.filterwarnings("ignore")

# Reuse machinery from CLP_final.  This pulls in constants, data prep,
# A matrix, B vector, vertex enumeration, multiplier bootstrap.
from CLP_final import (
    PSCORE_VARS, COV_VARS,
    BETA_NAMES, PI_NAMES,
    S_0N, S_1N, S_2N, S_0P, S_1P, S_2P,
    JF_DTA_PATH, POLICY_RULES_PATH, TABLE4_MAT_PATH,
    fit_pscore_logit, load_table4_mat, prepare_jf_data,
    build_A, compute_B,
    enumerate_dual_vertices, clp_estimate, multiplier_bootstrap_ci,
)

# =============================================================================
# Constants
# =============================================================================
N_BOOTSTRAP = 200       # raise to >=1000 for publication-quality CIs
RANDOM_SEED = 42

# Default cross-section quarter for spec 2 (post-RA quarter, 1..7)
DEFAULT_CROSSSEC_QUARTER = 4

# K values for spec 3
SPEC3_K_VALUES = (3, 5, 7)

# =============================================================================
# Covariate subsets for spec 4
#
# Variable-name conventions match KT.do / GetJFData.do:
#   ernpqK     = quarterly earnings in pre-RA quarter K (k = 1..8)
#   anyernpqK  = indicator that ernpqK > 0
#   adcpqK     = quarterly AFDC payment in pre-RA quarter K
#   fstpqK     = quarterly food-stamp payment in pre-RA quarter K
#   prevafdc   = previous AFDC enrollment indicator
#   age2534, agelt25, white, black, hisp, marnvr, marapt,
#   hsged, nohsged, kidcount, yngchtru, applcant, yremp,
#   misshs, misskidctgt2, missmar : demographic / missing-flag dummies.
# =============================================================================
COV_SUBSETS = {
    "demo": [
        "age2534", "agelt25", "white", "black", "hisp",
        "marnvr", "marapt", "hsged", "nohsged",
        "kidcount", "yngchtru", "applcant", "yremp",
        "misshs", "misskidctgt2", "missmar",
    ],
    "earn": (
        [f"ernpq{q}" for q in range(1, 9)]
        + [f"anyernpq{q}" for q in range(1, 9)]
    ),
    "welf": (
        [f"adcpq{q}" for q in range(1, 9)]
        + [f"fstpq{q}" for q in range(1, 8)]
        + ["prevafdc"]
    ),
    "all": COV_VARS,        # the default 28 baseline covariates
}


# =============================================================================
# FIRST-STAGE ESTIMATORS
# =============================================================================
def estimate_b0_no_split(D, state_obs, X_raw, pscorewt):
    """
    NO sample splitting: fit OLS on the full sample, predict in-sample.

    For PARAMETRIC in-sample first stages the CLP paper (Semenova, Section 4)
    explicitly notes that cross-fitting can be skipped; the o_P(N^{-1/4})
    rate condition (Assumption 4.1) is achieved trivially by parametric
    OLS under standard assumptions.

    Returns
    -------
    b_hat : (N, 5) array  cross-fitted (here: in-sample) E[B|X] estimates
    B_obs : (N, 5) array  realized B vectors (IPW-weighted state indicators)
    """
    B_obs = compute_B(D, state_obs, pscorewt=pscorewt)
    n, k = B_obs.shape
    b_hat = np.zeros((n, k))

    sc = StandardScaler()
    X_std = sc.fit_transform(X_raw)

    for j in range(k):
        y = B_obs[:, j]
        if y.std() < 1e-10:
            b_hat[:, j] = y.mean()
            continue
        ols = LinearRegression(n_jobs=1)
        ols.fit(X_std, y)
        b_hat[:, j] = ols.predict(X_std)

    return b_hat, B_obs


def estimate_b0_groupkfold(D, state_obs, X_raw, pscorewt, person_id, K):
    """
    GroupKFold cross-fitting with K folds, grouped by person id.
    Each woman's quarters all stay within the same fold.

    Parameters
    ----------
    K : int   number of folds (3, 5, or 7)

    Returns
    -------
    b_hat : (N, 5) array
    B_obs : (N, 5) array
    """
    B_obs = compute_B(D, state_obs, pscorewt=pscorewt)
    n, k = B_obs.shape
    b_hat = np.zeros((n, k))

    gkf = GroupKFold(n_splits=K)
    for fold_i, (train_idx, test_idx) in enumerate(
        gkf.split(X_raw, B_obs[:, 0], groups=person_id)
    ):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_raw[train_idx])
        X_te = sc.transform(X_raw[test_idx])

        for j in range(k):
            y = B_obs[train_idx, j]
            if y.std() < 1e-10:
                b_hat[test_idx, j] = y.mean()
                continue
            ols = LinearRegression(n_jobs=1)
            ols.fit(X_tr, y)
            b_hat[test_idx, j] = ols.predict(X_te)

    return b_hat, B_obs


# =============================================================================
# KT TABLE 5 COMPOSITE BOUNDS  (Take-Up Work, Take-Up Welfare, Exit 0r)
# =============================================================================
# Reference: Bounds.m lines 527-678.  KT solve LP min/max f' pi subject to the
# 5x9 conservation system A pi = b, 0 <= pi <= 1.  In CLP beta-parameterization
# the same composite is q_beta = f / P^a(source) (or equivalently each f_k
# divided by the AFDC source marginal of pi_k's source state), which converts
# the LP target to a linear combination of betas.  In CLP BETA_NAMES order:
#
#     [beta(0n,1r), beta(0r,0n), beta(2n,1r), beta(0r,2n), beta(0r,1r),
#      beta(0r,1n), beta(1n,1r), beta(0r,2u), beta(2u,1r)]
#
# the q_beta vectors are:
#
#  TUW    : [1, 0, 0, 1, 1, 1, 0, 1, 0] / (p00_c + p01_c)
#           = (β(0n,1r) + β(0r,2n) + β(0r,1r) + β(0r,1n) + β(0r,2u))
#                      / P^A(not working)
#           NOTE: KT (line 550) note this is point-identified.
#
#  TUWelf : [1, 0, 1, 0, 0, 0, 1, 0, 0] / (p00_c + p10_c + p20_c)
#           = (β(0n,1r) + β(2n,1r) + β(1n,1r)) / P^A(off welfare)
#
#  Exit   : [0, 1, 0, 1, 0, 1, 0, 0, 0] / p01_c
#           = (β(0r,0n) + β(0r,2n) + β(0r,1n)) / P^A(0p)  -- "Exit Welfare 2"
#                                                            in Bounds.m line 660.
def composite_q_vectors_kt(table4_p):
    """Three KT-Table-5 composite q-vectors in CLP beta-parameterization.
    Each q is a 9-vector that can be passed to clp_estimate as the LP
    target.  Returns dict {composite_name: q_beta}."""
    p00_c = table4_p['p00_c']; p01_c = table4_p['p01_c']
    p10_c = table4_p['p10_c']; p20_c = table4_p['p20_c']

    n_beta = 9
    q_tuw   = np.zeros(n_beta)
    q_tuwelf = np.zeros(n_beta)
    q_exit  = np.zeros(n_beta)

    # TUW: starts not-working (0n or 0r), ends working (anywhere)
    # Active beta indices (CLP order): 0=beta(0n,1r), 3=beta(0r,2n),
    #                                  4=beta(0r,1r), 5=beta(0r,1n),
    #                                  7=beta(0r,2u)
    for j in (0, 3, 4, 5, 7):
        q_tuw[j] = 1.0
    q_tuw /= (p00_c + p01_c)

    # TUWelf: starts off-welfare, ends on-welfare (with low earnings = 1r)
    # Active beta indices: 0=beta(0n,1r), 2=beta(2n,1r), 6=beta(1n,1r)
    for j in (0, 2, 6):
        q_tuwelf[j] = 1.0
    q_tuwelf /= (p00_c + p10_c + p20_c)

    # Exit 0r: starts on-welfare zero-earn (0r), ends off-welfare
    # Active beta indices: 1=beta(0r,0n), 3=beta(0r,2n), 5=beta(0r,1n)
    for j in (1, 3, 5):
        q_exit[j] = 1.0
    q_exit /= p01_c

    return {
        "TUW (Take-Up Work)":         q_tuw,
        "TUWelf (Take-Up Welfare)":   q_tuwelf,
        "Exit 0r (on-welfare zero earn -> off welfare)": q_exit,
    }


# =============================================================================
# COMMON CLP PIPELINE
# =============================================================================
def run_clp_pipeline(D, state_obs, b_hat, B_obs, person_id, label,
                      table4_p=None):
    """
    Full CLP estimation given first-stage predictions.

    Steps:
      1. Build deterministic 5x9 A matrix (build_A from CLP_final).
      2. For each j in {0..8}:
           - q_up = +e_j  -> compute UB on beta_j via clp_estimate.
           - q_dn = -e_j  -> compute LB on beta_j (negate result).
           - Multiplier bootstrap CI for each (person-clustered).
      3. beta -> pi: divide by source-state AFDC marginal P^a(s^a_j),
         which is loaded from Table4_mat_python.txt.

    Returns dict mapping beta name -> dict(lb, ub, width, ci_lb, ci_ub,
                                            lb_pi, ub_pi, ci_lb_pi, ci_ub_pi).
    """
    A_mat = build_A()
    if table4_p is None:
        table4_p, _ = load_table4_mat()

    # source_pop matched to BETA_NAMES order in CLP_final.py:
    #   ["beta(0n,1r)", "beta(0r,0n)", "beta(2n,1r)", "beta(0r,2n)",
    #    "beta(0r,1r)", "beta(0r,1n)", "beta(1n,1r)", "beta(0r,2u)",
    #    "beta(2u,1r)"]
    # For each: source = AFDC-arm marginal of the source state.
    # Note: index 8 (2u source) uses p21_c as a proxy for the unobserved 2u
    # marginal; the resulting "pi" should be interpreted as beta/p_{2p,c}.
    source_pop = np.array([
        table4_p['p00_c'], table4_p['p01_c'], table4_p['p20_c'],
        table4_p['p01_c'], table4_p['p01_c'], table4_p['p01_c'],
        table4_p['p10_c'], table4_p['p01_c'], table4_p['p21_c'],
    ])

    n_pi = 9
    results = {}

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  Sample size: {len(D):,}    "
          f"unique persons: {len(np.unique(person_id)):,}")
    print(f"  {'beta param':<13}  {'LB':>8}  {'UB':>8}  {'width':>8}    "
          f"{'95% CI(LB)':>16}    {'95% CI(UB)':>16}")
    print("  " + "-" * 78)

    for j in range(n_pi):
        q_up = np.zeros(n_pi); q_up[j] =  1.0
        q_dn = np.zeros(n_pi); q_dn[j] = -1.0

        ub_hat, c_up, _, _, _ = clp_estimate(q_up, b_hat, B_obs, A_mat)
        ci_ub = list(multiplier_bootstrap_ci(
            c_up, n_bs=N_BOOTSTRAP, person_id=person_id))

        neg_lb, c_dn, _, _, _ = clp_estimate(q_dn, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb
        ci_lb = [-x for x in
                 multiplier_bootstrap_ci(
                     c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]

        # NOTE: we deliberately do NOT clip beta to [0,1] here.
        # The user can post-hoc clip in summary tables if desired.
        width = ub_hat - lb_hat

        # beta -> pi conversion
        sp = source_pop[j]
        if sp > 1e-12:
            lb_pi = lb_hat / sp
            ub_pi = ub_hat / sp
            ci_lb_pi = (ci_lb[0] / sp, ci_lb[1] / sp)
            ci_ub_pi = (ci_ub[0] / sp, ci_ub[1] / sp)
        else:
            lb_pi = ub_pi = float("nan")
            ci_lb_pi = ci_ub_pi = (float("nan"), float("nan"))

        results[BETA_NAMES[j]] = dict(
            name=BETA_NAMES[j],
            lb=lb_hat, ub=ub_hat, width=width,
            ci_lb=tuple(ci_lb), ci_ub=tuple(ci_ub),
            lb_pi=lb_pi, ub_pi=ub_pi,
            ci_lb_pi=ci_lb_pi, ci_ub_pi=ci_ub_pi,
        )
        print(f"  {BETA_NAMES[j]:<13}  {lb_hat:8.4f}  {ub_hat:8.4f}  "
              f"{width:8.4f}    "
              f"[{ci_lb[0]:6.3f},{ci_lb[1]:6.3f}]    "
              f"[{ci_ub[0]:6.3f},{ci_ub[1]:6.3f}]")

    # Print pi-units rescaling
    print(f"\n  -- pi-units (beta / P^a(source)) --")
    print(f"  {'pi param':<13}  {'LB_pi':>8}  {'UB_pi':>8}  {'width':>8}")
    for nm, res in results.items():
        pi_w = res['ub_pi'] - res['lb_pi']
        idx = BETA_NAMES.index(nm)
        print(f"  {PI_NAMES[idx]:<13}  {res['lb_pi']:8.4f}  "
              f"{res['ub_pi']:8.4f}  {pi_w:8.4f}")

    # ----------------------------------------------------------
    # KT Table 5 composite bounds: TUW, TUWelf, Exit 0r (in pi-units).
    # Same LP machinery as the individual bounds, just with q = q_beta
    # for the composite (a 9-vector with weights; see composite_q_vectors_kt).
    # ----------------------------------------------------------
    print(f"\n  -- KT Table 5 composite bounds (pi-units, clipped to [0,1]) --")
    print(f"  {'composite':<48}  {'LB':>8}  {'UB':>8}  {'width':>8}    "
          f"{'95% CI(LB)':>16}    {'95% CI(UB)':>16}")
    print("  " + "-" * 110)
    composite_results = {}
    for cname, q_beta in composite_q_vectors_kt(table4_p).items():
        ub_hat, c_up, _, _, _ = clp_estimate(q_beta, b_hat, B_obs, A_mat)
        ci_ub_c = list(multiplier_bootstrap_ci(
            c_up, n_bs=N_BOOTSTRAP, person_id=person_id))
        neg_lb_c, c_dn, _, _, _ = clp_estimate(-q_beta, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb_c
        ci_lb_c = [-x for x in
                   multiplier_bootstrap_ci(
                       c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]
        # Clip for display (KT also clips to [0, 1]).
        lb_clip = max(0.0, min(1.0, lb_hat))
        ub_clip = max(0.0, min(1.0, ub_hat))
        composite_results[cname] = dict(
            lb=lb_hat, ub=ub_hat, width=ub_hat - lb_hat,
            lb_clip=lb_clip, ub_clip=ub_clip,
            ci_lb=tuple(ci_lb_c), ci_ub=tuple(ci_ub_c),
        )
        print(f"  {cname:<48}  {lb_clip:8.4f}  {ub_clip:8.4f}  "
              f"{ub_clip - lb_clip:8.4f}    "
              f"[{max(0.,ci_lb_c[0]):.3f},{min(1.,ci_lb_c[1]):.3f}]    "
              f"[{max(0.,ci_ub_c[0]):.3f},{min(1.,ci_ub_c[1]):.3f}]")

    # Stash composites alongside individual results so the caller can
    # build a unified summary; key composites under their full name to
    # avoid colliding with BETA_NAMES.
    results['_composites'] = composite_results
    return results


# =============================================================================
# SPEC RUNNERS
# =============================================================================
def run_spec1_full_panel(D, state_obs, df_incl, person_id, pscorewt,
                          table4_p):
    """Spec 1: no sample splitting, full panel."""
    print("\n\n" + "#"*80)
    print("# SPEC 1: No sample splitting -- full panel (woman x quarter)")
    print("#"*80)

    avail = [v for v in COV_VARS if v in df_incl.columns]
    X_raw = df_incl[avail].fillna(0).to_numpy(float)

    b_hat, B_obs = estimate_b0_no_split(D, state_obs, X_raw, pscorewt)
    return run_clp_pipeline(
        D, state_obs, b_hat, B_obs, person_id,
        f"Spec 1: full panel, OLS, no split ({len(avail)} covariates)",
        table4_p=table4_p,
    )


def run_spec2_cross_section(D, state_obs, df_incl, person_id, pscorewt,
                             table4_p, quarter=DEFAULT_CROSSSEC_QUARTER):
    """
    Spec 2: no sample splitting, cross-sectional slice.

    Filters df_incl to the specified post-RA quarter. Note: not every woman
    is `included` in every quarter (women with mixed welfare status are
    excluded), so the slice may have fewer than 4,461 rows.
    """
    print("\n\n" + "#"*80)
    print(f"# SPEC 2: No sample splitting -- cross-section (Q{quarter})")
    print("#"*80)

    quarter_arr = df_incl['quarter'].to_numpy(dtype=int)
    mask = quarter_arr == quarter
    n_slice = mask.sum()
    print(f"  Filtering to quarter {quarter}: {n_slice:,} women in this slice")
    if n_slice < 100:
        print(f"  [WARN] Slice has only {n_slice} rows; spec 2 may be unreliable.")

    D_x = D[mask]
    state_x = state_obs[mask]
    person_x = person_id[mask]
    pscorewt_x = pscorewt[mask]
    df_x = df_incl[mask].reset_index(drop=True)

    avail = [v for v in COV_VARS if v in df_x.columns]
    X_raw = df_x[avail].fillna(0).to_numpy(float)

    b_hat, B_obs = estimate_b0_no_split(D_x, state_x, X_raw, pscorewt_x)
    return run_clp_pipeline(
        D_x, state_x, b_hat, B_obs, person_x,
        f"Spec 2: cross-section Q{quarter}, OLS, no split ({len(avail)} covariates)",
        table4_p=table4_p,
    )


def run_spec3_groupkfold(D, state_obs, df_incl, person_id, pscorewt,
                          table4_p, K_values=SPEC3_K_VALUES):
    """Spec 3: GroupKFold (by woman ID) with K = 3, 5, 7."""
    print("\n\n" + "#"*80)
    print(f"# SPEC 3: GroupKFold by woman ID -- K = {list(K_values)}")
    print("#"*80)

    avail = [v for v in COV_VARS if v in df_incl.columns]
    X_raw = df_incl[avail].fillna(0).to_numpy(float)

    results_by_K = {}
    for K in K_values:
        print(f"\n  --- K = {K} ---")
        b_hat, B_obs = estimate_b0_groupkfold(
            D, state_obs, X_raw, pscorewt, person_id, K
        )
        results_by_K[K] = run_clp_pipeline(
            D, state_obs, b_hat, B_obs, person_id,
            f"Spec 3 (K={K}): GroupKFold by woman ID, OLS ({len(avail)} covariates)",
            table4_p=table4_p,
        )
    return results_by_K


def run_spec4_cov_subsets(D, state_obs, df_incl, person_id, pscorewt,
                           table4_p, subsets_to_run=None):
    """Spec 4: no sample splitting, distinct covariate subsets."""
    print("\n\n" + "#"*80)
    print(f"# SPEC 4: No sample splitting -- distinct covariate subsets")
    print("#"*80)

    if subsets_to_run is None:
        subsets_to_run = list(COV_SUBSETS.keys())

    results_by_subset = {}
    for subset_name in subsets_to_run:
        cov_list = COV_SUBSETS[subset_name]
        avail = [v for v in cov_list if v in df_incl.columns]
        if not avail:
            print(f"  [WARN] No covariates available for subset '{subset_name}', skipping")
            continue
        X_raw = df_incl[avail].fillna(0).to_numpy(float)
        print(f"\n  --- subset = '{subset_name}' ({len(avail)} variables) ---")
        print(f"     {avail[:6]}{'...' if len(avail) > 6 else ''}")

        b_hat, B_obs = estimate_b0_no_split(D, state_obs, X_raw, pscorewt)
        results_by_subset[subset_name] = run_clp_pipeline(
            D, state_obs, b_hat, B_obs, person_id,
            f"Spec 4 ({subset_name}): no split, OLS, {len(avail)} covariates",
            table4_p=table4_p,
        )
    return results_by_subset


# =============================================================================
# SUMMARY TABLE
# =============================================================================
def print_summary(all_results):
    """Side-by-side comparison of bound widths across all specs."""
    print("\n\n" + "="*120)
    print("SUMMARY: beta interval widths [UB - LB] across specifications")
    print("Tighter (smaller) widths = more informative bounds.")
    print("="*120)

    spec_names = list(all_results.keys())
    if not spec_names:
        print("  No results to summarize.")
        return

    col_w = 14
    print(f"\n  {'beta param':<13}  " +
          "  ".join(f"{n[:col_w]:>{col_w}}" for n in spec_names))
    print("  " + "-" * (13 + (col_w + 2) * len(spec_names)))

    for beta in BETA_NAMES:
        cells = []
        for sname in spec_names:
            res = all_results[sname].get(beta)
            cells.append(
                f"{res['width']:>{col_w}.4f}" if res
                else f"{'N/A':>{col_w}}"
            )
        print(f"  {beta:<13}  " + "  ".join(cells))

    print("\n" + "="*120)
    print("SUMMARY: pi interval widths [UB_pi - LB_pi] across specifications")
    print("="*120)
    print(f"\n  {'pi param':<13}  " +
          "  ".join(f"{n[:col_w]:>{col_w}}" for n in spec_names))
    print("  " + "-" * (13 + (col_w + 2) * len(spec_names)))
    for j, beta in enumerate(BETA_NAMES):
        cells = []
        for sname in spec_names:
            res = all_results[sname].get(beta)
            if res is None or np.isnan(res['lb_pi']):
                cells.append(f"{'N/A':>{col_w}}")
            else:
                w_pi = res['ub_pi'] - res['lb_pi']
                cells.append(f"{w_pi:>{col_w}.4f}")
        print(f"  {PI_NAMES[j]:<13}  " + "  ".join(cells))

    # KT Table 5 composite bounds across specs (LB / UB / width).
    print("\n" + "="*120)
    print("SUMMARY: KT Table 5 composite bounds across specifications "
          "(pi-units, clipped to [0,1])")
    print("="*120)
    # Composite names (drawn from any one spec; they should all agree).
    sample = next(iter(all_results.values()))
    composite_names = list(sample.get('_composites', {}).keys())
    if not composite_names:
        print("  No composite results to summarize.")
        return
    short = {
        "TUW (Take-Up Work)":         "TUW",
        "TUWelf (Take-Up Welfare)":   "TUWelf",
        "Exit 0r (on-welfare zero earn -> off welfare)": "Exit 0r",
    }
    for cname in composite_names:
        print(f"\n  {short.get(cname, cname[:20])}: {cname}")
        print(f"  {'spec':<13}  {'LB':>8}  {'UB':>8}  {'width':>8}    "
              f"{'95% CI(LB)':>16}    {'95% CI(UB)':>16}")
        for sname in spec_names:
            comp = all_results[sname].get('_composites', {}).get(cname)
            if comp is None:
                print(f"  {sname[:13]:<13}  {'N/A':>8}")
                continue
            print(f"  {sname[:13]:<13}  {comp['lb_clip']:8.4f}  "
                  f"{comp['ub_clip']:8.4f}  "
                  f"{comp['ub_clip'] - comp['lb_clip']:8.4f}    "
                  f"[{max(0., comp['ci_lb'][0]):.3f},"
                  f"{min(1., comp['ci_lb'][1]):.3f}]    "
                  f"[{max(0., comp['ci_ub'][0]):.3f},"
                  f"{min(1., comp['ci_ub'][1]):.3f}]")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*80)
    print("CLP_final OLS variants -- comparing four sample-splitting strategies")
    print("="*80)
    print(f"  N_BOOTSTRAP = {N_BOOTSTRAP}")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print(f"  Spec 2 cross-section quarter = Q{DEFAULT_CROSSSEC_QUARTER}")
    print(f"  Spec 3 K values = {SPEC3_K_VALUES}")
    print(f"  Spec 4 covariate subsets = {list(COV_SUBSETS.keys())}")

    np.random.seed(RANDOM_SEED)

    # --- Load data once (expensive: pscore logit is fit here) ---
    print("\n" + "-"*80)
    print("Loading JF data (this is the ONLY data load; reused across all specs)")
    print("-"*80)
    D, state_obs, df_incl, person_id, pscorewt = prepare_jf_data()
    table4_p, _ = load_table4_mat()

    all_results = {}

    # --- Spec 1 ---
    np.random.seed(RANDOM_SEED)
    all_results["Spec1_full_panel"] = run_spec1_full_panel(
        D, state_obs, df_incl, person_id, pscorewt, table4_p
    )

    # --- Spec 2 ---
    np.random.seed(RANDOM_SEED)
    all_results[f"Spec2_Q{DEFAULT_CROSSSEC_QUARTER}_cross"] = run_spec2_cross_section(
        D, state_obs, df_incl, person_id, pscorewt, table4_p,
        quarter=DEFAULT_CROSSSEC_QUARTER,
    )

    # --- Spec 3 ---
    np.random.seed(RANDOM_SEED)
    spec3_res = run_spec3_groupkfold(
        D, state_obs, df_incl, person_id, pscorewt, table4_p,
        K_values=SPEC3_K_VALUES,
    )
    for K, res in spec3_res.items():
        all_results[f"Spec3_K{K}"] = res

    # --- Spec 4 ---
    np.random.seed(RANDOM_SEED)
    spec4_res = run_spec4_cov_subsets(
        D, state_obs, df_incl, person_id, pscorewt, table4_p,
        subsets_to_run=list(COV_SUBSETS.keys()),
    )
    for sname, res in spec4_res.items():
        all_results[f"Spec4_{sname}"] = res

    # --- Final summary ---
    print_summary(all_results)


if __name__ == "__main__":
    main()
