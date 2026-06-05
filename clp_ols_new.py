"""
clp_ols_new.py
==============
EXACT clone of CLP_final_OLS_variants.py with ONE change: the source
marginal P^a(s) — used for both (a) the β→π conversion of individual
cells and (b) the q-vector normalization of KT composites (TUW, TUWelf,
Exit 0r) — is computed from the SAMPLE via Horvitz–Thompson IPW on the
AFDC arm, rather than read from KT's pre-computed Table 4
(`Table4_mat_python.txt`).

Motivation
----------
`CLP_final_OLS_variants.py` reads `table4_p['p00_c']`, `p01_c`, `p10_c`,
`p20_c`, `p21_c` from the KT replication `.mat` file and uses them
everywhere a source marginal is needed.  These are accurate but they
come from KT's own filtered sample (their Stata code on the full
pre-filtered data) and differ from in-sample IPW estimates by ~0.4%.

`clp_ols.py` already uses sample IPW for the per-cell β→π conversion,
but inconsistently — it still imports KT's `p_kt` for the composite
q-vector denominators.

This file implements the **internally consistent** sample-IPW variant
of CLP_final_OLS_variants: every place that needed P^a(s) now reads
the same in-sample IPW estimator built from the current `(D, state_obs,
pscorewt)` triple.

What is identical to CLP_final_OLS_variants.py
----------------------------------------------
- The four OLS specs (spec1_full_panel, spec2_cross_section,
  spec3_groupkfold, spec4_cov_subsets)
- The first-stage estimators (estimate_b0_no_split, estimate_b0_groupkfold)
- The A matrix (CLP_final.build_A())
- The B-vector definition (CLP_final.compute_B)
- The CLP plug-in (CLP_final.clp_estimate via vertex enumeration)
- The person-clustered Exp(1) multiplier bootstrap
- The KT composite q vectors (TUW, TUWelf, Exit) at positions
  {0,3,4,5,7}, {0,2,6}, {1,3,5} respectively — these are the KT-faithful
  q vectors that include β(0r,1r), matching Bounds.m

What is different
-----------------
- Every reference to `table4_p['p**_c']` is replaced by the IPW estimate
  `ipw_source_marginal(state_code, D, state_obs, pi)`
- The `pi` propensity score is derived from `pscorewt` and `D` (the same
  formula as clp_ols.py: pi = 1/pscorewt if D=1 else 1 - 1/pscorewt,
  clipped to [0.02, 0.98]).

Outputs
-------
Same console layout as CLP_final_OLS_variants.py.  No files written.

Usage
-----
    python3 clp_ols_new.py
"""

import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

import warnings
warnings.filterwarnings("ignore")

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
# Constants  (identical to CLP_final_OLS_variants.py)
# =============================================================================
N_BOOTSTRAP = 200
RANDOM_SEED = 42

DEFAULT_CROSSSEC_QUARTER = 4
SPEC3_K_VALUES = (3, 5, 7)

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
    "all": COV_VARS,
}


# =============================================================================
# FIRST-STAGE ESTIMATORS  (identical to CLP_final_OLS_variants.py)
# =============================================================================
def estimate_b0_no_split(D, state_obs, X_raw, pscorewt):
    """No sample splitting; fit OLS on the full sample, predict in-sample."""
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
    """GroupKFold cross-fit OLS, grouped by person id."""
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
# IN-SAMPLE IPW SOURCE MARGINALS  (THIS IS THE NEW PART)
# =============================================================================
# Replaces every `table4_p['p**_c']` lookup in CLP_final_OLS_variants.py
# with an IPW estimator computed from the current sample.  Same Horvitz–
# Thompson formula used by clp_ols.py: ipw_source_marginal.
#
# State codes (from CLP_final.py): S_0N=0, S_1N=1, S_2N=2, S_0P=3, S_2P=5.
# The reference state 1p (S_1P=4) is excluded; '0r' shares cells with '0p'
# and '2u' shares with '2p' (no underreporting separation).
# =============================================================================
def derive_pi_from_pscorewt(D, pscorewt):
    """Recover π = Pr(D=1 | X) from `pscorewt` using the KT IPW convention.

    pscorewt is built as 1/π for treated and 1/(1-π) for controls, so
    inverting gives π.  Clipped to [0.02, 0.98] for numerical safety.
    """
    pi = np.where(D == 1, 1.0 / pscorewt, 1.0 - 1.0 / pscorewt)
    return np.clip(pi, 0.02, 0.98)


def ipw_source_marginal(state_code_value, D, state_obs, pi):
    """P̂^A(state = state_code_value) on the AFDC arm via Horvitz–Thompson:
        (1/N) Σ_i (1 - W_i)/(1 - π_i) · 1{state_i = state_code_value}
    """
    contribs = np.where(D == 0,
                        (state_obs == state_code_value).astype(float) / (1.0 - pi),
                        0.0)
    return float(contribs.mean())


def all_source_marginals(D, state_obs, pi):
    """Compute IPW source marginals for the 5 observable AFDC-arm states.

    Returns a dict keyed by the KT-style names p00_c, p10_c, p20_c, p01_c, p21_c
    so it can be drop-in substituted for `table4_p` in functions that consume
    those keys.  Note:
        p00_c  ↔  P^A(S_0N)  -- AFDC, no work, no welfare
        p10_c  ↔  P^A(S_1N)  -- AFDC, low work, no welfare
        p20_c  ↔  P^A(S_2N)  -- AFDC, high work, no welfare
        p01_c  ↔  P^A(S_0P)  -- AFDC, no work, on welfare
        p21_c  ↔  P^A(S_2P)  -- AFDC, high work, on welfare (proxy for 2u)
    """
    return {
        'p00_c': ipw_source_marginal(S_0N, D, state_obs, pi),
        'p10_c': ipw_source_marginal(S_1N, D, state_obs, pi),
        'p20_c': ipw_source_marginal(S_2N, D, state_obs, pi),
        'p01_c': ipw_source_marginal(S_0P, D, state_obs, pi),
        'p21_c': ipw_source_marginal(S_2P, D, state_obs, pi),
    }


# =============================================================================
# KT COMPOSITE Q VECTORS  (same positions as CLP_final_OLS_variants.py;
# only the normalization denominator switches to IPW marginals)
# =============================================================================
# CLP β-ordering:
#   [β(0n,1r), β(0r,0n), β(2n,1r), β(0r,2n), β(0r,1r),
#    β(0r,1n), β(1n,1r), β(0r,2u), β(2u,1r)]
#
# q vectors (matching CLP_final_OLS_variants.py and KT's Bounds.m):
#   TUW    : 1's at {0, 3, 4, 5, 7} / (P^A(0n) + P^A(0p))    ← KT-faithful
#            (includes β(0r,1r) at position 4)
#   TUWelf : 1's at {0, 2, 6} / (P^A(0n) + P^A(1n) + P^A(2n))
#   Exit   : 1's at {1, 3, 5} / P^A(0p)
# =============================================================================
def composite_q_vectors_kt(pa):
    """Build the three KT composite q-vectors using IPW source marginals.

    Parameter `pa` is a dict with keys p00_c, p10_c, p20_c, p01_c, p21_c
    (same names as KT's table4_p dict, but values are sample-IPW estimates
    from `all_source_marginals(D, state_obs, pi)`).
    """
    p00_c = pa['p00_c']; p01_c = pa['p01_c']
    p10_c = pa['p10_c']; p20_c = pa['p20_c']

    n_beta = 9
    q_tuw    = np.zeros(n_beta)
    q_tuwelf = np.zeros(n_beta)
    q_exit   = np.zeros(n_beta)

    # TUW (Take-Up Work): positions {0, 3, 4, 5, 7}.  Includes β(0r,1r).
    for j in (0, 3, 4, 5, 7):
        q_tuw[j] = 1.0
    q_tuw /= (p00_c + p01_c)

    # TUWelf (Take-Up Welfare): positions {0, 2, 6}
    for j in (0, 2, 6):
        q_tuwelf[j] = 1.0
    q_tuwelf /= (p00_c + p10_c + p20_c)

    # Exit 0r (welfare-zero -> off welfare): positions {1, 3, 5}
    for j in (1, 3, 5):
        q_exit[j] = 1.0
    q_exit /= p01_c

    return {
        "TUW (Take-Up Work)":         q_tuw,
        "TUWelf (Take-Up Welfare)":   q_tuwelf,
        "Exit 0r (on-welfare zero earn -> off welfare)": q_exit,
    }


# =============================================================================
# COMMON CLP PIPELINE  (mirrors CLP_final_OLS_variants.run_clp_pipeline
# exactly; `table4_p` argument replaced by `pa_ipw` dict)
# =============================================================================
def run_clp_pipeline(D, state_obs, b_hat, B_obs, person_id, label, pa_ipw):
    """
    Full CLP estimation given first-stage predictions.
    Uses IPW source marginals (pa_ipw) for both β→π conversion and
    composite q-vector normalization.
    """
    A_mat = build_A()

    # source_pop matched to BETA_NAMES order.  Uses IPW marginals from pa_ipw.
    # Mapping:
    #   index  cell         source state    pa_ipw key
    #     0    β(0n,1r)         0n           p00_c
    #     1    β(0r,0n)         0r           p01_c   (0r shares cells with 0p)
    #     2    β(2n,1r)         2n           p20_c
    #     3    β(0r,2n)         0r           p01_c
    #     4    β(0r,1r)         0r           p01_c
    #     5    β(0r,1n)         0r           p01_c
    #     6    β(1n,1r)         1n           p10_c
    #     7    β(0r,2u)         0r           p01_c
    #     8    β(2u,1r)         2u           p21_c   (2u shares cells with 2p)
    source_pop = np.array([
        pa_ipw['p00_c'], pa_ipw['p01_c'], pa_ipw['p20_c'],
        pa_ipw['p01_c'], pa_ipw['p01_c'], pa_ipw['p01_c'],
        pa_ipw['p10_c'], pa_ipw['p01_c'], pa_ipw['p21_c'],
    ])

    n_pi = 9
    results = {}

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    print(f"  Sample size: {len(D):,}    "
          f"unique persons: {len(np.unique(person_id)):,}")
    print(f"  IPW source marginals (this sample): "
          f"p00_c={pa_ipw['p00_c']:.4f}  p10_c={pa_ipw['p10_c']:.4f}  "
          f"p20_c={pa_ipw['p20_c']:.4f}  p01_c={pa_ipw['p01_c']:.4f}  "
          f"p21_c={pa_ipw['p21_c']:.4f}")
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

        width = ub_hat - lb_hat

        # β -> π conversion using IPW source marginal
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

    print(f"\n  -- pi-units (beta / P^a(source)) --")
    print(f"  {'pi param':<13}  {'LB_pi':>8}  {'UB_pi':>8}  {'width':>8}")
    for nm, res in results.items():
        pi_w = res['ub_pi'] - res['lb_pi']
        idx = BETA_NAMES.index(nm)
        print(f"  {PI_NAMES[idx]:<13}  {res['lb_pi']:8.4f}  "
              f"{res['ub_pi']:8.4f}  {pi_w:8.4f}")

    # ----------------------------------------------------------
    # KT composite bounds: TUW, TUWelf, Exit 0r — same as
    # CLP_final_OLS_variants, but q vectors use IPW marginals.
    # ----------------------------------------------------------
    print(f"\n  -- KT Table 5 composite bounds (pi-units, clipped to [0,1]) --")
    print(f"  {'composite':<48}  {'LB':>8}  {'UB':>8}  {'width':>8}    "
          f"{'95% CI(LB)':>16}    {'95% CI(UB)':>16}")
    print("  " + "-" * 110)
    composite_results = {}
    for cname, q_beta in composite_q_vectors_kt(pa_ipw).items():
        ub_hat, c_up, _, _, _ = clp_estimate(q_beta, b_hat, B_obs, A_mat)
        ci_ub_c = list(multiplier_bootstrap_ci(
            c_up, n_bs=N_BOOTSTRAP, person_id=person_id))
        neg_lb_c, c_dn, _, _, _ = clp_estimate(-q_beta, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb_c
        ci_lb_c = [-x for x in
                   multiplier_bootstrap_ci(
                       c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]
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

    results['_composites'] = composite_results
    return results


# =============================================================================
# SPEC RUNNERS  (same as CLP_final_OLS_variants.py;
# `table4_p` argument renamed `pa_ipw`)
# =============================================================================
def run_spec1_full_panel(D, state_obs, df_incl, person_id, pscorewt, pa_ipw):
    """Spec 1: no sample splitting, full panel."""
    print("\n\n" + "#"*80)
    print("# SPEC 1: No sample splitting -- full panel (woman x quarter)")
    print("#"*80)

    avail = [v for v in COV_VARS if v in df_incl.columns]
    X_raw = df_incl[avail].fillna(0).to_numpy(float)

    b_hat, B_obs = estimate_b0_no_split(D, state_obs, X_raw, pscorewt)
    return run_clp_pipeline(
        D, state_obs, b_hat, B_obs, person_id,
        f"Spec 1: full panel, OLS, no split ({len(avail)} covariates) "
        f"[IPW source marginals]",
        pa_ipw=pa_ipw,
    )


def run_spec2_cross_section(D, state_obs, df_incl, person_id, pscorewt,
                             pa_ipw_full, quarter=DEFAULT_CROSSSEC_QUARTER):
    """Spec 2: no sample splitting, cross-sectional slice.

    NOTE: pa_ipw is recomputed *on the slice* since the cross-sectional
    sub-sample has its own AFDC-arm marginals.
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

    # Recompute IPW source marginals on the cross-section slice.
    pi_x = derive_pi_from_pscorewt(D_x, pscorewt_x)
    pa_ipw_slice = all_source_marginals(D_x, state_x, pi_x)
    print(f"  IPW source marginals on slice: "
          f"p00_c={pa_ipw_slice['p00_c']:.4f}  p01_c={pa_ipw_slice['p01_c']:.4f}")

    b_hat, B_obs = estimate_b0_no_split(D_x, state_x, X_raw, pscorewt_x)
    return run_clp_pipeline(
        D_x, state_x, b_hat, B_obs, person_x,
        f"Spec 2: cross-section Q{quarter}, OLS, no split ({len(avail)} covariates) "
        f"[IPW source marginals on slice]",
        pa_ipw=pa_ipw_slice,
    )


def run_spec3_groupkfold(D, state_obs, df_incl, person_id, pscorewt,
                          pa_ipw, K_values=SPEC3_K_VALUES):
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
            f"Spec 3 (K={K}): GroupKFold by woman ID, OLS ({len(avail)} covariates) "
            f"[IPW source marginals]",
            pa_ipw=pa_ipw,
        )
    return results_by_K


def run_spec4_cov_subsets(D, state_obs, df_incl, person_id, pscorewt,
                           pa_ipw, subsets_to_run=None):
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
            f"Spec 4 ({subset_name}): no split, OLS, {len(avail)} covariates "
            f"[IPW source marginals]",
            pa_ipw=pa_ipw,
        )
    return results_by_subset


# =============================================================================
# SUMMARY TABLE  (identical to CLP_final_OLS_variants.py)
# =============================================================================
def print_summary(all_results):
    """Side-by-side comparison of bound widths across all specs."""
    print("\n\n" + "="*120)
    print("SUMMARY: beta interval widths [UB - LB] across specifications "
          "[IPW source marginals]")
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
    print("SUMMARY: pi interval widths [UB_pi - LB_pi] across specifications "
          "[IPW source marginals]")
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

    print("\n" + "="*120)
    print("SUMMARY: KT Table 5 composite bounds across specifications "
          "(pi-units, clipped to [0,1]) [IPW source marginals]")
    print("="*120)
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
    print("clp_ols_new.py -- CLP_final_OLS_variants with IPW source marginals")
    print("="*80)
    print(f"  N_BOOTSTRAP = {N_BOOTSTRAP}")
    print(f"  RANDOM_SEED = {RANDOM_SEED}")
    print(f"  Spec 2 cross-section quarter = Q{DEFAULT_CROSSSEC_QUARTER}")
    print(f"  Spec 3 K values = {SPEC3_K_VALUES}")
    print(f"  Spec 4 covariate subsets = {list(COV_SUBSETS.keys())}")
    print(f"  Source marginal P^a(s): SAMPLE IPW (not KT Table 4)")

    np.random.seed(RANDOM_SEED)

    print("\n" + "-"*80)
    print("Loading JF data (this is the ONLY data load; reused across all specs)")
    print("-"*80)
    D, state_obs, df_incl, person_id, pscorewt = prepare_jf_data()

    # Compute IPW source marginals ONCE on the full sample.  Spec 2
    # recomputes on its cross-section slice; specs 1/3/4 use these.
    pi_full = derive_pi_from_pscorewt(D, pscorewt)
    pa_ipw_full = all_source_marginals(D, state_obs, pi_full)

    print(f"\n  IPW source marginals (full sample, N={len(D):,}):")
    for k in ['p00_c', 'p10_c', 'p20_c', 'p01_c', 'p21_c']:
        print(f"    {k} = {pa_ipw_full[k]:.6f}")

    # For comparison, also load KT Table 4 to show the difference.
    try:
        table4_p, _ = load_table4_mat()
        print(f"\n  KT Table 4 reference (for comparison only, NOT USED):")
        for k in ['p00_c', 'p10_c', 'p20_c', 'p01_c', 'p21_c']:
            diff = pa_ipw_full[k] - table4_p[k]
            print(f"    {k}: KT={table4_p[k]:.6f}  IPW={pa_ipw_full[k]:.6f}  "
                  f"diff={diff:+.6f}")
    except Exception as e:
        print(f"  (KT Table 4 load skipped: {e})")

    all_results = {}

    # --- Spec 1 ---
    np.random.seed(RANDOM_SEED)
    all_results["Spec1_full_panel"] = run_spec1_full_panel(
        D, state_obs, df_incl, person_id, pscorewt, pa_ipw_full
    )

    # --- Spec 2 ---
    np.random.seed(RANDOM_SEED)
    all_results[f"Spec2_Q{DEFAULT_CROSSSEC_QUARTER}_cross"] = run_spec2_cross_section(
        D, state_obs, df_incl, person_id, pscorewt, pa_ipw_full,
        quarter=DEFAULT_CROSSSEC_QUARTER,
    )

    # --- Spec 3 ---
    np.random.seed(RANDOM_SEED)
    spec3_res = run_spec3_groupkfold(
        D, state_obs, df_incl, person_id, pscorewt, pa_ipw_full,
        K_values=SPEC3_K_VALUES,
    )
    for K, res in spec3_res.items():
        all_results[f"Spec3_K{K}"] = res

    # --- Spec 4 ---
    np.random.seed(RANDOM_SEED)
    spec4_res = run_spec4_cov_subsets(
        D, state_obs, df_incl, person_id, pscorewt, pa_ipw_full,
        subsets_to_run=list(COV_SUBSETS.keys()),
    )
    for sname, res in spec4_res.items():
        all_results[f"Spec4_{sname}"] = res

    print_summary(all_results)


if __name__ == "__main__":
    main()
