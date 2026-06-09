"""
CLP_lasso_appendix_new.py
=========================
Self-contained (GitHub-ready) version of `clp_lasso_appendix_v2.py`.

WHY THIS FILE EXISTS
--------------------
`clp_lasso_appendix_v2.py` is only a thin wrapper: it does
`import clp_lasso_appendix as legacy`, monkey-patches the corrected
composite q-vectors and output paths onto that legacy module, then calls
`legacy.main()`.  That is convenient locally but awkward to publish,
because reading it requires also reading `clp_lasso_appendix.py` AND the
LP machinery in `CLP_final.py`.

This file inlines ALL of that machinery so it stands on its own:
  * LP / dual-vertex code (build_A, enumerate_dual_vertices,
    clp_estimate, multiplier_bootstrap_ci)            -- was CLP_final.py
  * data prep / first stage / diagnostics              -- was clp_lasso_appendix.py
  * the CORRECTED KT composite q-vectors               -- was clp_lasso_appendix_v2.py

The only remaining external dependency is `CLP_granular_final_group`,
which loads the actual KT `.dta` panel from disk (prepare_jf_data_granular,
load_table4_mat) and builds the engineered feature set.  Those are data-
loading utilities tied to the replication package's files, so they are
imported rather than inlined.

WHAT IS "CORRECTED" RELATIVE TO clp_lasso_appendix.py (v1)
---------------------------------------------------------
The KT composite q-vectors for TUW (Take-Up Work) and Exit-0r were missing
cells in v1.  This file uses the KT-Bounds.m-faithful positions:

    TUW    v1 = [1,0,0,1,0,1,0,1,0]   positions {0,3,5,7}
           v2 = [1,0,0,1,1,1,0,1,0]   positions {0,3,4,5,7}
                          ^--- beta(0r,1r) restored (Bounds.m line 533)

    Exit   v1 = [0,1,0,0,0,0,0,0,0]   position  {1}
           v2 = [0,1,0,1,0,1,0,0,0]   positions {1,3,5}
                       ^---^--- beta(0r,2n), beta(0r,1n) restored
                                (Bounds.m line 622, "on welfare -> off welfare")

    TUWelf v1 = [1,0,1,0,0,0,1,0,0]   positions {0,2,6}   (already correct)

Consequences (visible in the printed composite table):
  * TUW becomes point-identified for every config (UB == LB ~= 0.160),
    because the corrected q_tuw lies in row(A) so T_q has a single vertex.
  * Exit-0r reports the correct three-transition sum rather than a single
    transition.

COARSE 5x9 KT SPEC
------------------
   estimators        : {LASSO, Ridge}
   splitting regimes : {plain KFold cross-fit ("kfold"),
                        GroupKFold cross-fit by person ("groupkfold")}
   covariate sets    : {"base" (28 KT covariates),
                        "econ" (~255 engineered features)}
   Total             : 2 x 2 x 2 = 8 configurations

LP route             : vertex enumeration via enumerate_dual_vertices.
                       No box, no fallback -- vertex enumeration on the
                       coarse polytope returns the exact LP optimum under
                       the paper's standing regularity assumption.

Diagnostics block:
   D1  Per-direction vertex enumeration: |V_q|, ||.||_inf range
   D2  Binding-vertex distribution: counts, shares, entropy
   D3  Recession-cone test: which observations are recession-unbounded
   D4  Margin to second-best vertex (knife-edge fraction)

Output:
   CLP_lasso_appendix_new_results.csv      per-config x per-target bound table
   CLP_lasso_appendix_new_diagnostics.csv  per-config x per-direction x per-vertex
                                            usage / margin / recession diagnostics
"""

import os
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# The only external dependency: data loaders for the KT replication package.
# These read the actual `.dta` panel and build the engineered feature set,
# so they remain imports rather than being inlined.
# ---------------------------------------------------------------------------
from CLP_granular_final_group import (
    prepare_jf_data_granular, load_table4_mat, COV_VARS,
    engineer_features_econ, K_FOLDS, RANDOM_SEED,
)


# =============================================================================
# CONFIG
# =============================================================================
ESTIMATORS    = ["LASSO", "Ridge"]
SPLIT_MODES   = ["kfold", "groupkfold"]
FEATURE_SETS  = ["base", "econ"]
N_BOOTSTRAP   = 200
ALPHA         = 0.05
EPS_RECESS    = 1e-6
KNIFE_TOL     = 1e-3

OUT_DIR        = os.path.dirname(os.path.abspath(__file__))
OUT_CSV        = os.path.join(OUT_DIR, "CLP_lasso_appendix_new_results.csv")
OUT_DIAG_CSV   = os.path.join(OUT_DIR, "CLP_lasso_appendix_new_diagnostics.csv")

# Observable state codes (0=0n, 1=1n, 2=2n, 3=0p, 4=1p, 5=2p).
# '0r' shares cells with '0p'; '2u' shares cells with '2p'.
S_0N, S_1N, S_2N, S_0P, S_1P, S_2P = 0, 1, 2, 3, 4, 5

BETA_LABELS = [
    "beta(0n,1r)", "beta(0r,0n)", "beta(2n,1r)", "beta(0r,2n)",
    "beta(0r,1r)", "beta(0r,1n)", "beta(1n,1r)", "beta(0r,2u)",
    "beta(2u,1r)",
]


# =============================================================================
# LP MACHINERY  (inlined from CLP_final.py)
# =============================================================================
def build_A():
    """The coarse 5x9 KT structural matrix (pure +-1, base 3-bin model).

    Rows are the 5 observable AFDC-arm source states {0n, 1n, 2n, 0p, 2p};
    columns are the 9 transition cells in CLP beta-ordering:
      [beta(0n,1r), beta(0r,0n), beta(2n,1r), beta(0r,2n), beta(0r,1r),
       beta(0r,1n), beta(1n,1r), beta(0r,2u), beta(2u,1r)]
    """
    #               b0n1r b0r0n b2n1r b0r2n b0r1r b0r1n b1n1r b0r2u b2u1r
    return np.array([
        [ -1,    1,    0,    0,    0,    0,    0,    0,    0],   # 0n
        [  0,    0,    0,    0,    0,    1,   -1,    0,    0],   # 1n
        [  0,    0,   -1,    1,    0,    0,    0,    0,    0],   # 2n
        [  0,   -1,    0,   -1,   -1,   -1,    0,   -1,    0],   # 0p
        [  0,    0,    0,    0,    0,    0,    0,    1,   -1],   # 2p
    ], dtype=float)


def enumerate_dual_vertices(A, q):
    """Enumerate all C(9,5)=126 candidate vertices of T_q = {nu: A'nu >= q}.

    For each basis subset J of 5 columns, solve A[:,J]' nu = q[J] and keep nu
    if it is dual-feasible (A'nu >= q).  Exact for the coarse polytope under
    CLP regularity.
    """
    k, d = A.shape
    AT   = A.T
    q    = np.asarray(q, dtype=float)
    vertices, seen = [], set()
    for J in combinations(range(d), k):
        M = A[:, list(J)].T
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        try:
            nu = np.linalg.solve(M, q[list(J)])
        except np.linalg.LinAlgError:
            continue
        if np.all(AT @ nu >= q - 1e-8):
            key = tuple(np.round(nu, 8))
            if key not in seen:
                seen.add(key)
                vertices.append(nu)
    return vertices


def clp_estimate(q, b_hat, B_obs, A_mat):
    """CLP plug-in estimate of sigma(q) = E[ min_{nu in T_q} nu . b0(X) ].

    For each observation i, pick the dual vertex minimising nu . b_hat[i],
    then form the moment contribution nu* . B_obs[i].  Returns the sample
    mean of those contributions plus the per-observation arrays.
    """
    vertices = enumerate_dual_vertices(A_mat, q)
    n = len(b_hat)
    contribs  = np.zeros(n)
    nu_sel    = np.zeros((n, A_mat.shape[0]))
    vert_idx  = np.zeros(n, dtype=int)
    for i in range(n):
        vals = np.array([v @ b_hat[i] for v in vertices])
        idx  = int(np.argmin(vals))
        nu_sel[i]   = vertices[idx]
        vert_idx[i] = idx
        contribs[i] = nu_sel[i] @ B_obs[i]
    return contribs.mean(), contribs, nu_sel, vertices, vert_idx


def multiplier_bootstrap_ci(contribs, n_bs=N_BOOTSTRAP, alpha=0.05, person_id=None):
    """Studentized multiplier (Exp(1)-weighted) bootstrap CI.

    When `person_id` is supplied, weights are drawn once per person and
    broadcast to all of that person's panel rows (person-clustered).  The
    interval inverts the studentized pivot sqrt(n)(sigma* - sigma_hat).
    """
    n, sigma_hat = len(contribs), contribs.mean()
    if person_id is not None:
        unique_ids = np.unique(person_id)
        id_to_idx  = {pid: np.where(person_id == pid)[0] for pid in unique_ids}
        bs = np.zeros(n_bs)
        for b in range(n_bs):
            wts   = np.ones(n)
            exp_w = np.random.exponential(1, size=len(unique_ids))
            for c, pid in enumerate(unique_ids):
                wts[id_to_idx[pid]] = exp_w[c]
            bs[b] = (contribs * wts).mean()
    else:
        bs = np.array([
            (contribs * np.random.exponential(1, n)).mean()
            for _ in range(n_bs)
        ])
    bs_stat = np.sqrt(n) * (bs - sigma_hat)
    q_lo    = np.quantile(bs_stat, alpha / 2)
    q_hi    = np.quantile(bs_stat, 1 - alpha / 2)
    return sigma_hat - q_hi / np.sqrt(n), sigma_hat - q_lo / np.sqrt(n)


# =============================================================================
# DATA PREP
# =============================================================================
def state_code(ebin, partic):
    if partic == 0:
        if ebin == 0:           return S_0N
        if 1 <= ebin <= 5:      return S_1N
        return S_2N
    else:
        if ebin == 0:           return S_0P
        if 6 <= ebin <= 8:      return S_2P
        return -1


def prepare_data():
    data = prepare_jf_data_granular()
    p_kt, _ = load_table4_mat()
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data
    N = len(D)

    state_obs = np.array([state_code(int(ebin_arr[i]), int(partic_arr[i]))
                          for i in range(N)], dtype=int)

    X_base = df_incl[[v for v in COV_VARS if v in df_incl.columns]] \
                 .fillna(0).to_numpy(float)

    X_b, X_e, _, _ = engineer_features_econ(df_incl, cov_vars=COV_VARS)
    X_econ = np.hstack([X_b, X_e]) if X_e.size else X_b

    pi = np.where(D == 1, 1.0 / pscorewt, 1.0 - 1.0 / pscorewt)
    pi = np.clip(pi, 0.02, 0.98)

    return D, state_obs, X_base, X_econ, person_id, pscorewt, pi, df_incl, p_kt


def compute_B(D, state_obs, pscorewt):
    """B[i,r] = (2 D_i - 1) * pscorewt_i * 1{state_i == r} for the 5 observable
    states r in {0n, 1n, 2n, 0p, 2p}."""
    w = (2.0 * D - 1.0) * pscorewt
    N = len(D)
    B = np.zeros((N, 5))
    for j, s in enumerate([S_0N, S_1N, S_2N, S_0P, S_2P]):
        B[:, j] = (state_obs == s).astype(float) * w
    return B


# =============================================================================
# FIRST STAGE — LASSO / Ridge, cross-fit
# =============================================================================
def _make_model(estimator, seed):
    if estimator == "LASSO":
        return LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, eps=1e-4,
                       max_iter=4000, random_state=seed, n_jobs=1)
    if estimator == "Ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 40))
    raise ValueError(f"Unknown estimator: {estimator}")


def estimate_b0(B_obs, X_raw, D, person_id, estimator, mode,
                 K=K_FOLDS, seed=RANDOM_SEED, verbose=True):
    """First-stage cross-fit regression of B on X.

    Two regimes:
      mode == "kfold"      : plain K-fold cross-fit (no person grouping).
                             Person-quarter rows from the same individual
                             may end up split across train and test folds.
      mode == "groupkfold" : K-fold cross-fit with GroupKFold(n_splits=K)
                             grouped by person_id, so all rows belonging to
                             a given individual stay in the same fold.  The
                             more conservative regime when person-level
                             dependence is suspected.

    IMPORTANT: we deliberately DO NOT include the treatment indicator W in
    the design matrix.  The CLP framework specifies b0(X) = E[B | X];
    including W would let the regression predict the sign of B almost exactly
    (since B's sign is determined by W) and produce pathological in-sample
    b_hat ~= B that breaks the vertex-enumeration LP.
    """
    N, k = B_obs.shape
    b_hat = np.zeros((N, k))

    if mode == "kfold":
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        folds = list(kf.split(np.arange(N)))
        if verbose:
            print(f"    [{estimator}/kfold] plain KFold K={K} (shuffle=True) "
                  f"on {N:,} x {X_raw.shape[1]} features (X only)")
    elif mode == "groupkfold":
        gkf = GroupKFold(n_splits=K)
        folds = list(gkf.split(np.arange(N), groups=person_id))
        if verbose:
            print(f"    [{estimator}/groupkfold] GroupKFold K={K} by person "
                  f"on {N:,} x {X_raw.shape[1]} features (X only)")
    else:
        raise ValueError(f"Unknown mode: {mode} "
                          "(expected 'kfold' or 'groupkfold')")

    for tr, te in folds:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_raw[tr])
        X_te = sc.transform(X_raw[te])
        for j in range(k):
            y = B_obs[tr, j]
            if y.std() < 1e-10:
                b_hat[te, j] = y.mean()
                continue
            m = _make_model(estimator, seed)
            m.fit(X_tr, y)
            b_hat[te, j] = m.predict(X_te)
    return b_hat


# =============================================================================
# DIAGNOSTICS HELPERS
# =============================================================================
def recession_unbounded(b_hat_i, A_mat):
    k = A_mat.shape[0]
    res = linprog(c=b_hat_i,
                  A_ub=-A_mat.T, b_ub=np.zeros(A_mat.shape[1]),
                  bounds=[(-1.0, 1.0)] * k,
                  method='highs', options={'disp': False})
    if res.status != 0:
        return False, float('nan')
    slope = float(res.fun)
    return (slope < -EPS_RECESS), slope


def per_direction_diagnostic(q, b_hat, B, A_mat, dir_label, verbose=False):
    """Per-direction vertex usage, recession test, knife-edge fraction."""
    V = enumerate_dual_vertices(A_mat, q)
    if len(V) == 0:
        return None
    V = np.asarray(V)
    n_vert = V.shape[0]
    ninfs = np.linalg.norm(V, ord=np.inf, axis=1)
    n = len(b_hat)
    vals = b_hat @ V.T
    sorted_idx = np.argsort(vals, axis=1)
    bind_idx = sorted_idx[:, 0]
    second_idx = sorted_idx[:, 1] if n_vert > 1 else sorted_idx[:, 0]
    margins = vals[np.arange(n), second_idx] - vals[np.arange(n), bind_idx]

    recess_flag = np.zeros(n, bool)
    for i in range(n):
        unb, _ = recession_unbounded(b_hat[i], A_mat)
        recess_flag[i] = unb

    counts = np.bincount(bind_idx, minlength=n_vert)
    shares = counts / max(1, n)
    H = -np.sum([p * np.log(max(p, 1e-12)) for p in shares if p > 0])
    knife_share = float(np.mean(margins < KNIFE_TOL))
    n_unb = int(recess_flag.sum())

    if verbose:
        print(f"      direction {dir_label}: #vertices={n_vert} entropy={H:.3f} "
              f"modal_share={shares.max():.3f} n_unb={n_unb}/{n} "
              f"knife={100*knife_share:.1f}%")

    return dict(
        V=V, ninfs=ninfs, bind_idx=bind_idx, counts=counts, shares=shares,
        entropy=H, knife_share=knife_share,
        n_unbounded=n_unb, n_vert=n_vert,
        recess_flag=recess_flag, margins=margins,
    )


# =============================================================================
# IPW SOURCE MARGINALS  (point-identified)  →  for π = β / P^A(s) conversion
# =============================================================================
# Map each col j to the integer state code of its source under our encoding
# (0=0n, 1=1n, 2=2n, 3=0p, 4=2p).  '0r' shares cells with '0p', '2u' with '2p'.
COL_SOURCE_STATE = [
    S_0N,    # 0: (0n, 1r)
    S_0P,    # 1: (0r, 0n)    src = 0r = 0p (cells)
    S_2N,    # 2: (2n, 1r)
    S_0P,    # 3: (0r, 2n)
    S_0P,    # 4: (0r, 1r)
    S_0P,    # 5: (0r, 1n)
    S_1N,    # 6: (1n, 1r)
    S_0P,    # 7: (0r, 2u)
    S_2P,    # 8: (2u, 1r)    src = 2u = 2p (cells)
]

# Source labels for each col (used as `source_label` in the CSV)
COL_SOURCE_LABEL = ["0n", "0r", "2n", "0r", "0r", "0r", "1n", "0r", "2u"]


def ipw_source_marginal(state_code_value, D, state_obs, pi):
    """P̂^A(Y = state_code_value)  estimated via Horvitz–Thompson IPW on the
    AFDC arm:   (1/N) Σ_i (1 - W_i)/(1 - π_i) · 1{state_i = state_code_value}.
    """
    contribs = np.where(D == 0,
                        (state_obs == state_code_value).astype(float) / (1.0 - pi),
                        0.0)
    return float(contribs.mean())


def all_source_marginals(D, state_obs, pi):
    """Compute P̂^A for the 5 observable states."""
    return {s: ipw_source_marginal(s, D, state_obs, pi)
            for s in [S_0N, S_1N, S_2N, S_0P, S_2P]}


# =============================================================================
# KT COMPOSITE Q VECTORS  (CORRECTED — KT Bounds.m faithful)
# =============================================================================
def kt_composite_q_vectors(p_kt):
    """Build the three KT composite q-vectors in CLP β-ordering, with the TUW
    and Exit-0r positions matching KT Bounds.m.

    CLP β-ordering:
      [β(0n,1r), β(0r,0n), β(2n,1r), β(0r,2n), β(0r,1r),
       β(0r,1n), β(1n,1r), β(0r,2u), β(2u,1r)]

    q is pre-divided by the relevant source marginal so that σ̂(q) comes out
    directly in π-units.
    """
    p00_c = p_kt['p00_c']; p01_c = p_kt['p01_c']
    p10_c = p_kt['p10_c']; p20_c = p_kt['p20_c']

    # TUW (Take-Up Work) -- non-zero at positions {0, 3, 4, 5, 7}
    #   = β(0n,1r) + β(0r,2n) + β(0r,1r) + β(0r,1n) + β(0r,2u)
    q_tuw = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0], dtype=float)
    q_tuw /= max(1e-12, p00_c + p01_c)

    # TUWelf (Take-Up Welfare) -- non-zero at {0, 2, 6}
    #   = β(0n,1r) + β(2n,1r) + β(1n,1r)
    q_tuwelf = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=float)
    q_tuwelf /= max(1e-12, p00_c + p10_c + p20_c)

    # Exit 0r (on welfare 0r → off welfare anywhere) -- {1, 3, 5}
    #   = β(0r,0n) + β(0r,2n) + β(0r,1n)
    q_exit = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=float)
    q_exit /= max(1e-12, p01_c)

    return {
        "TUW (Take-Up Work)":                    q_tuw,
        "TUWelf (Take-Up Welfare)":              q_tuwelf,
        "Exit 0r (welfare-zero -> off welfare)": q_exit,
    }


# =============================================================================
# ONE CONFIG: ESTIMATE → BOUNDS → DIAGNOSTICS
# =============================================================================
def run_config(estimator, split_mode, feature_set,
                D, state_obs, X_base, X_econ, person_id, pscorewt, pi, p_kt):
    print("\n" + "=" * 96)
    print(f"  CONFIG: {estimator} / split={split_mode} / features={feature_set}")
    print("=" * 96)

    X = X_base if feature_set == "base" else X_econ
    B_obs = compute_B(D, state_obs, pscorewt)
    b_hat = estimate_b0(B_obs, X, D, person_id,
                         estimator=estimator, mode=split_mode, verbose=True)

    A_mat = build_A()
    n_total = len(D)

    # IPW source marginals  (used to convert β → π)
    pa_by_state = all_source_marginals(D, state_obs, pi)
    print(f"    IPW source marginals (control arm):")
    for s_lbl, s_code in [("0n", S_0N), ("1n", S_1N), ("2n", S_2N),
                          ("0p/0r", S_0P), ("2p/2u", S_2P)]:
        print(f"      P̂^A({s_lbl:<6}) = {pa_by_state[s_code]:.4f}")

    def clip01(x):
        return float(max(0., min(1., x))) if not np.isnan(x) else float('nan')

    rows = []
    # ----- Individual β bounds (and π conversion) -----
    for j in range(9):
        q_up = np.zeros(9); q_up[j] = +1.0
        q_dn = np.zeros(9); q_dn[j] = -1.0
        ub_hat, c_up, _, _, _ = clp_estimate(q_up, b_hat, B_obs, A_mat)
        neg_lb, c_dn, _, _, _ = clp_estimate(q_dn, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb
        ci_ub = multiplier_bootstrap_ci(c_up, n_bs=N_BOOTSTRAP,
                                         alpha=ALPHA, person_id=person_id)
        ci_lb_raw = multiplier_bootstrap_ci(c_dn, n_bs=N_BOOTSTRAP,
                                             alpha=ALPHA, person_id=person_id)
        ci_lb = (-ci_lb_raw[1], -ci_lb_raw[0])

        # π conversion : π(s, d) = β(s, d) / P^A(s)
        src_state = COL_SOURCE_STATE[j]
        src_lbl   = COL_SOURCE_LABEL[j]
        p_A_s     = pa_by_state[src_state]
        if p_A_s > 1e-12:
            lb_pi_raw = lb_hat / p_A_s
            ub_pi_raw = ub_hat / p_A_s
            ci_lb_pi = (ci_lb[0] / p_A_s, ci_lb[1] / p_A_s)
            ci_ub_pi = (ci_ub[0] / p_A_s, ci_ub[1] / p_A_s)
        else:
            lb_pi_raw = ub_pi_raw = float('nan')
            ci_lb_pi = ci_ub_pi = (float('nan'), float('nan'))

        rows.append(dict(
            estimator=estimator, split_mode=split_mode, feature_set=feature_set,
            kind="beta", target=BETA_LABELS[j],
            source_label=src_lbl, source_pa=p_A_s,
            lb_beta=lb_hat, ub_beta=ub_hat, width_beta=ub_hat - lb_hat,
            lb_pi_raw=lb_pi_raw, ub_pi_raw=ub_pi_raw,
            lb_pi=clip01(lb_pi_raw), ub_pi=clip01(ub_pi_raw),
            width_pi=clip01(ub_pi_raw) - clip01(lb_pi_raw),
            ci_lb_lo=ci_lb[0], ci_lb_hi=ci_lb[1],
            ci_ub_lo=ci_ub[0], ci_ub_hi=ci_ub[1],
            ci_lb_pi_lo=clip01(ci_lb_pi[0]), ci_lb_pi_hi=clip01(ci_lb_pi[1]),
            ci_ub_pi_lo=clip01(ci_ub_pi[0]), ci_ub_pi_hi=clip01(ci_ub_pi[1]),
        ))

    # ----- KT composites (q is pre-divided by source marginal in q-build,
    #       so σ̂(q) is ALREADY in π-units; we just clip to [0, 1]) -----
    for cname, q_beta in kt_composite_q_vectors(p_kt).items():
        ub_hat, c_up, _, _, _ = clp_estimate(q_beta, b_hat, B_obs, A_mat)
        neg_lb_c, c_dn, _, _, _ = clp_estimate(-q_beta, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb_c
        ci_ub = multiplier_bootstrap_ci(c_up, n_bs=N_BOOTSTRAP,
                                         alpha=ALPHA, person_id=person_id)
        ci_lb_raw = multiplier_bootstrap_ci(c_dn, n_bs=N_BOOTSTRAP,
                                             alpha=ALPHA, person_id=person_id)
        ci_lb = (-ci_lb_raw[1], -ci_lb_raw[0])
        rows.append(dict(
            estimator=estimator, split_mode=split_mode, feature_set=feature_set,
            kind="composite", target=cname,
            source_label="composite", source_pa=float('nan'),
            lb_beta=lb_hat, ub_beta=ub_hat, width_beta=ub_hat - lb_hat,
            lb_pi_raw=lb_hat, ub_pi_raw=ub_hat,
            lb_pi=clip01(lb_hat), ub_pi=clip01(ub_hat),
            width_pi=clip01(ub_hat) - clip01(lb_hat),
            ci_lb_lo=ci_lb[0], ci_lb_hi=ci_lb[1],
            ci_ub_lo=ci_ub[0], ci_ub_hi=ci_ub[1],
            ci_lb_pi_lo=clip01(ci_lb[0]), ci_lb_pi_hi=clip01(ci_lb[1]),
            ci_ub_pi_lo=clip01(ci_ub[0]), ci_ub_pi_hi=clip01(ci_ub[1]),
        ))

    # ---- Diagnostics (only on β(2n,1r) and β(1n,1r)) ----
    diag_rows = []
    for col_idx, lbl in [(2, "beta(2n,1r)"), (6, "beta(1n,1r)")]:
        for sign, dir_lbl in [(+1.0, "UB"), (-1.0, "LB")]:
            q = np.zeros(9); q[col_idx] = sign
            d = per_direction_diagnostic(q, b_hat, B_obs, A_mat,
                                          dir_label=f"{lbl} {dir_lbl}",
                                          verbose=True)
            if d is None:
                continue
            for k in range(d['n_vert']):
                diag_rows.append(dict(
                    estimator=estimator, split_mode=split_mode,
                    feature_set=feature_set,
                    target=lbl, direction=dir_lbl, v_idx=k,
                    nu_value=";".join(f"{x:+.2f}" for x in d['V'][k]),
                    nu_inf_norm=float(d['ninfs'][k]),
                    bind_count=int(d['counts'][k]),
                    bind_share=float(d['shares'][k]),
                    sum_contrib=float(
                        (d['V'][k] * B_obs[d['bind_idx'] == k]).sum()),
                    binding_entropy=float(d['entropy']),
                    knife_share=float(d['knife_share']),
                    n_unbounded=int(d['n_unbounded']),
                    n_total=n_total,
                ))
    return rows, diag_rows


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 96)
    print("  CLP_lasso_appendix_new.py  --  self-contained coarse 5x9 KT,")
    print("                                 LASSO + Ridge, corrected q-vectors")
    print(f"                                 splits={SPLIT_MODES}, features={FEATURE_SETS}")
    print("  LP route: VERTEX ENUMERATION   (no box, no fallback)")
    print("  q-vectors: TUW {0,3,4,5,7}, Exit-0r {1,3,5}, TUWelf {0,2,6}")
    print("=" * 96)

    print("\n[STAGE 0] Load data")
    D, state_obs, X_base, X_econ, person_id, pscorewt, pi, df_incl, p_kt = \
        prepare_data()
    print(f"           N = {len(D):,}")
    print(f"           X_base = {X_base.shape}   X_econ = {X_econ.shape}")

    all_rows = []
    all_diag = []
    for estimator in ESTIMATORS:
        for split_mode in SPLIT_MODES:
            for feature_set in FEATURE_SETS:
                rows, diag = run_config(
                    estimator, split_mode, feature_set,
                    D, state_obs, X_base, X_econ, person_id, pscorewt, pi, p_kt,
                )
                all_rows.extend(rows)
                all_diag.extend(diag)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Wrote {OUT_CSV}  ({len(df)} rows)")

    df_diag = pd.DataFrame(all_diag)
    df_diag.to_csv(OUT_DIAG_CSV, index=False)
    print(f"  Wrote {OUT_DIAG_CSV}  ({len(df_diag)} rows)")

    # Cross-config summary for the two flagship transitions (β + π side-by-side)
    print("\n[STAGE 5] Cross-config summary for β(2n,1r) and β(1n,1r)")
    for tgt in ["beta(2n,1r)", "beta(1n,1r)"]:
        print(f"\n  {tgt}")
        sub = df[df['target'] == tgt]
        print(f"  {'estimator':<8} {'split':<11} {'features':<6}  "
              f"{'lb_β':>9} {'ub_β':>9} {'wβ':>9}    "
              f"{'lb_π':>9} {'ub_π':>9} {'wπ':>9}    "
              f"{'P^A(src)':>9}")
        for _, r in sub.iterrows():
            print(f"  {r['estimator']:<8} {r['split_mode']:<11} "
                  f"{r['feature_set']:<6}  "
                  f"{r['lb_beta']:>+9.4f} {r['ub_beta']:>+9.4f} "
                  f"{r['width_beta']:>+9.4f}    "
                  f"{r['lb_pi']:>+9.4f} {r['ub_pi']:>+9.4f} "
                  f"{r['width_pi']:>+9.4f}    "
                  f"{r['source_pa']:>9.4f}")

    print("\n[STAGE 6] Cross-config summary for KT composites (π-units)")
    for tgt in ["TUW (Take-Up Work)",
                "TUWelf (Take-Up Welfare)",
                "Exit 0r (welfare-zero -> off welfare)"]:
        print(f"\n  {tgt}")
        sub = df[df['target'] == tgt]
        print(f"  {'estimator':<8} {'split':<11} {'features':<6}  "
              f"{'lb_π':>9} {'ub_π':>9} {'wπ':>9}")
        for _, r in sub.iterrows():
            print(f"  {r['estimator']:<8} {r['split_mode']:<11} "
                  f"{r['feature_set']:<6}  "
                  f"{r['lb_pi']:>+9.4f} {r['ub_pi']:>+9.4f} "
                  f"{r['width_pi']:>+9.4f}")

    print(f"\nTotal runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
