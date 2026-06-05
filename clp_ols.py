"""
clp_ols.py
==========
Coarse 5x9 KT spec, OLS first stage, FULL-SAMPLE (in-sample) only.

Counterpart of CLP_final_OLS_variants.py, restricted to a single
configuration: no cross-fitting.  Includes a diagnostics block that
inspects the vertex-enumeration LP per observation.

Pipeline:
  Data        : prepare_jf_data_granular()
  Spec        : coarse 5 x 9 KT
  First stage : OLS (LinearRegression) fit in-sample on (X, W) → b_hat
                 mode = 'full' (no GroupKFold)
  LP          : vertex enumeration via enumerate_dual_vertices  + per-i argmin
                  (no box, no fallback; vertex enumeration is the paper's
                   function-2 estimator and does not need either)

Diagnostics (Section D):
  D1  vertex set per direction (size, ||·||_∞, components)
  D2  binding-vertex distribution: counts, shares, entropy, modal share
  D3  recession-cone test per observation
  D4  Σ-contribution decomposition (unbounded vs ok)
  D5  margin to second-best vertex (knife-edge share)
  D6  contribution share per vertex

Output:
  clp_ols_results.csv         per-target bound table (individual β + KT composites)
  clp_ols_diagnostics.csv     per-(target, direction, vertex) diagnostic rows
"""

import os
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from CLP_final import (
    build_A, enumerate_dual_vertices, clp_estimate, multiplier_bootstrap_ci,
    S_0N, S_1N, S_2N, S_0P, S_2P,
)
from CLP_granular_final_group import (
    prepare_jf_data_granular, load_table4_mat, COV_VARS, RANDOM_SEED,
)


# =============================================================================
# CONFIG
# =============================================================================
N_BOOTSTRAP    = 200
ALPHA          = 0.05
EPS_RECESS     = 1e-6
EPS_NORM       = 1e-9
KNIFE_TOL      = 1e-3

OUT_DIR        = "/Users/gevorgkhandamiryan/Desktop/cursorclp/check/finalcheck"
OUT_CSV        = os.path.join(OUT_DIR, "clp_ols_results.csv")
OUT_DIAG_CSV   = os.path.join(OUT_DIR, "clp_ols_diagnostics.csv")

# 9-state coarse spec col labels (matches CLP_final.py)
BETA_LABELS = [
    "beta(0n,1r)", "beta(0r,0n)", "beta(2n,1r)", "beta(0r,2n)",
    "beta(0r,1r)", "beta(0r,1n)", "beta(1n,1r)", "beta(0r,2u)",
    "beta(2u,1r)",
]


# =============================================================================
# DATA PREP
# =============================================================================
def state_code(ebin, partic):
    """Encode (ebin, partic) → 5-state observable in the coarse spec.
       0=0n, 1=1n, 2=2n, 3=0p, 4=2p.  Reference state (1p/1r) is implicit."""
    if partic == 0:
        if ebin == 0:           return S_0N
        if 1 <= ebin <= 5:      return S_1N
        return S_2N
    else:
        if ebin == 0:           return S_0P
        if 6 <= ebin <= 8:      return S_2P
        return -1                # 1p reference; not in the 5-row spec


def prepare_data():
    """Load JF sample, encode observable state, build (X, W, Y, π)."""
    data = prepare_jf_data_granular()
    p_kt, _ = load_table4_mat()
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data
    N = len(D)

    state_obs = np.array([state_code(int(ebin_arr[i]), int(partic_arr[i]))
                          for i in range(N)], dtype=int)

    X_cols = [v for v in COV_VARS if v in df_incl.columns]
    X = df_incl[X_cols].fillna(0).to_numpy(float)

    # Recover π from pscorewt + D (KT IPW convention)
    pi = np.where(D == 1, 1.0 / pscorewt, 1.0 - 1.0 / pscorewt)
    pi = np.clip(pi, 0.02, 0.98)

    return D, state_obs, X, person_id, pscorewt, pi, df_incl, p_kt


def compute_B(D, state_obs, pscorewt):
    """B[i, r] = (2 D_i − 1) · pscorewt_i · 1{state_i = r}, for r ∈ {0n, 1n, 2n, 0p, 2p}."""
    w = (2.0 * D - 1.0) * pscorewt
    N = len(D)
    B = np.zeros((N, 5))
    for j, s in enumerate([S_0N, S_1N, S_2N, S_0P, S_2P]):
        B[:, j] = (state_obs == s).astype(float) * w
    return B


# =============================================================================
# FIRST STAGE: OLS, FULL-SAMPLE (NO CROSS-FITTING)
# =============================================================================
def estimate_b0_full(D, state_obs, X_raw, pscorewt, verbose=True):
    """Fit OLS on the full sample (no cross-validation, no splitting).

    For each row r ∈ {0n, 1n, 2n, 0p, 2p}, regress B[:, r] on X (NOT
    on W) using `LinearRegression` and return the in-sample predictions.

    IMPORTANT: we deliberately DO NOT include the treatment indicator W
    in the design matrix.  The CLP framework specifies b₀(X) = E[B | X],
    marginalised over W via the IPW factor (2D − 1) · pscorewt embedded
    in B.  Including W in the regression would let OLS predict the
    sign of B almost exactly (since B's sign is determined by W),
    overfitting b̂ ≈ B and producing pathological vertex selections in
    the downstream LP.  Matches the convention in
    `CLP_final_OLS_variants.py: estimate_b0_no_split` and
    `CLP_granular_final_combined.py: estimate_b0`.
    """
    N = len(D)
    B = compute_B(D, state_obs, pscorewt)
    k = B.shape[1]
    sc = StandardScaler()
    Xs = sc.fit_transform(X_raw)

    b_hat = np.zeros((N, k))
    for j in range(k):
        y = B[:, j]
        if y.std() < 1e-10:
            b_hat[:, j] = y.mean()
            continue
        mdl = LinearRegression(n_jobs=1)
        mdl.fit(Xs, y)
        b_hat[:, j] = mdl.predict(Xs)
    if verbose:
        residuals = B - b_hat
        max_diff = float(np.max(np.abs(B.mean(axis=0) - b_hat.mean(axis=0))))
        per_r2 = 1.0 - residuals.var(axis=0) / (B.var(axis=0) + 1e-12)
        print(f"    [OLS/full] {N:,} obs x {X_raw.shape[1]} features (X only, NOT X+W); "
              f"per-row R²: min={per_r2.min():+.3f} mean={per_r2.mean():+.3f} max={per_r2.max():+.3f}")
        print(f"               max|E[B] - E[b_hat]| = {max_diff:.2e}")
    return b_hat, B


# =============================================================================
# RECESSION-CONE TEST  (single small LP per observation)
# =============================================================================
def recession_unbounded(b_hat_i, A_mat):
    """min_d d·b_hat_i  s.t.  Aᵀd ≥ 0,  d ∈ [-1, 1]^k.
    Returns (is_unbounded, slope, d_opt).  If slope < -EPS_RECESS, the
    per-i LP `min ν · b_hat_i s.t. Aᵀν ≥ q` would be unbounded along
    that direction."""
    k = A_mat.shape[0]
    res = linprog(c=b_hat_i,
                  A_ub=-A_mat.T, b_ub=np.zeros(A_mat.shape[1]),
                  bounds=[(-1.0, 1.0)] * k,
                  method='highs', options={'disp': False})
    if res.status != 0:
        return False, float('nan'), None
    slope = float(res.fun)
    return (slope < -EPS_RECESS), slope, res.x


# =============================================================================
# DIAGNOSTICS PER (TARGET, DIRECTION)
# =============================================================================
def vertex_label(v):
    return "(" + ", ".join(
        ("  0" if abs(x) < EPS_NORM
         else (f"+{x:.0f}" if abs(x - round(x)) < EPS_NORM
                else f"+{x:.2f}") if x > 0
         else (f"-{abs(x):.0f}" if abs(x - round(x)) < EPS_NORM
                else f"{x:.2f}"))
        for x in v
    ) + ")"


def per_direction_diagnostic(q, b_hat, B, A_mat, label, dir_label, verbose=True):
    """For one direction q (+e_j or −e_j), compute:
        - vertex set V_q
        - per-i binding vertex index
        - per-i recession-cone unbounded flag
        - per-i contribution v* · B_obs[i]
        - margin to second-best vertex per i
    Returns a dict with all of the above plus aggregates.
    """
    V = enumerate_dual_vertices(A_mat, q)
    n_vert = len(V)
    if n_vert == 0:
        return None
    V = np.asarray(V)
    ninfs = np.linalg.norm(V, ord=np.inf, axis=1)

    # Per-i: binding vertex and contribution
    n = len(b_hat)
    vals = b_hat @ V.T                              # (n, V)
    sorted_idx = np.argsort(vals, axis=1)
    bind_idx = sorted_idx[:, 0]
    second_idx = sorted_idx[:, 1] if n_vert > 1 else sorted_idx[:, 0]
    margins = vals[np.arange(n), second_idx] - vals[np.arange(n), bind_idx]
    contribs = (V[bind_idx] * B).sum(axis=1)         # (n,)

    # Recession test per i
    recess_flag = np.zeros(n, bool)
    recess_slope = np.zeros(n)
    for i in range(n):
        unb, slope, _ = recession_unbounded(b_hat[i], A_mat)
        recess_flag[i] = unb
        recess_slope[i] = slope

    # Aggregates
    counts = np.bincount(bind_idx, minlength=n_vert)
    shares = counts / max(1, n)
    H = -np.sum([p * np.log(max(p, 1e-12)) for p in shares if p > 0])
    knife_share = float(np.mean(margins < KNIFE_TOL))
    n_unb = int(recess_flag.sum())

    if verbose:
        print(f"\n        ── direction = {dir_label} ──")
        print(f"           #vertices = {n_vert}  ||·||_∞ ∈ [{ninfs.min():.2f}, {ninfs.max():.2f}]")
        print(f"           Binding-vertex entropy = {H:.4f}  "
              f"(uniform = {np.log(n_vert):.4f})")
        print(f"           Recession unbounded: {n_unb}/{n} ({100*n_unb/n:.1f}%)")
        print(f"           Knife-edge (margin < {KNIFE_TOL:.0e}): {100*knife_share:.2f}%")
        print(f"           Vertex usage:")
        for k in range(n_vert):
            if counts[k] == 0:
                continue
            sum_c = float((V[k] * B[bind_idx == k]).sum())
            mean_c = sum_c / counts[k] if counts[k] else 0.0
            print(f"             v[{k}]  {vertex_label(V[k])}  "
                  f"count={counts[k]:>7d}  share={shares[k]:.3f}  "
                  f"Σ(v·B)={sum_c:+10.4f}  mean={mean_c:+.4f}")

    return dict(
        V=V, ninfs=ninfs,
        bind_idx=bind_idx,
        contribs=contribs,
        recess_flag=recess_flag, recess_slope=recess_slope,
        margins=margins,
        counts=counts, shares=shares, entropy=H,
        knife_share=knife_share, n_unbounded=n_unb,
        n_vert=n_vert,
    )


# =============================================================================
# IPW SOURCE MARGINALS  →  π = β / P^A(s)
# =============================================================================
# For each col j of the 5x9 KT spec, map to the integer state code of its
# source under our encoding (0=0n, 1=1n, 2=2n, 3=0p, 4=2p).  '0r' shares
# cells with '0p', '2u' with '2p'.
COL_SOURCE_STATE = [
    S_0N,    # 0: (0n, 1r)
    S_0P,    # 1: (0r, 0n)
    S_2N,    # 2: (2n, 1r)
    S_0P,    # 3: (0r, 2n)
    S_0P,    # 4: (0r, 1r)
    S_0P,    # 5: (0r, 1n)
    S_1N,    # 6: (1n, 1r)
    S_0P,    # 7: (0r, 2u)
    S_2P,    # 8: (2u, 1r)
]
COL_SOURCE_LABEL = ["0n", "0r", "2n", "0r", "0r", "0r", "1n", "0r", "2u"]


def ipw_source_marginal(state_code_value, D, state_obs, pi):
    """P̂^A(Y = state_code_value) via Horvitz–Thompson IPW on the AFDC arm."""
    contribs = np.where(D == 0,
                        (state_obs == state_code_value).astype(float) / (1.0 - pi),
                        0.0)
    return float(contribs.mean())


def all_source_marginals(D, state_obs, pi):
    """Return P̂^A for the 5 observable states."""
    return {s: ipw_source_marginal(s, D, state_obs, pi)
            for s in [S_0N, S_1N, S_2N, S_0P, S_2P]}


def clip01(x):
    return float(max(0., min(1., x))) if not np.isnan(x) else float('nan')


# =============================================================================
# COMPOSITE Q VECTORS  (KT-style: TUW, TUWelf, Exit)
# =============================================================================
def kt_composite_q_vectors(p_kt):
    """Build the three KT composite q vectors in β-units (matches
    CLP_final.py's convention).  All three are pre-divided by their
    source-marginal denominators (we keep this as-is to mirror the
    legacy CLP_final implementation)."""
    p00_c = p_kt['p00_c']; p01_c = p_kt['p01_c']
    p10_c = p_kt['p10_c']; p20_c = p_kt['p20_c']

    # TUW = take-up work: not-working source → working destination
    # 5x9 cols: 0=(0n,1r), 1=(0r,0n), 2=(2n,1r), 3=(0r,2n), 4=(0r,1r),
    #           5=(0r,1n), 6=(1n,1r), 7=(0r,2u), 8=(2u,1r)
    q_tuw = np.array([1, 0, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    q_tuw /= max(1e-12, p00_c + p01_c)

    # TUWelf = take-up welfare: off-welfare → on-welfare
    q_tuwelf = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=float)
    q_tuwelf /= max(1e-12, p00_c + p10_c + p20_c)

    # Exit 0r: 0r → off-welfare
    q_exit = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    q_exit /= max(1e-12, p01_c)

    return {
        "TUW (Take-Up Work)":      q_tuw,
        "TUWelf (Take-Up Welfare)": q_tuwelf,
        "Exit 0r (welfare-zero -> off welfare)": q_exit,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 96)
    print("  clp_ols.py  --  coarse 5x9 KT, OLS in-sample, vertex enumeration")
    print("=" * 96)

    print("\n[STAGE 0] Load data")
    D, state_obs, X, person_id, pscorewt, pi, df_incl, p_kt = prepare_data()
    print(f"           N = {len(D):,}  X = {X.shape}  P(D=1) = {D.mean():.4f}")
    print(f"           pi range: [{pi.min():.3f}, {pi.max():.3f}]")

    print("\n[STAGE 1] First stage — OLS, full sample (no cross-fitting)")
    b_hat, B_obs = estimate_b0_full(D, state_obs, X, pscorewt, verbose=True)

    A_mat = build_A()
    print(f"           A = {A_mat.shape}  rank = {np.linalg.matrix_rank(A_mat)}")

    # IPW source marginals  (used to convert β → π)
    pa_by_state = all_source_marginals(D, state_obs, pi)
    print(f"\n           IPW source marginals (control arm):")
    for s_lbl, s_code in [("0n", S_0N), ("1n", S_1N), ("2n", S_2N),
                          ("0p/0r", S_0P), ("2p/2u", S_2P)]:
        print(f"             P̂^A({s_lbl:<6}) = {pa_by_state[s_code]:.4f}")

    print("\n[STAGE 2] Individual β bounds (and π = β / P^A(s)) via vertex-enum CLP")
    results = []
    diag_rows = []
    print(f"    {'target':<14}  {'src':<4}  {'P^A':>7}  "
          f"{'lb_β':>9} {'ub_β':>9} {'wβ':>9}    "
          f"{'lb_π':>7} {'ub_π':>7} {'wπ':>7}")
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

        src_state = COL_SOURCE_STATE[j]
        src_lbl   = COL_SOURCE_LABEL[j]
        p_A_s     = pa_by_state[src_state]
        if p_A_s > 1e-12:
            lb_pi_raw = lb_hat / p_A_s
            ub_pi_raw = ub_hat / p_A_s
            ci_lb_pi  = (ci_lb[0] / p_A_s, ci_lb[1] / p_A_s)
            ci_ub_pi  = (ci_ub[0] / p_A_s, ci_ub[1] / p_A_s)
        else:
            lb_pi_raw = ub_pi_raw = float('nan')
            ci_lb_pi  = ci_ub_pi  = (float('nan'), float('nan'))

        results.append(dict(
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
        print(f"    {BETA_LABELS[j]:<14}  {src_lbl:<4}  {p_A_s:>7.4f}  "
              f"{lb_hat:>+9.4f} {ub_hat:>+9.4f} {ub_hat - lb_hat:>+9.4f}    "
              f"{clip01(lb_pi_raw):>7.4f} {clip01(ub_pi_raw):>7.4f} "
              f"{clip01(ub_pi_raw) - clip01(lb_pi_raw):>7.4f}")

    print("\n[STAGE 3] KT composite bounds  (q pre-divided ⇒ already in π-units)")
    composites = kt_composite_q_vectors(p_kt)
    for cname, q_beta in composites.items():
        ub_hat, c_up, _, _, _ = clp_estimate(q_beta, b_hat, B_obs, A_mat)
        neg_lb_c, c_dn, _, _, _ = clp_estimate(-q_beta, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb_c
        ci_ub = multiplier_bootstrap_ci(c_up, n_bs=N_BOOTSTRAP,
                                         alpha=ALPHA, person_id=person_id)
        ci_lb_raw = multiplier_bootstrap_ci(c_dn, n_bs=N_BOOTSTRAP,
                                             alpha=ALPHA, person_id=person_id)
        ci_lb = (-ci_lb_raw[1], -ci_lb_raw[0])
        results.append(dict(
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
        print(f"    {cname:<48}  "
              f"β ∈ [{lb_hat:+.4f}, {ub_hat:+.4f}]   "
              f"π ∈ [{clip01(lb_hat):.4f}, {clip01(ub_hat):.4f}]")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_CSV, index=False)
    print(f"\n  Wrote {OUT_CSV}  ({len(df_results)} rows)")

    # ------------------------------------------------------------------
    # DIAGNOSTICS BLOCK
    # ------------------------------------------------------------------
    print("\n[STAGE 4] Diagnostics — vertex usage and recession test")

    # Diagnose β(2n,1r) and β(1n,1r) in both directions (most informative)
    diag_targets = [
        (2, "beta(2n,1r)"),
        (6, "beta(1n,1r)"),
    ]
    for col_idx, lbl in diag_targets:
        print(f"\n    Target: {lbl}  (col {col_idx})")
        for sign, dir_lbl in [(+1.0, "+q (UB)"), (-1.0, "-q (LB)")]:
            q = np.zeros(9); q[col_idx] = sign
            diag = per_direction_diagnostic(q, b_hat, B_obs, A_mat,
                                             label=lbl, dir_label=dir_lbl,
                                             verbose=True)
            if diag is None:
                continue
            for k in range(diag['n_vert']):
                diag_rows.append(dict(
                    target=lbl, direction=dir_lbl, v_idx=k,
                    nu_value=";".join(f"{x:+.2f}" for x in diag['V'][k]),
                    nu_inf_norm=float(diag['ninfs'][k]),
                    bind_count=int(diag['counts'][k]),
                    bind_share=float(diag['shares'][k]),
                    sum_contrib=float(
                        (diag['V'][k] * B_obs[diag['bind_idx'] == k]).sum()),
                    binding_entropy=float(diag['entropy']),
                    knife_share=float(diag['knife_share']),
                    n_unbounded=int(diag['n_unbounded']),
                    n_total=len(b_hat),
                ))

    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(OUT_DIAG_CSV, index=False)
    print(f"\n  Wrote {OUT_DIAG_CSV}  ({len(df_diag)} rows)")

    print(f"\nTotal runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
