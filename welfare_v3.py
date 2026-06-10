"""
welfare_v3.py
=============

Implements three specs:
  spec0  KT 5x9 coarse baseline
  spec1  13x33 canonical granular layout (CLP_granular_final_group)
  spec5  11x53 granular with full G3+G7 destination split + pooled 2u

For each spec, reports:
  (i)   Per-transition Delta_m bounds (one per (src,dst) in spec.cols)
  (ii)  Composite welfare questions:
          - Take-Up Work (not working -> working)
          - Take-Up Welfare (off welfare -> on welfare)
          - Exit 0r (welfare-zero-earn -> off welfare)
          - Underreporting transitions
  (iii) Coarse flow aggregates (for spec1 and spec5):
          - Coarse 2n -> 1r:  q has Delta_m at each granular sub-bin
                              transition that aggregates into the coarse flow
          - Coarse 1n -> 1r:  similar
          - Coarse 0n -> 1r:  similar

Three-mode diagnostic per (spec, question):
  Mode A "use_phase1"   : on LP failure, substitute Phase-I feasible nu
  Mode B "toss_lp_fail" : on LP failure, DROP the observation
  Mode C "toss_fail_cap": drop LP failures AND cap-binders


GRANT SCHEDULE (KT Connecticut Jobs First)
==========================================
G_bar by AU size:
  size 2 -> $423/mo   size 3 -> $543/mo   size 4 -> $614/mo
We use the IPW-weighted SAMPLE MEAN of G_bar.

AFDC phase-out: G(e) = max(0, G_bar - tau * max(0, e - delta))
  delta = $90/mo   tau = 0.73

JF flat disregard: G_bar for any on-welfare type with reported
earnings <= FPL.  Above-FPL underreporters (2u) still report $0 and
get full G_bar.

We compute Delta_m at the IPW-weighted sample mean of FPL_monthly.


OUTPUTS
=======
  welfare_v3_results.csv  -- long-format per (spec, question, mode, target)
"""

# =============================================================================
# 1. IMPORTS AND CONSTANTS
# =============================================================================
import os
import time
import warnings
from typing import Dict, List, Tuple, FrozenSet, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

from CLP_granular_final_group import (
    PSCORE_VARS, COV_VARS,
    JF_DTA_PATH, POLICY_RULES_PATH, TABLE4_MAT_PATH,
    K_FOLDS, RANDOM_SEED,
    load_table4_mat,
    prepare_jf_data_granular,
    multiplier_bootstrap_ci,
)

N_BOOTSTRAP = 200
np.random.seed(RANDOM_SEED)

OUT_DIR = "/Users/gevorgkhandamiryan/Desktop/cursorclp/check/finalcheck"
OUT_CSV = os.path.join(OUT_DIR, "welfare_v3_results.csv")

# Wide ν-box.  We always rescale q to unit-norm before LP so |nu| stays small;
# the wide box is defense in depth.
NU_LO, NU_HI = -200.0, 200.0

# Connecticut Jobs First grant schedule
GRANT_BY_SIZE: Dict[int, float] = {2: 423.0, 3: 543.0, 4: 614.0, 5: 672.0}
AFDC_DELTA = 90.0
AFDC_TAU = 0.73


# =============================================================================
# 2. STATE_DEF + LATENT TYPE
# =============================================================================
def _fc(*pairs):
    return frozenset(pairs)


STATE_DEF: Dict[str, FrozenSet[Tuple[int, int]]] = {
    "0n":  _fc((0, 0)),
    "b1n": _fc((1, 0)), "b2n": _fc((2, 0)),
    "b3n": _fc((3, 0)), "b4n": _fc((4, 0)),
    "b5n": _fc((5, 0)),
    "b6n": _fc((6, 0)), "b7n": _fc((7, 0)),
    "b8n": _fc((8, 0)),
    "0p":  _fc((0, 1)),
    "0r":  _fc((0, 1)),
    "b1r": _fc((1, 1)), "b2r": _fc((2, 1)),
    "b3r": _fc((3, 1)), "b4r": _fc((4, 1)),
    "b5r": _fc((5, 1)),
    "b6u": _fc((6, 1)), "b7u": _fc((7, 1)),
    "b8u": _fc((8, 1)),
    "b6p": _fc((6, 1)), "b7p": _fc((7, 1)),
    "b8p": _fc((8, 1)),
    "1n":  _fc((1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
    "2n":  _fc((6, 0), (7, 0), (8, 0)),
    "1r":  _fc((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
    "1p":  _fc((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
    "2u":  _fc((6, 1), (7, 1), (8, 1)),
    "2p":  _fc((6, 1), (7, 1), (8, 1)),
    # Low-tail 1r splits (needed for spec 18 G3 destinations)
    "low_b1r": _fc((1, 1)),
    "low_b2r": _fc((2, 1)),
    "low_b3r": _fc((3, 1), (4, 1), (5, 1)),
    # Low-tail 1p observable rows (constraint rows for spec 18; strict
    # cell-equality applied in build_A — see SUBBIN_CONSTRAINT_ROWS).
    "low_b1p": _fc((1, 1)),
    "low_b2p": _fc((2, 1)),
}


# Rows in this set are treated by build_A with the strict cell-equality
# rule (cells == r_cells) rather than the overlap rule (cells & r_cells).
# This pins each constraint row to a SINGLE column with matching cells.
SUBBIN_CONSTRAINT_ROWS = {"low_b1p", "low_b2p"}


def get_latent_type(label: str) -> str:
    """Return 'n' (off-welfare), 'r' (truthful on-welfare), or 'u'
    (underreporter on-welfare).

    Conventions:
      0n, 1n, 2n, b*n          -> 'n' (off welfare)
      0r, 0p, 1r, 1p, b*r, b*p -> 'r' (truthful or zero-earnings welfare)
      2u, 2p, b6u, b7u, b8u    -> 'u' (above-FPL underreporter)

    Note: 0p == 0r (zero earnings can't underreport).
          1p is treated as truthful (1u contamination is implicit and
          rare for sub-FPL earnings).
          2p == 2u (above-FPL on welfare = underreporting, per KT
          framework where the 2r cell is empirically near-empty).
    """
    if label in ("0n", "1n", "2n") or (label.startswith("b") and label.endswith("n")):
        return "n"
    if label in ("2u", "2p") or (label.startswith("b") and label.endswith("u")):
        return "u"
    if label.startswith("b") and label.endswith("p") and label[1] in "678":
        # b6p, b7p, b8p observable -> latent 2u
        return "u"
    # Default: truthful welfare or zero-earnings on welfare
    return "r"


# =============================================================================
# 3. SPECS
# =============================================================================
def _spec0():
    """KT 5x9 coarse baseline."""
    return dict(
        name="Spec 0: 5 x 9 (KT coarse)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=[
            ("0n", "1r"),
            ("0r", "0n"),
            ("2n", "1r"),
            ("0r", "2n"),
            ("0r", "1r"),
            ("0r", "1n"),
            ("1n", "1r"),
            ("0r", "2u"),
            ("2u", "1r"),
        ],
        expected_shape=(5, 9),
    )


def _spec1():
    """13 x 33 canonical granular layout."""
    return dict(
        name="Spec 1: 13 x 33 (canonical granular)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", f"b{j}r") for j in range(1, 6)]                          # G1
            + [("0r", "0n")]                                                  # G2
            + [(f"b{j}n", "1r") for j in range(6, 9)]                         # G3
            + [("0r", f"b{j}n") for j in range(6, 9)]                         # G4
            + [("0r", f"b{j}r") for j in range(1, 6)]                         # G5
            + [("0r", f"b{j}n") for j in range(1, 6)]                         # G6
            + [(f"b{j}n", "1r") for j in range(1, 6)]                         # G7
            + [("0r", f"b{j}u") for j in range(6, 9)]                         # G8
            + [(f"b{j}u", "1r") for j in range(6, 9)]                         # G9
        ),
        expected_shape=(13, 33),
    )


def _spec5():
    """11 x 53 granular with G3, G7 fully split + pooled 2u rows."""
    return dict(
        name="Spec 5: 11 x 53 (G3+G7 fully split, 2u row pooled)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                    # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]                        # G4
            + [("0r", "1r")]                                                  # G5
            + [("0r", f"b{j}n") for j in range(1, 6)]                        # G6
            + [(f"b{j}n", f"b{i}r")
               for j in range(1, 6) for i in range(1, 6)]                    # G7: 25
            + [("0r", "2u")]                                                  # G8 pooled
            + [("2u", "1r")]                                                  # G9 pooled
        ),
        expected_shape=(11, 53),
    )


def _spec18():
    """7 x 11 (spec 15 + low_b1p, low_b2p constraint rows).

    The two sub-bin constraint rows are treated with strict cell-
    equality in build_A (via SUBBIN_CONSTRAINT_ROWS).  This pins down
    each of  beta(2n, low_b1r)  and  beta(2n, low_b2r)  via a single
    row of A, point-identifying those columns modulo the flow-balance
    rows.  See CLP_granular_final_combined.py docstring for derivation.
    """
    return dict(
        name="Spec 18: 7 x 11 (spec 15 + low_b1p/low_b2p constraints)",
        rows=["0n", "1n", "2n", "0p", "2p", "low_b1p", "low_b2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [("2n", "low_b1r"), ("2n", "low_b2r"), ("2n", "low_b3r")]
            + [("0r", "2n")]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "1r")]
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(7, 11),
    )


SPECS = {"spec0": _spec0(), "spec1": _spec1(), "spec5": _spec5(),
         "spec18": _spec18()}


# =============================================================================
# 4. PREDICATES
# =============================================================================
def _is_not_working(s):
    return all(eb == 0 for (eb, _) in STATE_DEF[s])


def _is_working(s):
    return all(eb >= 1 for (eb, _) in STATE_DEF[s])


def _is_off_welfare(s):
    return all(pa == 0 for (_, pa) in STATE_DEF[s])


def _is_on_welfare(s):
    return all(pa == 1 for (_, pa) in STATE_DEF[s])


def _is_subset_of(s_label, target_label):
    return STATE_DEF[s_label] <= STATE_DEF[target_label]


# =============================================================================
# 5. A MATRIX + B VECTOR + SOURCE PROBS
# =============================================================================
def build_A(spec) -> np.ndarray:
    rows, cols = spec["rows"], spec["cols"]
    A = np.zeros((len(rows), len(cols)), dtype=float)
    for r_idx, r_label in enumerate(rows):
        r_cells = STATE_DEF[r_label]
        is_subbin = r_label in SUBBIN_CONSTRAINT_ROWS
        for c_idx, (src, dst) in enumerate(cols):
            src_cells = STATE_DEF[src]
            dst_cells = STATE_DEF[dst]
            if is_subbin:
                # Strict cell-equality (used for sub-bin constraint rows)
                src_in = (src_cells == r_cells)
                dst_in = (dst_cells == r_cells)
            else:
                src_in = bool(src_cells & r_cells)
                dst_in = bool(dst_cells & r_cells)
            if dst_in and not src_in:
                A[r_idx, c_idx] = +1.0
            elif src_in and not dst_in:
                A[r_idx, c_idx] = -1.0
    return A


def compute_B(spec, D, ebin_arr, partic_arr, pscorewt) -> np.ndarray:
    rows = spec["rows"]
    N = len(D)
    w = (2.0 * D - 1.0) * pscorewt
    B = np.zeros((N, len(rows)), dtype=float)
    for r_idx, r_label in enumerate(rows):
        cells = STATE_DEF[r_label]
        mask = np.zeros(N, dtype=bool)
        for (eb, pa) in cells:
            mask |= (ebin_arr == eb) & (partic_arr == pa)
        B[:, r_idx] = mask.astype(float) * w
    return B


def compute_source_probs(spec, D, ebin_arr, partic_arr, pscorewt) -> np.ndarray:
    cols = spec["cols"]
    ctrl = (D == 0)
    wc_sum = pscorewt[ctrl].sum() or 1.0
    src_probs = np.zeros(len(cols))
    for c_idx, (src, _dst) in enumerate(cols):
        cells = STATE_DEF[src]
        mask = np.zeros(len(D), dtype=bool)
        for (eb, pa) in cells:
            mask |= (ebin_arr == eb) & (partic_arr == pa)
        mask &= ctrl
        src_probs[c_idx] = pscorewt[mask].sum() / wc_sum
    return src_probs


# =============================================================================
# 6. IPW-WEIGHTED CELL MEANS  (criterion D)
# =============================================================================
def compute_cell_means(df_incl, D, ebin_arr, partic_arr, pscorewt):
    """For each (ebin, partic) cell and each arm, return the IPW-weighted
    sample mean of earnq/Cbin.

    Returns
    -------
    cell_means : dict  (ebin, partic) -> {'c': float, 't': float}
                 with NaN if the cell is empty in that arm.
    """
    earnq = df_incl['earnq'].to_numpy(float)
    Cbin = df_incl['Cbin'].to_numpy(float)
    ratio = earnq / np.maximum(Cbin, 1e-12)

    means = {}
    seen_pairs = set(zip(ebin_arr.tolist(), partic_arr.tolist()))
    for (eb, pa) in seen_pairs:
        cell_mask = (ebin_arr == eb) & (partic_arr == pa)
        ent = {}
        for arm, label in [(0, 'c'), (1, 't')]:
            mask = cell_mask & (D == arm)
            if mask.sum() == 0:
                ent[label] = float('nan'); continue
            w = pscorewt[mask]
            if w.sum() < 1e-9:
                ent[label] = float('nan'); continue
            ent[label] = float((ratio[mask] * w).sum() / w.sum())
        means[(eb, pa)] = ent

    return means


def compute_pool_mean(df_incl, D, ebin_arr, partic_arr, pscorewt,
                      pool_cells, arm: str):
    """IPW-weighted mean of earnq/Cbin among observations in pool_cells
    and the specified arm.  More accurate than averaging cell means when
    the cells have different masses."""
    earnq = df_incl['earnq'].to_numpy(float)
    Cbin = df_incl['Cbin'].to_numpy(float)
    ratio = earnq / np.maximum(Cbin, 1e-12)

    pool_mask = np.zeros(len(D), dtype=bool)
    for (eb, pa) in pool_cells:
        pool_mask |= (ebin_arr == eb) & (partic_arr == pa)
    arm_mask = (D == 0) if arm == 'c' else (D == 1)
    mask = pool_mask & arm_mask
    if mask.sum() == 0:
        return float('nan')
    w = pscorewt[mask]
    if w.sum() < 1e-9:
        return float('nan')
    return float((ratio[mask] * w).sum() / w.sum())


# =============================================================================
# 7. WELFARE FORMULA HELPERS
# =============================================================================
def afdc_grant(e_monthly: float, g_bar: float,
                delta: float = AFDC_DELTA, tau: float = AFDC_TAU) -> float:
    """AFDC monthly grant given monthly earnings and base grant.

    G(e) = max(0, G_bar - tau * max(0, e - delta))   if e <= E_bar
    G(e) = 0                                          if e >  E_bar
    where E_bar = G_bar/tau + delta is the eligibility cutoff.
    """
    e_bar = g_bar / tau + delta
    if e_monthly > e_bar:
        return 0.0
    return max(0.0, g_bar - max(0.0, (e_monthly - delta) * tau))


def transfer(latent_type: str, regime: str, earnings_monthly: float,
              g_bar: float) -> float:
    """Cash transfer received by a person of given latent type under regime."""
    if latent_type == 'n':                  # off welfare
        return 0.0
    if regime == 'JF':
        # JF: flat G_bar for all on-welfare types with reported <= FPL
        # (underreporters also get G_bar because they report low).
        return g_bar
    if regime == 'AFDC':
        if latent_type == 'u' or earnings_monthly <= 0:
            # underreporter reports $0; zero-earnings get full grant
            return g_bar
        # truthful with positive earnings: phase-out
        return afdc_grant(earnings_monthly, g_bar)
    raise ValueError(f"Unknown regime: {regime}")


# =============================================================================
# 8. COMPUTE Delta_m  (CONSTANT per transition; criterion A)
# =============================================================================
def compute_delta_table(spec, df_incl, D, ebin_arr, partic_arr, pscorewt,
                        fpl_monthly: float, g_bar: float, verbose: bool = True):
    """For each (src, dst) in spec.cols, compute Delta_m using:
       earn_src = IPW-weighted CONTROL-arm mean of earnq/Cbin over src's cells
       earn_dst = IPW-weighted TREATMENT-arm mean of earnq/Cbin over dst's cells
       transfer = based on latent type of src/dst and regime

    For latent types whose dst doesn't exist in JF (e.g. b1u, b2u under
    Lemma 2): we use the source cell as the (true) earnings proxy and
    apply the JF transfer formula based on the dst latent type.

    Returns
    -------
    delta : (n_cols,) float array  in $/month
    debug : list of dicts            -- per-col breakdown for inspection
    """
    cols = spec["cols"]
    delta = np.zeros(len(cols))
    debug = []

    for j, (src, dst) in enumerate(cols):
        src_cells = STATE_DEF[src]
        dst_cells = STATE_DEF[dst]
        src_lat = get_latent_type(src)
        dst_lat = get_latent_type(dst)

        # Earnings via IPW-weighted pool mean over the cell pool.
        src_frac = compute_pool_mean(df_incl, D, ebin_arr, partic_arr, pscorewt,
                                      src_cells, arm='c')
        dst_frac = compute_pool_mean(df_incl, D, ebin_arr, partic_arr, pscorewt,
                                      dst_cells, arm='t')

        # Latent underreporter cells (b1u, b2u, 2u) under AFDC have true
        # earnings = same as their observable cell (b1p, b2p, 2p) but
        # their REPORTED earnings = 0.  TotalIncome uses TRUE earnings.
        # For dst under JF: Lemma 2 strips b1u, b2u under JF; 2u remains.
        if np.isnan(src_frac):
            src_frac = 0.0   # empty cell -> no earnings (degenerate)
        if np.isnan(dst_frac):
            # Empty dst cell in treatment arm (e.g. 2u dst): fall back to
            # AFDC-arm mean (KT framework's identification target).
            dst_frac = compute_pool_mean(df_incl, D, ebin_arr, partic_arr,
                                          pscorewt, dst_cells, arm='c')
            if np.isnan(dst_frac):
                dst_frac = 0.0

        src_earn = src_frac * fpl_monthly
        dst_earn = dst_frac * fpl_monthly
        src_xfer = transfer(src_lat, 'AFDC', src_earn, g_bar)
        dst_xfer = transfer(dst_lat, 'JF', dst_earn, g_bar)

        src_total = src_earn + src_xfer
        dst_total = dst_earn + dst_xfer
        delta[j] = dst_total - src_total

        debug.append(dict(
            j=j, src=src, dst=dst, src_lat=src_lat, dst_lat=dst_lat,
            src_frac=src_frac, dst_frac=dst_frac,
            src_earn=src_earn, dst_earn=dst_earn,
            src_xfer=src_xfer, dst_xfer=dst_xfer,
            src_total=src_total, dst_total=dst_total,
            delta=delta[j],
        ))

    if verbose:
        print(f"\n  Delta_m per transition (constant; in $/month):")
        print(f"  {'j':>3}  {'src':>4} -> {'dst':<4}  "
              f"{'src_frac':>9} {'dst_frac':>9}  "
              f"{'src_tot':>8} {'dst_tot':>8}  {'Delta':>8}")
        for d in debug:
            print(f"  {d['j']:>3}  {d['src']:>4} -> {d['dst']:<4}  "
                  f"{d['src_frac']:>9.4f} {d['dst_frac']:>9.4f}  "
                  f"{d['src_total']:>8.2f} {d['dst_total']:>8.2f}  "
                  f"{d['delta']:+8.2f}")

    return delta, debug


# =============================================================================
# 9. LP CORE  (Phase-I + three modes; criteria B, C)
# =============================================================================
LP_DIAG = {'total_solves': 0, 'fallback_count': 0, 'binding_nu_count': 0,
            'phase1_infeasible_count': 0}


def reset_lp_diagnostics():
    for k in LP_DIAG:
        LP_DIAG[k] = 0


def _find_feasible_nu(q, A_mat, nu_lo, nu_hi):
    """Phase-I LP: find any feasible nu for {A' nu >= q, nu in [lo, hi]}."""
    k = A_mat.shape[0]
    res = linprog(c=np.zeros(k), A_ub=-A_mat.T, b_ub=-np.asarray(q, float),
                  bounds=[(nu_lo, nu_hi)] * k, method='highs',
                  options={'disp': False})
    return res.x if res.status == 0 else None


def clp_estimate_three_modes(q_unit, b_hat, B_obs, A_mat,
                               nu_lo=NU_LO, nu_hi=NU_HI):
    """Per-i LP with three-status tracking.  q_unit should be unit-norm
    (max |q| <= 1) to keep dual nu small; caller rescales the result.

    Returns
    -------
    c_phase1   : (N,)   contribution using Phase-I fallback on failure
    c_actual   : (N,)   nu' B using ACTUAL LP optimum (NaN if LP failed)
    status     : (N,)   'ok' / 'cap' / 'fail'
    fallback   : (k,)   Phase-I feasible nu (None if Phase-I infeasible)
    """
    n, k = b_hat.shape
    A_ub_neg = -A_mat.T
    b_ub_neg = -np.asarray(q_unit, dtype=float)
    bounds = [(nu_lo, nu_hi)] * k
    nu_cap = max(abs(nu_lo), abs(nu_hi))

    fallback = _find_feasible_nu(q_unit, A_mat, nu_lo, nu_hi)
    if fallback is None:
        LP_DIAG['phase1_infeasible_count'] += 1
        return (np.full(n, np.nan), np.full(n, np.nan),
                np.full(n, 'phase1_infeas', dtype=object), None)

    c_phase1 = np.zeros(n)
    c_actual = np.full(n, np.nan)
    status = np.empty(n, dtype=object)

    for i in range(n):
        LP_DIAG['total_solves'] += 1
        res = linprog(c=b_hat[i], A_ub=A_ub_neg, b_ub=b_ub_neg,
                      bounds=bounds, method='highs', options={'disp': False})
        if res.status != 0:
            LP_DIAG['fallback_count'] += 1
            status[i] = 'fail'
            c_phase1[i] = float(fallback @ B_obs[i])
        else:
            nu_i = res.x
            if np.any(np.abs(nu_i) > nu_cap - 1e-6):
                LP_DIAG['binding_nu_count'] += 1
                status[i] = 'cap'
            else:
                status[i] = 'ok'
            val = float(nu_i @ B_obs[i])
            c_phase1[i] = val
            c_actual[i] = val

    return c_phase1, c_actual, status, fallback


def bounds_three_modes_dollars(q_dollars, b_hat, B_obs, A_mat, person_id,
                                 n_bs=N_BOOTSTRAP):
    """Compute UB and LB in three modes (Phase-I / toss-fail / toss-fail+cap).

    Rescales q to unit-norm before LP, then rescales sigma back to dollars."""
    qmax = float(np.max(np.abs(q_dollars))) if q_dollars.size else 0.0
    scale = max(qmax, 1.0)
    q_unit = q_dollars / scale

    out = {}
    for sign, label in [(+1.0, "ub"), (-1.0, "lb_neg")]:
        c_phase1, c_actual, status, fb = clp_estimate_three_modes(
            sign * q_unit, b_hat, B_obs, A_mat)
        if fb is None:
            out[label] = dict(
                sigma_phase1=float('nan'), ci_phase1=(float('nan'),)*2,
                sigma_toss=float('nan'), ci_toss=(float('nan'),)*2,
                sigma_toss_cap=float('nan'), ci_toss_cap=(float('nan'),)*2,
                n_total=len(b_hat), n_fail=len(b_hat), n_cap=0, n_ok=0,
                n_kept_toss=0, n_kept_toss_cap=0,
            )
            continue

        n_total = len(b_hat)
        n_fail = int(np.sum(status == 'fail'))
        n_cap = int(np.sum(status == 'cap'))
        n_ok = int(np.sum(status == 'ok'))

        # MODE A: Phase-I fallback for failed obs, all N contribute
        sigA = float(c_phase1.mean()) * scale
        ciA_unit = multiplier_bootstrap_ci(c_phase1, n_bs=n_bs,
                                            person_id=person_id)
        ciA = (ciA_unit[0] * scale, ciA_unit[1] * scale)

        # MODE B: toss LP failures, keep cap-binders
        mask_B = status != 'fail'
        n_kept_B = int(mask_B.sum())
        if n_kept_B > 0:
            c_B = c_actual[mask_B]
            pid_B = person_id[mask_B] if person_id is not None else None
            sigB = float(c_B.mean()) * scale
            ciB_unit = multiplier_bootstrap_ci(c_B, n_bs=n_bs, person_id=pid_B)
            ciB = (ciB_unit[0] * scale, ciB_unit[1] * scale)
        else:
            sigB = float('nan'); ciB = (float('nan'),)*2

        # MODE C: toss failures AND cap-binders
        mask_C = status == 'ok'
        n_kept_C = int(mask_C.sum())
        if n_kept_C > 0:
            c_C = c_actual[mask_C]
            pid_C = person_id[mask_C] if person_id is not None else None
            sigC = float(c_C.mean()) * scale
            ciC_unit = multiplier_bootstrap_ci(c_C, n_bs=n_bs, person_id=pid_C)
            ciC = (ciC_unit[0] * scale, ciC_unit[1] * scale)
        else:
            sigC = float('nan'); ciC = (float('nan'),)*2

        out[label] = dict(
            sigma_phase1=sigA, ci_phase1=tuple(ciA),
            sigma_toss=sigB, ci_toss=tuple(ciB),
            sigma_toss_cap=sigC, ci_toss_cap=tuple(ciC),
            n_total=n_total, n_fail=n_fail, n_cap=n_cap, n_ok=n_ok,
            n_kept_toss=n_kept_B, n_kept_toss_cap=n_kept_C,
        )
    return out


# =============================================================================
# 10. FIRST STAGE (LASSO + GroupKFold)
# =============================================================================
def estimate_b0_lasso(B_obs, X_raw, person_id, K=K_FOLDS, seed=RANDOM_SEED,
                       verbose=True):
    n, k = B_obs.shape
    b_hat = np.zeros((n, k))
    gkf = GroupKFold(n_splits=K)
    folds = list(gkf.split(np.arange(n), groups=person_id))
    if verbose:
        print(f"    [LASSO/cv] {n:,} obs x {X_raw.shape[1]} features, "
              f"{k} B-components, K={K} folds")
    for tr, te in folds:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_raw[tr])
        X_te = sc.transform(X_raw[te])
        for j in range(k):
            y = B_obs[tr, j]
            if y.std() < 1e-10:
                b_hat[te, j] = y.mean(); continue
            model = LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, eps=1e-4,
                            max_iter=4000, random_state=seed, n_jobs=1)
            model.fit(X_tr, y)
            b_hat[te, j] = model.predict(X_te)
    return b_hat


# =============================================================================
# 11. COMPOSITE QUESTIONS PER SPEC
# =============================================================================
def composite_questions(spec):
    """Return dict {question_name: list of col indices}.

    Always includes Take-Up Work / Take-Up Welfare / Exit 0r.  For specs
    with granular sub-bin destinations, also include Coarse 2n->1r,
    Coarse 1n->1r, Coarse 0n->1r as q-aggregations.
    """
    cols = spec["cols"]
    out = {}

    tuw = [j for j, (s, d) in enumerate(cols)
            if _is_not_working(s) and _is_working(d)]
    out["Take-Up Work"] = tuw

    tuwelf = [j for j, (s, d) in enumerate(cols)
              if _is_off_welfare(s) and _is_on_welfare(d)]
    out["Take-Up Welfare"] = tuwelf

    exit_ = [j for j, (s, d) in enumerate(cols)
             if STATE_DEF[s] == STATE_DEF["0r"] and _is_off_welfare(d)]
    out["Exit 0r"] = exit_

    underrep = [j for j, (s, d) in enumerate(cols)
                if get_latent_type(s) == 'u' or get_latent_type(d) == 'u']
    out["Underreporting (any leg touches u)"] = underrep

    # Coarse aggregates (mainly for spec1, spec5; for spec0 these coincide
    # with individual betas)
    c2n1r = [j for j, (s, d) in enumerate(cols)
              if _is_subset_of(s, "2n") and _is_subset_of(d, "1r")]
    c1n1r = [j for j, (s, d) in enumerate(cols)
              if _is_subset_of(s, "1n") and _is_subset_of(d, "1r")]
    c0n1r = [j for j, (s, d) in enumerate(cols)
              if _is_subset_of(s, "0n") and _is_subset_of(d, "1r")]
    out["Coarse 2n -> 1r"] = c2n1r
    out["Coarse 1n -> 1r"] = c1n1r
    out["Coarse 0n -> 1r"] = c0n1r

    return out


def build_q_dollars(spec, active_cols: List[int], delta: np.ndarray) -> np.ndarray:
    """q vector in $/month: zero everywhere except delta[j] at j in active_cols."""
    n = len(spec["cols"])
    q = np.zeros(n)
    for j in active_cols:
        q[j] = delta[j]
    return q


# =============================================================================
# 12. RUN ONE SPEC
# =============================================================================
def run_one_spec(spec_key: str, data, fpl_monthly, g_bar, n_bs=N_BOOTSTRAP):
    """Compute all welfare bounds for one spec."""
    spec = SPECS[spec_key]
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data

    print(f"\n{'='*88}")
    print(f"  {spec['name']}")
    print(f"{'='*88}")

    A_mat = build_A(spec)
    rank = int(np.linalg.matrix_rank(A_mat))
    print(f"  A shape: {A_mat.shape}, rank: {rank}, nnz: {int((A_mat != 0).sum())}")

    B_obs = compute_B(spec, D, ebin_arr, partic_arr, pscorewt)
    src_probs = compute_source_probs(spec, D, ebin_arr, partic_arr, pscorewt)

    # First stage
    avail = [v for v in COV_VARS if v in df_incl.columns]
    X_raw = df_incl[avail].fillna(0).to_numpy(float)
    print(f"  First-stage LASSO with GroupKFold (K={K_FOLDS}, {len(avail)} cov)")
    b_hat = estimate_b0_lasso(B_obs, X_raw, person_id)
    print(f"  max|E[B]-E[b_hat]|: "
          f"{np.max(np.abs(B_obs.mean(axis=0) - b_hat.mean(axis=0))):.2e}")

    # Constant Delta_m per transition
    delta, debug = compute_delta_table(spec, df_incl, D, ebin_arr, partic_arr,
                                        pscorewt, fpl_monthly, g_bar)

    rows = []

    # ---- Per-transition (individual beta) bounds ----
    print(f"\n  --- Per-transition bounds (in $/month) ---")
    print(f"  {'j':>3}  {'transition':<22}  {'Delta':>8}  "
          f"{'mode':<8}  {'LB':>10}  {'UB':>10}  {'width':>10}  "
          f"{'n_fail':>7}  {'n_cap':>6}")
    for j, (s, d) in enumerate(spec["cols"]):
        # Use q with Delta only at this col
        q_dollars = np.zeros(len(spec["cols"]))
        q_dollars[j] = delta[j]
        if abs(delta[j]) < 1e-9:
            # Trivial transition: Delta=0 means q=0 and bounds collapse to 0
            # We still report it for completeness.
            for mode in ["A_use_phase1", "B_toss_lp_fail", "C_toss_fail_cap"]:
                rows.append(dict(
                    spec=spec_key, kind="per_transition",
                    question=f"beta({s},{d})",
                    j=j, src=s, dst=d, delta=delta[j],
                    mode=mode,
                    lb=0.0, ub=0.0, width=0.0,
                    ci_lb_lo=0.0, ci_lb_hi=0.0,
                    ci_ub_lo=0.0, ci_ub_hi=0.0,
                    n_total=len(D), n_lp_fail=0, n_cap=0, n_ok=len(D),
                    n_kept=len(D),
                ))
            continue
        reset_lp_diagnostics()
        out = bounds_three_modes_dollars(q_dollars, b_hat, B_obs, A_mat,
                                          person_id, n_bs=n_bs)

        # Print one row per mode (showing 'A' mode for compactness)
        for mode_label, mode_key in [("A_use_phase1", "phase1"),
                                       ("B_toss_lp_fail", "toss"),
                                       ("C_toss_fail_cap", "toss_cap")]:
            ub_o = out['ub']
            lb_o = out['lb_neg']
            ub = ub_o[f'sigma_{mode_key}']
            lb = -lb_o[f'sigma_{mode_key}']
            width = ub - lb if (not np.isnan(ub) and not np.isnan(lb)) else float('nan')
            ci_ub = ub_o[f'ci_{mode_key}']
            ci_lb = (-lb_o[f'ci_{mode_key}'][1], -lb_o[f'ci_{mode_key}'][0])
            n_kept_ub = ub_o[f'n_kept_{mode_key}'] if mode_key != 'phase1' else ub_o['n_total']
            n_kept = n_kept_ub
            rows.append(dict(
                spec=spec_key, kind="per_transition",
                question=f"beta({s},{d})",
                j=j, src=s, dst=d, delta=delta[j],
                mode=mode_label,
                lb=lb, ub=ub, width=width,
                ci_lb_lo=ci_lb[0], ci_lb_hi=ci_lb[1],
                ci_ub_lo=ci_ub[0], ci_ub_hi=ci_ub[1],
                n_total=ub_o['n_total'],
                n_lp_fail=ub_o['n_fail'],
                n_cap=ub_o['n_cap'],
                n_ok=ub_o['n_ok'],
                n_kept=n_kept,
            ))
            if mode_label == "A_use_phase1":
                print(f"  {j:>3}  {f'{s}->{d}':<22}  {delta[j]:>+8.2f}  "
                      f"{mode_label:<8}  {lb:>10.4f}  {ub:>10.4f}  "
                      f"{width:>10.4f}  {ub_o['n_fail']:>7,}  "
                      f"{ub_o['n_cap']:>6,}")

    # ---- Composite questions ----
    print(f"\n  --- Composite welfare bounds (in $/month per person-quarter) ---")
    print(f"  {'question':<40}  {'n_active':>8}  {'mode':<18}  "
          f"{'LB':>10}  {'UB':>10}  {'width':>10}  "
          f"{'n_fail':>7}  {'n_cap':>6}")
    for qname, active_cols in composite_questions(spec).items():
        if len(active_cols) == 0:
            continue
        q_dollars = build_q_dollars(spec, active_cols, delta)
        if np.all(np.abs(q_dollars) < 1e-9):
            continue
        reset_lp_diagnostics()
        out = bounds_three_modes_dollars(q_dollars, b_hat, B_obs, A_mat,
                                          person_id, n_bs=n_bs)
        for mode_label, mode_key in [("A_use_phase1", "phase1"),
                                       ("B_toss_lp_fail", "toss"),
                                       ("C_toss_fail_cap", "toss_cap")]:
            ub_o = out['ub']; lb_o = out['lb_neg']
            ub = ub_o[f'sigma_{mode_key}']
            lb = -lb_o[f'sigma_{mode_key}']
            width = ub - lb if (not np.isnan(ub) and not np.isnan(lb)) else float('nan')
            ci_ub = ub_o[f'ci_{mode_key}']
            ci_lb = (-lb_o[f'ci_{mode_key}'][1], -lb_o[f'ci_{mode_key}'][0])
            n_kept = ub_o[f'n_kept_{mode_key}'] if mode_key != 'phase1' else ub_o['n_total']
            rows.append(dict(
                spec=spec_key, kind="composite",
                question=qname,
                j=-1, src='', dst='', delta=float('nan'),
                mode=mode_label,
                lb=lb, ub=ub, width=width,
                ci_lb_lo=ci_lb[0], ci_lb_hi=ci_lb[1],
                ci_ub_lo=ci_ub[0], ci_ub_hi=ci_ub[1],
                n_total=ub_o['n_total'],
                n_lp_fail=ub_o['n_fail'],
                n_cap=ub_o['n_cap'],
                n_ok=ub_o['n_ok'],
                n_kept=n_kept,
            ))
            print(f"  {qname:<40}  {len(active_cols):>8}  {mode_label:<18}  "
                  f"{lb:>10.4f}  {ub:>10.4f}  {width:>10.4f}  "
                  f"{ub_o['n_fail']:>7,}  {ub_o['n_cap']:>6,}")

    return rows


# =============================================================================
# 13. MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("="*88)
    print("welfare_v3.py")
    print("="*88)
    print(f"  N_BOOTSTRAP = {N_BOOTSTRAP}")
    print(f"  NU_BOX = [{NU_LO}, {NU_HI}]   (q always rescaled to unit norm "
          f"before LP)")
    print(f"  Specs: {list(SPECS.keys())}")

    print("\n[STAGE 0] Loading JF data ...")
    data = prepare_jf_data_granular()
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data

    # IPW-weighted sample means of FPL_monthly and G_bar
    fpl_arr = df_incl['F3_nextsizeup'].to_numpy(float)
    size_arr = df_incl['size'].to_numpy(int)
    g_bar_arr = np.array([GRANT_BY_SIZE.get(s, GRANT_BY_SIZE[3]) for s in size_arr])
    w = pscorewt
    fpl_mean = float((fpl_arr * w).sum() / w.sum())
    g_bar_mean = float((g_bar_arr * w).sum() / w.sum())
    print(f"\n  Constant Delta_m inputs (IPW-weighted sample means):")
    print(f"    FPL_monthly = ${fpl_mean:.2f}/mo")
    print(f"    G_bar       = ${g_bar_mean:.2f}/mo")

    # Cell-mean summary (control arm, observable cells)
    cell_means = compute_cell_means(df_incl, D, ebin_arr, partic_arr, pscorewt)
    print(f"\n  IPW-weighted cell means (earnq / Cbin) by observable cell:")
    print(f"  {'cell':<8}  {'control':>9}  {'JF':>9}")
    for (eb, pa), ent in sorted(cell_means.items()):
        if pa == 0:
            lab = f"b{eb}n" if eb > 0 else "0n"
        else:
            lab = f"b{eb}p" if eb > 0 else "0p"
        c = ent.get('c', float('nan'))
        t = ent.get('t', float('nan'))
        print(f"  {lab:<8}  {c:>9.4f}  {t:>9.4f}")

    all_rows = []
    for sk in ["spec0", "spec1", "spec5", "spec18"]:
        all_rows.extend(run_one_spec(sk, data, fpl_mean, g_bar_mean,
                                       n_bs=N_BOOTSTRAP))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Wrote {OUT_CSV} ({len(df):,} rows)")

    # Cross-spec summary of coarse flows (mode A only)
    print("\n\n" + "="*92)
    print("CROSS-SPEC SUMMARY: Coarse flow bounds (mode A_use_phase1, $/mo per pq)")
    print("="*92)
    print(f"  {'spec':<8}  {'question':<28}  {'n_active':>8}  "
          f"{'LB':>10}  {'UB':>10}  {'width':>10}  {'n_fail':>7}")
    flow_qs = ["Coarse 2n -> 1r", "Coarse 1n -> 1r", "Coarse 0n -> 1r"]
    for sk in ["spec0", "spec1", "spec5", "spec18"]:
        for qname in flow_qs:
            r = df[(df['spec']==sk) & (df['question']==qname)
                   & (df['mode']=='A_use_phase1')]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            print(f"  {sk:<8}  {qname:<28}  {r['n_total']:>8}  "
                  f"{r['lb']:>10.4f}  {r['ub']:>10.4f}  {r['width']:>10.4f}  "
                  f"{r['n_lp_fail']:>7,}")

    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
