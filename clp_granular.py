"""
clp_granular.py
===============
Granular CLP pipeline with carefully-instrumented LP box / fallback
handling, 5 modes per estimator config, and diagnostics.

Counterpart of CLP_granular_final_combined.py with the following
deliberate changes:

  Specs       : {spec1, spec5, spec8, spec11, spec13, spec14, spec14_alt,
                 spec18}                                              (8 specs)
  Estimators  : {LASSO, Ridge}  × splitting {"cv"} × covariates {"base", "econ"}
                plus  OLS  × splitting {"full"} × covariates {"base"}
                                                                 (5 configs)
  Targets per spec (for runtime):
        - granular_cell             : every single-col β in 2n→1r and 1n→1r flows
        - src_subbin_composite      : β(src_sub_bin, →1r) summed across dest sub-bins
                                      e.g. β(b6n, →1r) = Σ_dest β(b6n, dest)
        - dest_subbin_composite     : β(2n, dest_sub_bin) summed across src sub-bins,
                                      emitted only when the destination side is split
                                      (e.g. spec5 G3 has dest 1r split into b1r..b5r)
        - coarse_composite          : β(2n→1r), β(1n→1r) — full sum over the flow

LP modes (5):
  A   box = [−5, 5]      ν*_box-truncated + CONSTANT fallback on fail
      sample mean over ALL N (no observation dropped)
  B   box = [−200, 200]  ν*_box-truncated + CONSTANT fallback on fail
      sample mean over ALL N (no observation dropped)
  C   box = [−200, 200]  drop fails; IQR-trim contribution OUTLIERS (1 % each tail)
  D   box = [−200, 200]  drop BOTH fails AND cap-binders
  E   box = [−5, 5]      drop BOTH fails AND cap-binders

For each (spec × config × target × mode) the CSV records BOTH β-units AND
π-units bounds, the real source marginal P̂^A(src), and the diagnostic
counts (n_fail_ub, n_cap_ub, ..., n_kept_lb).

Output:
  clp_granular_results.csv          per-row bounds + diagnostic counts
"""

import os
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

import CLP_granular_final_combined as v1
from CLP_granular_final_group import (
    prepare_jf_data_granular, load_table4_mat,
    COV_VARS, engineer_features_econ,
)


# =============================================================================
# CONFIG
# =============================================================================
SPECS_TO_RUN = ["spec1", "spec5", "spec8", "spec11",
                "spec13", "spec14", "spec14_alt", "spec18"]

# (estimator, split_mode, feature_set)
CONFIGS = [
    ("LASSO", "cv",   "base"),
    ("LASSO", "cv",   "econ"),
    ("Ridge", "cv",   "base"),
    ("Ridge", "cv",   "econ"),
    ("OLS",   "full", "base"),
]

EPS_CAP            = 1e-6
OUTLIER_TAIL_PCT   = 1.0           # Mode C: trim 1% from each tail
ADDITIVITY_TOL     = 1e-6          # max gap allowed when sanity-checking
                                   # sum(granular cells) ≈ coarse composite

OUT_DIR            = "/Users/gevorgkhandamiryan/Desktop/cursorclp/check/finalcheck"
OUT_CSV            = os.path.join(OUT_DIR, "clp_granular_results.csv")


# =============================================================================
# SOURCE / DESTINATION LABEL HELPERS
# =============================================================================
def _is_subset_of(s_label, target_label):
    return v1.STATE_DEF[s_label] <= v1.STATE_DEF[target_label]


def source_subbin_cols(spec, src_target, dest_target="1r"):
    """{src_sub_bin → [col indices]} for cols whose src ⊆ src_target and
    dest ⊆ dest_target."""
    out = {}
    for j, (s, d) in enumerate(spec['cols']):
        if not _is_subset_of(s, src_target):     continue
        if not _is_subset_of(d, dest_target):    continue
        out.setdefault(s, []).append(j)
    return out


def dest_subbin_cols(spec, src_target, dest_target="1r"):
    """{dest_sub_bin → [col indices]} for cols whose src ⊆ src_target and
    dest ⊆ dest_target."""
    out = {}
    for j, (s, d) in enumerate(spec['cols']):
        if not _is_subset_of(s, src_target):     continue
        if not _is_subset_of(d, dest_target):    continue
        out.setdefault(d, []).append(j)
    return out


def coarse_flow_cols(spec, src_target, dest_target="1r"):
    return [j for j, (s, d) in enumerate(spec['cols'])
            if _is_subset_of(s, src_target) and _is_subset_of(d, dest_target)]


# =============================================================================
# FIRST-STAGE FALLTHROUGH TO v1.estimate_b0
# =============================================================================
def estimate_first_stage(estimator, split_mode, feature_set,
                          B_obs, X_raw, person_id, verbose=False):
    return v1.estimate_b0(B_obs, X_raw, person_id,
                           estimator=estimator, mode=split_mode,
                           verbose=verbose)


def build_X(feature_set, df_incl):
    if feature_set == "base":
        cols = [v for v in COV_VARS if v in df_incl.columns]
        return df_incl[cols].fillna(0).to_numpy(float)
    elif feature_set == "econ":
        X_b, X_e, _, _ = engineer_features_econ(df_incl, cov_vars=COV_VARS)
        return np.hstack([X_b, X_e]) if X_e.size else X_b
    raise ValueError(feature_set)


# =============================================================================
# CORE LP WITH DIAGNOSTIC STATUS  (per-observation, one direction)
# =============================================================================
def solve_lp_per_i(q_signed, b_hat, A_mat, nu_lo, nu_hi, hub_row=None):
    """For each i, solve  min ν·b_hat[i]  s.t.  Aᵀν ≥ q_signed,
                                                ν ∈ [nu_lo, nu_hi]^k."""
    N, k = b_hat.shape
    fallback_nu = np.full(k, -1.0)
    if hub_row is not None and 0 <= hub_row < k:
        fallback_nu[hub_row] = -2.0
    nu_cap = max(abs(nu_lo), abs(nu_hi))

    nu_full = np.full((N, k), np.nan)
    status = np.empty(N, dtype=object)
    for i in range(N):
        res = linprog(c=b_hat[i],
                      A_ub=-A_mat.T,
                      b_ub=-np.asarray(q_signed, float),
                      bounds=[(nu_lo, nu_hi)] * k,
                      method='highs', options={'disp': False})
        if res.status != 0:
            status[i] = 'fail'
            continue
        nu = res.x
        if np.any(np.abs(nu) > nu_cap - EPS_CAP):
            status[i] = 'cap'
        else:
            status[i] = 'ok'
        nu_full[i] = nu
    return nu_full, status, fallback_nu


def contribs_for_box(q_signed, b_hat, B_obs, A_mat, nu_lo, nu_hi, hub_row=None):
    N = len(b_hat)
    nu_full, status, fallback_nu = solve_lp_per_i(q_signed, b_hat, A_mat,
                                                    nu_lo=nu_lo, nu_hi=nu_hi,
                                                    hub_row=hub_row)
    contribs = np.zeros(N)
    for i in range(N):
        if status[i] == 'fail':
            contribs[i] = float(fallback_nu @ B_obs[i])
        else:
            contribs[i] = float(nu_full[i] @ B_obs[i])
    return contribs, status


# =============================================================================
# AGGREGATE OVER 5 MODES
# =============================================================================
def aggregate_mode(contribs_box5, status_box5,
                    contribs_box200, status_box200,
                    outlier_tail_pct=OUTLIER_TAIL_PCT):
    N = len(contribs_box5)
    out = {}

    # Mode A: box [-5,5], use all N
    sigma_A = float(contribs_box5.mean())
    n_fail_A = int((status_box5 == 'fail').sum())
    n_cap_A  = int((status_box5 == 'cap').sum())
    out['A'] = dict(sigma=sigma_A, n_fail=n_fail_A, n_cap=n_cap_A,
                    n_outliers=0, n_kept=N)

    # Mode B: box [-200,200], use all N
    sigma_B = float(contribs_box200.mean())
    n_fail_B = int((status_box200 == 'fail').sum())
    n_cap_B  = int((status_box200 == 'cap').sum())
    out['B'] = dict(sigma=sigma_B, n_fail=n_fail_B, n_cap=n_cap_B,
                    n_outliers=0, n_kept=N)

    # Mode C: box [-200,200], drop fails + IQR-trim
    mask_keep = (status_box200 != 'fail')
    kept = contribs_box200[mask_keep]
    if kept.size > 10:
        lo_q = np.percentile(kept, outlier_tail_pct)
        hi_q = np.percentile(kept, 100.0 - outlier_tail_pct)
        inlier_mask = (kept >= lo_q) & (kept <= hi_q)
        sigma_C = float(kept[inlier_mask].mean())
        n_outliers_C = int(kept.size - int(inlier_mask.sum()))
        n_kept_C = int(inlier_mask.sum())
    else:
        sigma_C = float('nan')
        n_outliers_C = 0
        n_kept_C = int(kept.size)
    out['C'] = dict(sigma=sigma_C, n_fail=n_fail_B, n_cap=n_cap_B,
                    n_outliers=n_outliers_C, n_kept=n_kept_C)

    # Mode D: box [-200,200], drop fails AND caps
    mask_ok = (status_box200 == 'ok')
    sigma_D = (float(contribs_box200[mask_ok].mean()) if mask_ok.any()
               else float('nan'))
    out['D'] = dict(sigma=sigma_D, n_fail=n_fail_B, n_cap=n_cap_B,
                    n_outliers=0, n_kept=int(mask_ok.sum()))

    # Mode E: box [-5,5], drop fails AND caps
    mask_ok_5 = (status_box5 == 'ok')
    sigma_E = (float(contribs_box5[mask_ok_5].mean()) if mask_ok_5.any()
               else float('nan'))
    out['E'] = dict(sigma=sigma_E, n_fail=n_fail_A, n_cap=n_cap_A,
                    n_outliers=0, n_kept=int(mask_ok_5.sum()))
    return out


# =============================================================================
# BOUND COMPUTATION  for one target
# =============================================================================
def _clip01(x):
    return max(0.0, min(1.0, float(x))) if not np.isnan(x) else float('nan')


def bounds_per_target(q_unit, source_pa, b_hat, B_obs, A_mat, hub_row=None):
    """LP in β-units (no division), then convert to π by dividing by
    source_pa = P̂^A(src) and clipping to [0, 1].  CSV stores BOTH.

    Sign convention:  UB direction uses q_signed = +q_unit; LB direction
    uses -q_unit, with the LB then flipped via lb_beta = −σ_dn.
    """
    c_u_5,   st_u_5   = contribs_for_box(+q_unit, b_hat, B_obs, A_mat,
                                           nu_lo=-5.0,   nu_hi=+5.0,
                                           hub_row=hub_row)
    c_u_200, st_u_200 = contribs_for_box(+q_unit, b_hat, B_obs, A_mat,
                                           nu_lo=-200.0, nu_hi=+200.0,
                                           hub_row=hub_row)
    c_d_5,   st_d_5   = contribs_for_box(-q_unit, b_hat, B_obs, A_mat,
                                           nu_lo=-5.0,   nu_hi=+5.0,
                                           hub_row=hub_row)
    c_d_200, st_d_200 = contribs_for_box(-q_unit, b_hat, B_obs, A_mat,
                                           nu_lo=-200.0, nu_hi=+200.0,
                                           hub_row=hub_row)
    agg_U = aggregate_mode(c_u_5, st_u_5, c_u_200, st_u_200)
    agg_D = aggregate_mode(c_d_5, st_d_5, c_d_200, st_d_200)

    rows = []
    for mode in ['A', 'B', 'C', 'D', 'E']:
        sig_up = agg_U[mode]['sigma']
        sig_dn = agg_D[mode]['sigma']
        ub_beta = sig_up if not np.isnan(sig_up) else float('nan')
        lb_beta = (-sig_dn) if not np.isnan(sig_dn) else float('nan')
        width_beta = ((ub_beta - lb_beta)
                      if not (np.isnan(ub_beta) or np.isnan(lb_beta))
                      else float('nan'))

        # π conversion with the REAL source marginal
        if source_pa > 1e-12:
            lb_pi_raw = lb_beta / source_pa
            ub_pi_raw = ub_beta / source_pa
        else:
            lb_pi_raw = ub_pi_raw = float('nan')
        lb_pi = _clip01(lb_pi_raw)
        ub_pi = _clip01(ub_pi_raw)
        width_pi = ((ub_pi - lb_pi)
                    if not (np.isnan(ub_pi) or np.isnan(lb_pi))
                    else float('nan'))

        rows.append(dict(
            mode=mode,
            lb_beta=lb_beta, ub_beta=ub_beta, width_beta=width_beta,
            lb_pi=lb_pi, ub_pi=ub_pi, width_pi=width_pi,
            lb_pi_raw=lb_pi_raw, ub_pi_raw=ub_pi_raw,
            n_fail_ub=agg_U[mode]['n_fail'],   n_cap_ub=agg_U[mode]['n_cap'],
            n_fail_lb=agg_D[mode]['n_fail'],   n_cap_lb=agg_D[mode]['n_cap'],
            n_outliers_ub=agg_U[mode]['n_outliers'],
            n_outliers_lb=agg_D[mode]['n_outliers'],
            n_kept_ub=agg_U[mode]['n_kept'],
            n_kept_lb=agg_D[mode]['n_kept'],
        ))
    return rows


# =============================================================================
# RUN ONE SPEC × CONFIG
# =============================================================================
def run_one_spec_config(spec_key, estimator, split_mode, feature_set,
                         data, p_kt, X_cache):
    """Returns a list of result rows for this (spec, config)."""
    print("\n" + "=" * 100)
    print(f"  SPEC: {spec_key}    "
          f"CONFIG: {estimator} / split={split_mode} / features={feature_set}")
    print("=" * 100)
    spec = v1.SPECS[spec_key]
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data

    A_mat = v1.build_A(spec)
    rank = int(np.linalg.matrix_rank(A_mat))
    B_obs = v1.compute_B(spec, D, ebin_arr, partic_arr, pscorewt)
    print(f"    A: {A_mat.shape}  rank={rank}    B: {B_obs.shape}")

    if feature_set not in X_cache:
        X_cache[feature_set] = build_X(feature_set, df_incl)
    X_raw = X_cache[feature_set]

    b_hat = estimate_first_stage(estimator, split_mode, feature_set,
                                   B_obs, X_raw, person_id, verbose=False)
    e_mismatch = np.max(np.abs(B_obs.mean(axis=0) - b_hat.mean(axis=0)))
    print(f"    [{estimator}/{split_mode}/{feature_set}]  "
          f"max|E[B] − E[b̂]| = {e_mismatch:.2e}")

    hub_row = None
    for r_idx, r_label in enumerate(spec['rows']):
        if r_label in ('0p', '0r'):
            hub_row = r_idx
            break

    n_cols = len(spec['cols'])
    all_rows = []

    # ---------- IPW-weighted source marginal P̂^A(src) ----------
    def source_pa_for(src_label):
        if src_label not in v1.STATE_DEF:
            return float('nan')
        cells = v1.STATE_DEF[src_label]
        mask = np.zeros(len(D), dtype=bool)
        for (eb, pa) in cells:
            mask |= (ebin_arr == eb) & (partic_arr == pa)
        ctrl = (D == 0)
        wc_sum = float(pscorewt[ctrl].sum()) or 1.0
        return float(pscorewt[mask & ctrl].sum() / wc_sum)

    def run_target(label, q_unit, source_label, parent_flow, kind):
        if np.all(q_unit == 0):
            return
        pa = source_pa_for(source_label)
        if not (pa > 1e-12):
            return
        rows = bounds_per_target(q_unit, pa, b_hat, B_obs, A_mat,
                                  hub_row=hub_row)
        for r in rows:
            r['spec_key']     = spec_key
            r['estimator']    = estimator
            r['split_mode']   = split_mode
            r['feature_set']  = feature_set
            r['kind']         = kind
            r['target']       = label
            r['parent_flow']  = parent_flow
            r['source_label'] = source_label
            r['source_pa']    = pa
            r['A_rows']       = A_mat.shape[0]
            r['A_cols']       = A_mat.shape[1]
            r['rank']         = rank
            r['n_active_cols']= int(q_unit.astype(bool).sum())
            all_rows.append(r)

    # =====================================================================
    # Pretty-print one block of targets (β + π side-by-side, per mode)
    # =====================================================================
    def _print_block(title, block_rows):
        if not block_rows:
            return
        print(f"\n    {title}")
        hdr = (f"      {'target':<24}  {'src':<10}  {'#cols':>5}  "
               f"{'mode':<4}  "
               f"{'lb_β':>9}  {'ub_β':>9}  {'wβ':>8}    "
               f"{'lb_π':>8}  {'ub_π':>8}  {'wπ':>8}    "
               f"{'P^A(src)':>9}  {'fail_ub':>7}  {'cap_ub':>6}")
        print(hdr)
        for r in block_rows:
            print(f"      {r['target']:<24}  {r['source_label']:<10}  "
                  f"{r['n_active_cols']:>5d}  {r['mode']:<4}  "
                  f"{r['lb_beta']:>+9.4f}  {r['ub_beta']:>+9.4f}  "
                  f"{r['width_beta']:>8.4f}    "
                  f"{r['lb_pi']:>8.4f}  {r['ub_pi']:>8.4f}  "
                  f"{r['width_pi']:>8.4f}    "
                  f"{r['source_pa']:>9.4f}  "
                  f"{int(r['n_fail_ub']):>7d}  "
                  f"{int(r['n_cap_ub']):>6d}")

    # =====================================================================
    # 1) GRANULAR CELLS — every individual β column in the 2n→1r and 1n→1r flows
    # =====================================================================
    for flow_label, src_pool in [("2n → 1r", "2n"), ("1n → 1r", "1n")]:
        block = []
        block_start = len(all_rows)
        for j, (s, d) in enumerate(spec['cols']):
            if not (_is_subset_of(s, src_pool) and _is_subset_of(d, "1r")):
                continue
            q = np.zeros(n_cols); q[j] = 1.0
            target_lbl = f"β({s},{d})"
            run_target(target_lbl, q, source_label=s,
                       parent_flow=flow_label, kind="granular_cell")
        block = all_rows[block_start:]
        _print_block(f"Granular cells ({flow_label}):", block)

    # =====================================================================
    # 2) SOURCE SUB-BIN COMPOSITES — β(src_sub_bin, →1r) summed across dests
    # =====================================================================
    for flow_label, src_pool in [("2n → 1r", "2n"), ("1n → 1r", "1n")]:
        block_start = len(all_rows)
        for src_label, col_list in sorted(source_subbin_cols(spec, src_pool).items()):
            q = np.zeros(n_cols); q[col_list] = 1.0
            target_lbl = f"β({src_label},→1r)"
            run_target(target_lbl, q, source_label=src_label,
                       parent_flow=flow_label, kind="src_subbin_composite")
        _print_block(f"Source sub-bin composites ({flow_label}):",
                     all_rows[block_start:])

    # =====================================================================
    # 3) DEST SUB-BIN COMPOSITES — β(src_pool, dest_sub_bin) summed across srcs
    #    Only emit when destination is actually split (more than one dest sub-bin
    #    and at least one is NOT the pooled "1r" label)
    # =====================================================================
    for flow_label, src_pool in [("2n → 1r", "2n"), ("1n → 1r", "1n")]:
        dest_map = dest_subbin_cols(spec, src_pool)
        if len(dest_map) <= 1:
            continue
        block_start = len(all_rows)
        for dest_label, col_list in sorted(dest_map.items()):
            q = np.zeros(n_cols); q[col_list] = 1.0
            target_lbl = f"β({src_pool},{dest_label})"
            run_target(target_lbl, q, source_label=src_pool,
                       parent_flow=flow_label, kind="dest_subbin_composite")
        _print_block(f"Dest sub-bin composites ({flow_label}):",
                     all_rows[block_start:])

    # =====================================================================
    # 4) COARSE COMPOSITES — β(2n → 1r), β(1n → 1r)
    # =====================================================================
    block_start = len(all_rows)
    for flow_lbl, src_target in [("Coarse 2n→1r", "2n"),
                                  ("Coarse 1n→1r", "1n")]:
        cols = coarse_flow_cols(spec, src_target)
        if not cols:
            continue
        q = np.zeros(n_cols); q[cols] = 1.0
        run_target(flow_lbl, q, source_label=src_target,
                   parent_flow=flow_lbl, kind="coarse_composite")
    _print_block("Coarse composites:", all_rows[block_start:])

    return all_rows


# =============================================================================
# COMPREHENSIVE SANITY CHECKS  (cross-target additivity)
# =============================================================================
def run_sanity_checks(df, tol=ADDITIVITY_TOL):
    """For each (spec, config, mode):
       (a) Σ_(granular cells of flow F) lb_beta  ≈  coarse_composite[F] lb_beta
       (b) Σ_(src_subbin composites of flow F) lb_beta  ≈  coarse[F] lb_beta
       (c) Σ_(dest_subbin composites of flow F, when present) lb_beta
                                                        ≈  coarse[F] lb_beta
       Same for ub_beta.  Print any (spec, config, mode, flow, side) where
       the gap exceeds `tol`.  These are not bugs in the LP — they signal
       that the LP optimum at q = e_j depends on the per-cell direction in
       a way that's not exactly additive across the sum.  Reporting them
       gives you a sense of how internally consistent each spec is.
    """
    print("\n" + "=" * 100)
    print("  SANITY CHECKS: Σ(granular β) ≈ Σ(src sub-bin β) ≈ Σ(dest sub-bin β)"
          " ≈ coarse-composite β")
    print(f"  Tolerance: {tol:.0e}")
    print("=" * 100)

    cfg_cols = ['spec_key', 'estimator', 'split_mode', 'feature_set', 'mode']
    flows = [("2n → 1r", "Coarse 2n→1r"),
             ("1n → 1r", "Coarse 1n→1r")]
    sides = ['lb_beta', 'ub_beta']

    n_violations = 0
    for (cfg_key, group) in df.groupby(cfg_cols):
        for flow_label, coarse_label in flows:
            coarse = group[(group['kind'] == 'coarse_composite')
                           & (group['target'] == coarse_label)]
            if coarse.empty:
                continue
            coarse_lb = float(coarse['lb_beta'].iloc[0])
            coarse_ub = float(coarse['ub_beta'].iloc[0])

            for kind in ('granular_cell', 'src_subbin_composite',
                         'dest_subbin_composite'):
                sub = group[(group['kind'] == kind)
                            & (group['parent_flow'] == flow_label)]
                if sub.empty:
                    continue
                sum_lb = float(sub['lb_beta'].sum())
                sum_ub = float(sub['ub_beta'].sum())
                gap_lb = abs(sum_lb - coarse_lb)
                gap_ub = abs(sum_ub - coarse_ub)
                if gap_lb > tol or gap_ub > tol:
                    n_violations += 1
                    print(f"  [{','.join(str(x) for x in cfg_key)}]  "
                          f"flow={flow_label}  vs {kind}: "
                          f"Δlb={gap_lb:.2e}  Δub={gap_ub:.2e}  "
                          f"(n_cells={len(sub)})")
    if n_violations == 0:
        print("  ✓  All additivity identities hold within tolerance.")
    else:
        print(f"\n  {n_violations} violations.  These are EXPECTED when the per-cell"
              " LP optimum lies on a different vertex than the\n  sum-direction LP"
              " optimum (this is normal for granular specs with many cells and"
              " is one of the\n  reasons reporting both granular and aggregate"
              " bounds is informative).")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 100)
    print("  clp_granular.py  --  granular CLP with 5 LP modes + diagnostics")
    print(f"  Specs   : {SPECS_TO_RUN}")
    print(f"  Configs : {CONFIGS}")
    print("  Modes   : A=[−5,5] all N, B=[−200,200] all N, "
          "C=[−200,200] toss fail+IQR trim, D=[−200,200] toss fail+cap, "
          "E=[−5,5] toss fail+cap")
    print("=" * 100)

    print("\n[STAGE 0] Load data")
    data = prepare_jf_data_granular()
    p_kt, _ = load_table4_mat()

    all_rows = []
    X_cache = {}
    for spec_key in SPECS_TO_RUN:
        if spec_key not in v1.SPECS:
            print(f"  WARNING: {spec_key} not in v1.SPECS; skipping")
            continue
        for estimator, split_mode, feature_set in CONFIGS:
            rows = run_one_spec_config(
                spec_key, estimator, split_mode, feature_set,
                data, p_kt, X_cache,
            )
            all_rows.extend(rows)
            print(f"\n  elapsed = {(time.time() - t0)/60:.1f} min")

    df = pd.DataFrame(all_rows)
    front = ['spec_key', 'estimator', 'split_mode', 'feature_set',
             'kind', 'parent_flow', 'source_label', 'target', 'mode',
             'lb_beta', 'ub_beta', 'width_beta',
             'lb_pi', 'ub_pi', 'width_pi',
             'lb_pi_raw', 'ub_pi_raw',
             'source_pa', 'n_active_cols',
             'A_rows', 'A_cols', 'rank',
             'n_fail_ub', 'n_cap_ub', 'n_outliers_ub', 'n_kept_ub',
             'n_fail_lb', 'n_cap_lb', 'n_outliers_lb', 'n_kept_lb']
    front = [c for c in front if c in df.columns]
    other = [c for c in df.columns if c not in front]
    df = df[front + other]
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  Wrote {OUT_CSV}  ({len(df)} rows)")

    # -------- Cross-(spec, config) summary for the two coarse composites --------
    print("\n" + "=" * 100)
    print("  SUMMARY: coarse composites across (spec × config × mode)  "
          "[β + π side-by-side]")
    print("=" * 100)
    sub = df[df['kind'] == 'coarse_composite'].copy()
    for flow in ["Coarse 2n→1r", "Coarse 1n→1r"]:
        print(f"\n  {flow}")
        s = sub[sub['target'] == flow]
        if s.empty:
            continue
        print(f"  {'spec':<10} {'est':<6} {'split':<5} {'feat':<5} {'mode':<4}  "
              f"{'lb_β':>9} {'ub_β':>9} {'wβ':>8}    "
              f"{'lb_π':>8} {'ub_π':>8} {'wπ':>8}    "
              f"{'P^A':>7}  "
              f"{'fail_ub':>7} {'cap_ub':>6} {'fail_lb':>7} {'cap_lb':>6}")
        for _, r in s.iterrows():
            print(f"  {r['spec_key']:<10} {r['estimator']:<6} "
                  f"{r['split_mode']:<5} {r['feature_set']:<5} {r['mode']:<4}  "
                  f"{r['lb_beta']:>+9.4f} {r['ub_beta']:>+9.4f} "
                  f"{r['width_beta']:>8.4f}    "
                  f"{r['lb_pi']:>8.4f} {r['ub_pi']:>8.4f} "
                  f"{r['width_pi']:>8.4f}    "
                  f"{r['source_pa']:>7.4f}  "
                  f"{int(r['n_fail_ub']):>7d} {int(r['n_cap_ub']):>6d} "
                  f"{int(r['n_fail_lb']):>7d} {int(r['n_cap_lb']):>6d}")

    # -------- Coverage report: rows per (spec, kind) --------
    print("\n" + "=" * 100)
    print("  COVERAGE: # target rows per (spec, kind)  "
          "[expect 5 modes × #targets per (spec, kind)]")
    print("=" * 100)
    cov = (df.groupby(['spec_key', 'kind']).size()
           .unstack(fill_value=0))
    print(cov.to_string())

    # -------- Sanity check: granular ≈ src sub-bin ≈ dest sub-bin ≈ coarse --------
    run_sanity_checks(df)

    print(f"\nTotal runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
