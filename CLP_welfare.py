"""
CLP_welfare.py
=====================


For each (spec ∈ {spec0, spec1, spec5, spec18}) × (composite question) ×
(mode ∈ {A_use_phase1, B_toss_lp_fail, C_toss_fail_cap}) × (direction):

  - solve the per-i LP at q_unit = q_dollars / ‖q_dollars‖∞ (matches
    welfare_v3's unit-norm rescaling)
  - group kept observations by their rounded ν vector
  - per binding ν: n_picked, n_ok, n_cap, n_fail (welfare_v3 doesn't
    use cap-vs-ok distinction in its A-mode but we still flag is_box_face)
  - mean_b_hat per group, sum_B_obs per group
  - contribution_term, contribution_share — both RESCALED to dollars by
    multiplying by q_scale, so Σ contribution_share = σ̂_dollars exactly

Bound identity (per spec, question, mode):
   ub_welfare ($)  =  Σ contribution_share over direction='UB'
   lb_welfare ($)  = −Σ contribution_share over direction='LB'

Per-transition (individual β) inspects are also written for spec0 and
spec1 (skipped for spec5 / spec18 because they have many cells and the
per-transition CSV would balloon).

Outputs (incremental, one file per spec):
  out_welfare_v3_inspect/welfare_v3_results_<spec>.csv     bound rows
  out_welfare_v3_inspect/welfare_v3_inspect_<spec>.csv     per-vertex decomposition
  welfare_v3_inspect_results_all.csv                       cumulative
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# Import all welfare_v3 machinery — do NOT modify the source module
from welfare_v3 import (
    SPECS, COV_VARS, K_FOLDS, RANDOM_SEED,
    NU_LO, NU_HI, GRANT_BY_SIZE, AFDC_DELTA, AFDC_TAU, N_BOOTSTRAP,
    prepare_jf_data_granular,
    build_A, compute_B, compute_source_probs,
    compute_cell_means, compute_pool_mean,
    compute_delta_table,
    _find_feasible_nu,
    estimate_b0_lasso,
    composite_questions, build_q_dollars,
    _is_subset_of, _is_off_welfare, _is_on_welfare,
    _is_working, _is_not_working,
)


# =============================================================================
# CONFIG
# =============================================================================
SPECS_TO_RUN = ["spec0", "spec1", "spec5", "spec18"]
PER_TRANSITION_SPECS = {"spec0", "spec1"}   # write per-β inspects only here

ROUND_NU       = 4
EPS_CAP        = 1e-6
MODES_TO_RUN   = ["A_use_phase1", "B_toss_lp_fail", "C_toss_fail_cap"]

OUT_DIR        = "/Users/gevorgkhandamiryan/Desktop/cursorclp/check/finalcheck"
OUT_SUBDIR     = os.path.join(OUT_DIR, "out_welfare_v3_inspect")
os.makedirs(OUT_SUBDIR, exist_ok=True)
OUT_RESULTS_ALL = os.path.join(OUT_DIR, "welfare_v3_inspect_results_all.csv")


# =============================================================================
# LP CORE WITH PER-i ν TRACKING
# Mirrors welfare_v3.clp_estimate_three_modes but also returns nu_used.
# =============================================================================
def clp_estimate_three_modes_with_nu(q_unit, b_hat, B_obs, A_mat,
                                       nu_lo=NU_LO, nu_hi=NU_HI):
    """Same LP loop as welfare_v3.clp_estimate_three_modes but stores
    per-i ν so we can group observations by their ν vector for inspection.

    Returns
    -------
    c_phase1 : (N,) contributions using Phase-I fallback on failure (mode A)
    c_actual : (N,) ν' B from the ACTUAL LP optimum (NaN if LP failed)
    status   : (N,) 'ok' / 'cap' / 'fail'
    fallback : (k,) Phase-I feasible ν (None if Phase-I infeasible)
    nu_used  : (N, k) ν actually used per i (LP optimum for ok/cap;
                fallback for fail; the SAME nu_used is used for both
                c_phase1 (fail rows get fallback) and c_actual
                (fail rows are left at fallback but we set c_actual to NaN))
    """
    n, k = b_hat.shape
    A_ub_neg = -A_mat.T
    b_ub_neg = -np.asarray(q_unit, dtype=float)
    bounds = [(nu_lo, nu_hi)] * k
    nu_cap = max(abs(nu_lo), abs(nu_hi))

    fallback = _find_feasible_nu(q_unit, A_mat, nu_lo, nu_hi)
    if fallback is None:
        return (np.full(n, np.nan), np.full(n, np.nan),
                np.full(n, 'phase1_infeas', dtype=object),
                None, np.zeros((n, k)))

    c_phase1 = np.zeros(n)
    c_actual = np.full(n, np.nan)
    status = np.empty(n, dtype=object)
    nu_used = np.zeros((n, k))

    for i in range(n):
        res = linprog(c=b_hat[i], A_ub=A_ub_neg, b_ub=b_ub_neg,
                      bounds=bounds, method='highs', options={'disp': False})
        if res.status != 0:
            status[i] = 'fail'
            nu_used[i] = fallback
            c_phase1[i] = float(fallback @ B_obs[i])
        else:
            nu_i = res.x
            status[i] = 'cap' if np.any(np.abs(nu_i) > nu_cap - EPS_CAP) else 'ok'
            nu_used[i] = nu_i
            val = float(nu_i @ B_obs[i])
            c_phase1[i] = val
            c_actual[i] = val
    return c_phase1, c_actual, status, fallback, nu_used


# =============================================================================
# INSPECT ROW BUILDER
# =============================================================================
def _vertex_key(nu_vec, ndp=ROUND_NU):
    return ";".join(f"{x:+.{ndp}f}" for x in np.round(nu_vec, ndp))


def build_inspect_rows(spec_key, target, kind,
                        mode, direction, kept_mask,
                        nu_arr, status_arr, b_hat, B_obs, box_nu_cap,
                        q_vec_unit, q_scale):
    """Group kept observations by their (rounded) ν vector and emit one row
    per binding ν.  contribution_term and contribution_share are rescaled
    to dollars by multiplying ν·sum_B by q_scale.
    """
    if not kept_mask.any():
        return []
    idx_kept = np.where(kept_mask)[0]
    N_kept = len(idx_kept)
    k = B_obs.shape[1]

    groups = {}
    for i in idx_kept:
        key = _vertex_key(nu_arr[i])
        groups.setdefault(key, []).append(i)

    rows = []
    for vkey, members in groups.items():
        idxs = np.asarray(members, dtype=int)
        nu_rep = nu_arr[idxs[0]]
        ninf = float(np.max(np.abs(nu_rep)))
        is_box_face = bool(ninf >= box_nu_cap - EPS_CAP)
        statuses = status_arr[idxs]
        n_ok   = int((statuses == 'ok').sum())
        n_cap  = int((statuses == 'cap').sum())
        n_fail = int((statuses == 'fail').sum())
        mean_b = b_hat[idxs].mean(axis=0)
        sum_B  = B_obs[idxs].sum(axis=0)
        contribution_term_dol  = float(nu_rep @ sum_B) * q_scale
        contribution_share_dol = contribution_term_dol / N_kept
        rows.append(dict(
            spec_key=spec_key,
            kind=kind, target=target,
            mode=mode, direction=direction,
            q_vector=";".join(f"{x:+.4f}" for x in q_vec_unit),  # unit-norm q
            q_scale=q_scale,                                       # $ rescaling
            box_nu_cap=box_nu_cap,
            vertex_nu=vkey, vertex_ninf=ninf,
            is_box_face=is_box_face, is_binding=True,
            n_picked=len(idxs), n_ok=n_ok, n_cap=n_cap, n_fail=n_fail,
            mean_b_hat=";".join(f"{x:+.5f}" for x in mean_b),
            sum_B_obs=";".join(f"{x:+.5f}" for x in sum_B),
            contribution_term=contribution_term_dol,
            contribution_share=contribution_share_dol,
            N_kept=N_kept,
        ))
    rows.sort(key=lambda r: -abs(r['contribution_share']))
    cum = 0.0
    for rank, r in enumerate(rows, 1):
        cum += r['contribution_share']
        r['rank_in_group']    = rank
        r['cumulative_share'] = cum
    return rows


# =============================================================================
# WELFARE BOUND + INSPECT FOR ONE TARGET (one q_dollars)
# =============================================================================
def welfare_bound_with_detail(target_label, kind,
                                q_dollars, b_hat, B_obs, A_mat,
                                spec_key, person_id):
    """Run welfare_v3's three-mode LP (with q-unit rescaling) for both
    UB and LB directions; capture per-i ν; build per-vertex inspect rows.

    Returns (bound_rows, detail_rows).  Identity check (per mode):
        ub_welfare_dollars = Σ contribution_share over direction='UB'
    """
    qmax = float(np.max(np.abs(q_dollars))) if q_dollars.size else 0.0
    q_scale = max(qmax, 1.0)
    q_unit = q_dollars / q_scale
    box_nu_cap = max(abs(NU_LO), abs(NU_HI))

    bound_rows = []
    detail_rows = []

    # Run UB and LB separately so we capture per-i ν per direction.
    direction_outputs = {}
    for direction, sign in [('UB', +1.0), ('LB', -1.0)]:
        c_phase1, c_actual, status, fb, nu_used = (
            clp_estimate_three_modes_with_nu(
                sign * q_unit, b_hat, B_obs, A_mat))
        direction_outputs[direction] = dict(
            c_phase1=c_phase1, c_actual=c_actual,
            status=status, fallback=fb, nu_used=nu_used)

    out_U = direction_outputs['UB']
    out_D = direction_outputs['LB']

    for mode_label, mode_key in [('A_use_phase1', 'phase1'),
                                   ('B_toss_lp_fail', 'toss'),
                                   ('C_toss_fail_cap', 'toss_cap')]:
        # Build kept masks per mode (UB direction)
        if mode_key == 'phase1':
            kept_U = np.ones(len(b_hat), dtype=bool)
            kept_D = np.ones(len(b_hat), dtype=bool)
        elif mode_key == 'toss':
            kept_U = (out_U['status'] != 'fail')
            kept_D = (out_D['status'] != 'fail')
        else:   # 'toss_cap'
            kept_U = (out_U['status'] == 'ok')
            kept_D = (out_D['status'] == 'ok')

        # Compute σ at unit-norm scale → multiply by q_scale for $-bounds
        if mode_key == 'phase1':
            sig_up_unit = (float(out_U['c_phase1'].mean())
                            if out_U['fallback'] is not None else float('nan'))
            sig_dn_unit = (float(out_D['c_phase1'].mean())
                            if out_D['fallback'] is not None else float('nan'))
        else:
            # Use c_actual restricted to kept set
            c_U = out_U['c_actual'][kept_U]
            c_D = out_D['c_actual'][kept_D]
            sig_up_unit = float(c_U.mean()) if c_U.size > 0 else float('nan')
            sig_dn_unit = float(c_D.mean()) if c_D.size > 0 else float('nan')

        ub_welf = (sig_up_unit * q_scale) if not np.isnan(sig_up_unit) else float('nan')
        lb_welf = (-sig_dn_unit * q_scale) if not np.isnan(sig_dn_unit) else float('nan')
        width   = ((ub_welf - lb_welf)
                   if not (np.isnan(ub_welf) or np.isnan(lb_welf))
                   else float('nan'))

        bound_rows.append(dict(
            spec_key=spec_key, kind=kind, target=target_label,
            mode=mode_label,
            lb_welfare=lb_welf, ub_welfare=ub_welf, width_welfare=width,
            q_scale=q_scale,
            n_total=len(b_hat),
            n_fail_ub=int((out_U['status'] == 'fail').sum()),
            n_cap_ub =int((out_U['status'] == 'cap').sum()),
            n_ok_ub  =int((out_U['status'] == 'ok').sum()),
            n_kept_ub=int(kept_U.sum()),
            n_fail_lb=int((out_D['status'] == 'fail').sum()),
            n_cap_lb =int((out_D['status'] == 'cap').sum()),
            n_ok_lb  =int((out_D['status'] == 'ok').sum()),
            n_kept_lb=int(kept_D.sum()),
        ))

        detail_rows.extend(build_inspect_rows(
            spec_key, target_label, kind,
            mode_label, 'UB', kept_U,
            out_U['nu_used'], out_U['status'], b_hat, B_obs, box_nu_cap,
            q_vec_unit=+q_unit, q_scale=q_scale))
        detail_rows.extend(build_inspect_rows(
            spec_key, target_label, kind,
            mode_label, 'LB', kept_D,
            out_D['nu_used'], out_D['status'], b_hat, B_obs, box_nu_cap,
            q_vec_unit=-q_unit, q_scale=q_scale))

    return bound_rows, detail_rows


# =============================================================================
# RUN ONE SPEC
# =============================================================================
def run_one_spec(spec_key, data, fpl_monthly, g_bar_mean):
    spec = SPECS[spec_key]
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data

    print("\n" + "=" * 100)
    print(f"  {spec['name']}")
    print("=" * 100)

    A_mat = build_A(spec)
    rank = int(np.linalg.matrix_rank(A_mat))
    B_obs = compute_B(spec, D, ebin_arr, partic_arr, pscorewt)
    print(f"  A shape: {A_mat.shape}, rank: {rank}")

    avail = [v for v in COV_VARS if v in df_incl.columns]
    X_raw = df_incl[avail].fillna(0).to_numpy(float)
    b_hat = estimate_b0_lasso(B_obs, X_raw, person_id, verbose=True)
    print(f"  max|E[B]-E[b_hat]|: "
          f"{np.max(np.abs(B_obs.mean(axis=0) - b_hat.mean(axis=0))):.2e}")

    # Constant Delta_m per transition
    delta, _ = compute_delta_table(spec, df_incl, D, ebin_arr, partic_arr,
                                     pscorewt, fpl_monthly, g_bar_mean,
                                     verbose=False)

    all_bound_rows = []
    all_detail_rows = []

    # --- Composite questions (always) ---
    print(f"\n  -- Composite welfare bounds + per-vertex inspect --")
    for qname, active_cols in composite_questions(spec).items():
        if len(active_cols) == 0:
            continue
        q_dollars = build_q_dollars(spec, active_cols, delta)
        if np.all(np.abs(q_dollars) < 1e-9):
            continue
        bound_rows, detail_rows = welfare_bound_with_detail(
            qname, "composite", q_dollars, b_hat, B_obs, A_mat,
            spec_key, person_id)
        all_bound_rows.extend(bound_rows)
        all_detail_rows.extend(detail_rows)
        # One-line summary per question (mode A only)
        r_A = next((r for r in bound_rows if r['mode'] == 'A_use_phase1'), None)
        if r_A is not None:
            print(f"    {qname:<40} n_active={len(active_cols):>3}  "
                  f"lb=${r_A['lb_welfare']:>+10.2f}  "
                  f"ub=${r_A['ub_welfare']:>+10.2f}  "
                  f"width=${r_A['width_welfare']:>10.2f}  "
                  f"q_scale=${r_A['q_scale']:>8.2f}")

    # --- Per-transition (β columns) — only for spec0 and spec1 ---
    if spec_key in PER_TRANSITION_SPECS:
        print(f"\n  -- Per-transition welfare bounds + per-vertex inspect --")
        for j, (s, d) in enumerate(spec["cols"]):
            if abs(delta[j]) < 1e-9:
                continue
            q_dollars = np.zeros(len(spec["cols"])); q_dollars[j] = delta[j]
            bound_rows, detail_rows = welfare_bound_with_detail(
                f"beta({s},{d})", "per_transition", q_dollars,
                b_hat, B_obs, A_mat, spec_key, person_id)
            all_bound_rows.extend(bound_rows)
            all_detail_rows.extend(detail_rows)
            r_A = next((r for r in bound_rows if r['mode'] == 'A_use_phase1'), None)
            if r_A is not None:
                print(f"    j={j:>2}  {s}→{d:<10}  "
                      f"Δ=${delta[j]:>+8.2f}  "
                      f"lb=${r_A['lb_welfare']:>+10.2f}  "
                      f"ub=${r_A['ub_welfare']:>+10.2f}")

    return all_bound_rows, all_detail_rows


# =============================================================================
# WRITE OUTPUTS PER SPEC
# =============================================================================
RESULTS_FRONT = ['spec_key', 'kind', 'target', 'mode',
                  'lb_welfare', 'ub_welfare', 'width_welfare', 'q_scale',
                  'n_total',
                  'n_fail_ub', 'n_cap_ub', 'n_ok_ub', 'n_kept_ub',
                  'n_fail_lb', 'n_cap_lb', 'n_ok_lb', 'n_kept_lb']

INSPECT_FRONT = ['spec_key', 'kind', 'target', 'mode', 'direction',
                  'q_vector', 'q_scale', 'box_nu_cap',
                  'vertex_nu', 'vertex_ninf', 'is_box_face', 'is_binding',
                  'n_picked', 'n_ok', 'n_cap', 'n_fail',
                  'mean_b_hat', 'sum_B_obs',
                  'contribution_term', 'contribution_share',
                  'rank_in_group', 'cumulative_share', 'N_kept']


def _ordered_df(rows, front_cols):
    df = pd.DataFrame(rows)
    front = [c for c in front_cols if c in df.columns]
    other = [c for c in df.columns if c not in front]
    return df[front + other]


def write_spec_outputs(spec_key, bound_rows, detail_rows):
    df_b = _ordered_df(bound_rows, RESULTS_FRONT)
    df_i = _ordered_df(detail_rows, INSPECT_FRONT)
    p_b = os.path.join(OUT_SUBDIR, f"welfare_v3_results_{spec_key}.csv")
    p_i = os.path.join(OUT_SUBDIR, f"welfare_v3_inspect_{spec_key}.csv")
    df_b.to_csv(p_b, index=False)
    df_i.to_csv(p_i, index=False)
    print(f"    wrote {p_b}  ({len(df_b)} rows)")
    print(f"    wrote {p_i}  ({len(df_i)} rows)")
    return df_b


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("=" * 100)
    print(f"welfare_v3_inspect.py  --  per-vertex inspect of welfare_v3 bounds")
    print(f"  Specs: {SPECS_TO_RUN}")
    print(f"  Modes: {MODES_TO_RUN}")
    print(f"  ν-box: [{NU_LO}, {NU_HI}]  (q rescaled to unit-norm before LP)")
    print(f"  Per-spec output: {OUT_SUBDIR}/")
    print("=" * 100)

    print("\n[STAGE 0] Load data")
    data = prepare_jf_data_granular()
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data

    # IPW-weighted sample means of FPL_monthly and G_bar (same as welfare_v3)
    fpl_arr = df_incl['F3_nextsizeup'].to_numpy(float)
    size_arr = df_incl['size'].to_numpy(int)
    g_bar_arr = np.array([GRANT_BY_SIZE.get(s, GRANT_BY_SIZE[3]) for s in size_arr])
    w = pscorewt
    fpl_mean = float((fpl_arr * w).sum() / w.sum())
    g_bar_mean = float((g_bar_arr * w).sum() / w.sum())
    print(f"  FPL_monthly = ${fpl_mean:.2f}/mo   G_bar = ${g_bar_mean:.2f}/mo")

    all_bound_dfs = []
    for spec_key in SPECS_TO_RUN:
        spec_t0 = time.time()
        bound_rows, detail_rows = run_one_spec(
            spec_key, data, fpl_mean, g_bar_mean)
        df_b = write_spec_outputs(spec_key, bound_rows, detail_rows)
        all_bound_dfs.append(df_b)
        print(f"    spec {spec_key} done in "
              f"{(time.time() - spec_t0)/60:.1f} min  "
              f"(elapsed total {(time.time() - t0)/60:.1f} min)")

    if all_bound_dfs:
        df_all = pd.concat(all_bound_dfs, ignore_index=True)
        df_all.to_csv(OUT_RESULTS_ALL, index=False)
        print(f"\n  Wrote {OUT_RESULTS_ALL}  ({len(df_all)} rows)")

    print(f"\nTotal runtime: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
