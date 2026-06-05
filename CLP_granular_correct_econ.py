"""
CLP_granular_correct_econ.py
============================
Robustness check variant of CLP_granular_correct.py.

Uses the LARGER economically-motivated feature set (~255 features)
=== 28 base covariates plus ~227 squared and interaction terms
(see engineer_features_econ in CLP_granular_final_group.py for the
detailed feature breakdown) instead of the 28 base covariates.

Same data, same A matrices for all 9 specs, same target/composite
machinery, same person-clustered Exp(1) multiplier bootstrap.  Only
the first-stage feature set X is changed.  First stage is still
LASSO with GroupKFold cross-fitting by person.

Output paths: granular_correct_econ_{composite,target}.csv
"""

import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Reuse infrastructure from the base file and the granular pipeline.
import CLP_granular_correct as cgc
from CLP_granular_final_group import (
    prepare_jf_data_granular, load_table4_mat,
    engineer_features_econ,
)


# ===========================================================================
# X-builder: 255-feature econ matrix (28 base + ~227 interactions)
# ===========================================================================
def build_X_econ(df_incl):
    X_base_raw, X_econ_raw, feat_names, group_info = engineer_features_econ(df_incl)
    X_full = np.hstack([X_base_raw, X_econ_raw]) if X_econ_raw.size else X_base_raw
    return X_full, feat_names


# ===========================================================================
# run_spec for the econ variant
# ===========================================================================
def run_spec_econ(spec_key, data, p, run_individual_betas=False):
    spec = cgc.SPECS[spec_key]
    print(f"\n{'='*88}")
    print(f"  Running {spec_key}  [econ feature set]")
    print(f"  {spec['name']}")
    print(f"{'='*88}")

    D, ebin_arr, partic_arr, _state_row_old, df_incl, person_id, pscorewt = data

    # 1. Build A and verify
    A_mat = cgc.build_A(spec)
    cgc.verify_A(A_mat, spec)

    # 2. B-vector and source probs
    B_obs = cgc.compute_B(spec, D, ebin_arr, partic_arr, pscorewt)
    src_probs = cgc.compute_source_probs(spec, D, ebin_arr, partic_arr, pscorewt)
    print(f"    B_obs: {B_obs.shape}, sample mean B (= b0): "
          f"{B_obs.mean(axis=0).round(4)[:8]}{'...' if B_obs.shape[1] > 8 else ''}")

    # 3. Build the econ X matrix (~255 features)
    X_raw, feat_names = build_X_econ(df_incl)
    print(f"    First stage: cross-fit LASSO + GroupKFold on "
          f"{X_raw.shape[1]} features (econ set)")

    # 4. First-stage LASSO with GroupKFold (reusing cgc's estimator)
    b_hat = cgc.estimate_b0_lasso(B_obs, X_raw, person_id,
                                    K=cgc.K_FOLDS, seed=cgc.RANDOM_SEED)
    print(f"    Max |E[B] - E[b_hat]|: "
          f"{np.max(np.abs(B_obs.mean(axis=0) - b_hat.mean(axis=0))):.2e}")

    # 5. Hub row (for LP fallback)
    hub_row = None
    for r_idx, r_label in enumerate(spec["rows"]):
        if r_label in ("0p", "0r"):
            hub_row = r_idx
            break

    # 6. Target parameter individual bounds (always)
    target_results = cgc.report_target_parameters(
        b_hat, B_obs, A_mat, person_id, spec, src_probs,
        hub_row=hub_row, label=spec_key,
    )

    # 7. Composite bounds
    composite_results = cgc.compute_composite_bounds(
        b_hat, B_obs, A_mat, person_id, spec, p,
        hub_row=hub_row, label=spec_key,
    )

    # 8. Optional: all individual beta bounds
    individual_results = {}
    if run_individual_betas:
        cols = spec["cols"]
        n_beta = len(cols)
        print(f"\n    Individual beta bounds for ALL {n_beta} cols:")
        for j in range(n_beta):
            r = cgc._bounds_for_col(j, spec, b_hat, B_obs, A_mat, person_id,
                                      hub_row=hub_row)
            sp = src_probs[j]
            r['source_pa'] = sp
            r['lb_pi'] = (max(0., min(1., r['lb']/sp))
                          if sp > 1e-12 else float('nan'))
            r['ub_pi'] = (max(0., min(1., r['ub']/sp))
                          if sp > 1e-12 else float('nan'))
            beta_name = f"beta({cols[j][0]},{cols[j][1]})"
            individual_results[beta_name] = r

    return dict(
        spec_key=spec_key,
        spec=spec,
        A_shape=A_mat.shape,
        rank=np.linalg.matrix_rank(A_mat),
        target=target_results,
        composite=composite_results,
        individual=individual_results,
    )


# ===========================================================================
# Cross-spec summary + CSV writing  (mirrors cgc.main; uses econ paths)
# ===========================================================================
def write_summaries_and_csv(all_results, out_suffix):
    print(f"\n\n{'='*88}")
    print("SUMMARY: Composite bounds across specifications [econ]")
    print("=" * 88)
    print(f"{'spec':<8}  {'shape':<10}  "
          f"{'TUW':<28}  {'TUWelf':<28}  {'Exit':<28}")
    print("-" * 110)
    for spec_key, r in all_results.items():
        sh = f"{r['A_shape'][0]}x{r['A_shape'][1]}"
        cmp = r['composite']

        def fmt(c):
            if c not in cmp:
                return "N/A"
            d = cmp[c]
            return (f"[{d['lb_clipped']:.3f},{d['ub_clipped']:.3f}] "
                    f"w={d['ub_clipped']-d['lb_clipped']:.3f}")

        tuw   = fmt("Take-Up Work (not working -> working)")
        tuwf  = fmt("Take-Up Welfare (off -> on welfare)")
        exit_ = fmt("Exit 0r (on-welfare zero earn -> off welfare)")
        print(f"{spec_key:<8}  {sh:<10}  {tuw:<28}  {tuwf:<28}  {exit_:<28}")

    # Cross-spec target-parameter summary (beta and pi)
    print(f"\n\n{'='*88}")
    print("SUMMARY: Target parameters across specifications [econ]")
    print("Each entry: 'beta(s,d) = [LB_beta, UB_beta]   pi(s,d) = [LB_pi, UB_pi]'")
    print("=" * 88)
    for tp_name, _, _ in cgc.TARGET_PARAMETERS:
        print(f"\n  {tp_name}")
        print(f"  {'-' * len(tp_name)}")
        for spec_key, r in all_results.items():
            tp = r['target'].get(tp_name, {})
            n_betas = len(tp)
            sh = f"{r['A_shape'][0]}x{r['A_shape'][1]}"
            if n_betas == 0:
                print(f"    {spec_key:<8} {sh:<10}  (none)")
                continue
            print(f"    {spec_key:<8} {sh:<10} ({n_betas} matching col"
                  f"{'s' if n_betas != 1 else ''})")
            for beta_name, d in tp.items():
                pi_name = d.get('pi_name',
                                 beta_name.replace('beta(', 'pi('))
                lb_b = d['lb']; ub_b = d['ub']
                lb_pi = d.get('lb_pi', float('nan'))
                ub_pi = d.get('ub_pi', float('nan'))
                print(f"      {beta_name:<24}=[{lb_b:+.4f},{ub_b:+.4f}]    "
                      f"{pi_name:<24}=[{lb_pi:.4f},{ub_pi:.4f}]")

    # CSV: composite
    composite_path = ("/Users/gevorgkhandamiryan/Desktop/cursorclp/check/"
                      f"granular_correct{out_suffix}_composite.csv")
    rows = []
    for spec_key, r in all_results.items():
        for cname, d in r['composite'].items():
            rows.append(dict(
                spec=spec_key,
                spec_name=r['spec']['name'],
                A_rows=r['A_shape'][0], A_cols=r['A_shape'][1],
                composite=cname,
                lb=d['lb'], ub=d['ub'], width=d['width'],
                lb_clipped=d['lb_clipped'], ub_clipped=d['ub_clipped'],
                ci_outer_lo=d['ci_outer'][0], ci_outer_hi=d['ci_outer'][1],
            ))
    pd.DataFrame(rows).to_csv(composite_path, index=False)
    print(f"\n  Wrote composite results to {composite_path}")

    # CSV: target
    target_path = ("/Users/gevorgkhandamiryan/Desktop/cursorclp/check/"
                   f"granular_correct{out_suffix}_target.csv")
    rows = []
    for spec_key, r in all_results.items():
        for tp_name, betas in r['target'].items():
            for beta_name, d in betas.items():
                ci_lb_pi = d.get('ci_lb_pi', (float('nan'), float('nan')))
                ci_ub_pi = d.get('ci_ub_pi', (float('nan'), float('nan')))
                rows.append(dict(
                    spec=spec_key,
                    spec_name=r['spec']['name'],
                    A_rows=r['A_shape'][0], A_cols=r['A_shape'][1],
                    target_parameter=tp_name,
                    beta=beta_name,
                    pi=d.get('pi_name', beta_name.replace('beta(', 'pi(')),
                    source_pa=d.get('source_pa', float('nan')),
                    lb_beta=d['lb'], ub_beta=d['ub'], width_beta=d['width'],
                    ci_lb_beta_lo=d['ci_lb'][0], ci_lb_beta_hi=d['ci_lb'][1],
                    ci_ub_beta_lo=d['ci_ub'][0], ci_ub_beta_hi=d['ci_ub'][1],
                    lb_pi=d.get('lb_pi', float('nan')),
                    ub_pi=d.get('ub_pi', float('nan')),
                    width_pi=(d.get('ub_pi', float('nan'))
                              - d.get('lb_pi', float('nan'))),
                    ci_lb_pi_lo=ci_lb_pi[0], ci_lb_pi_hi=ci_lb_pi[1],
                    ci_ub_pi_lo=ci_ub_pi[0], ci_ub_pi_hi=ci_ub_pi[1],
                ))
    pd.DataFrame(rows).to_csv(target_path, index=False)
    print(f"  Wrote target-parameter results to {target_path}")


# ===========================================================================
# MAIN
# ===========================================================================
SPECS_TO_RUN = ["spec1", "spec2", "spec3", "spec4", "spec5",
                "spec6", "spec7", "spec8", "spec9"]
RUN_INDIVIDUAL_BETAS = False


def main():
    t0 = time.time()
    print("=" * 88)
    print("CLP_granular_correct_econ -- ROBUSTNESS CHECK (econ feature set)")
    print("Same as CLP_granular_correct.py except first stage uses ~255 econ")
    print("features (28 base + ~227 squared/interaction terms) instead of 28.")
    print("=" * 88)
    print(f"  Specs to run: {SPECS_TO_RUN}")
    print(f"  Individual beta bounds: {RUN_INDIVIDUAL_BETAS}")

    print(f"\n[STAGE 1] Load data via prepare_jf_data_granular()")
    data = prepare_jf_data_granular()
    p, _ = load_table4_mat()

    print(f"\n[STAGE 2] Verify all specs build cleanly")
    for spec_key in SPECS_TO_RUN:
        cgc.verify_A(cgc.build_A(cgc.SPECS[spec_key]), cgc.SPECS[spec_key])

    print(f"\n[STAGE 3] Run all specs sequentially")
    all_results = {}
    for spec_key in SPECS_TO_RUN:
        all_results[spec_key] = run_spec_econ(
            spec_key, data, p, run_individual_betas=RUN_INDIVIDUAL_BETAS,
        )

    write_summaries_and_csv(all_results, out_suffix="_econ")
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
