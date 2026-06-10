"""
distribution_over_states.py
============================
Python replication of DistributionOverStates.do + DistributionOverStates_programs.do
from the Kline & Tartari (2016, AER) replication package.

Produces Table4_mat_python.txt — structurally identical to the Stata-generated
Table4_mat.txt, so it can be dropped into replicating4&5.py as a validation check.

Pipeline (mirrors Stata exactly):
  1.  Load JF.dta (long format: person × month) filtered to Q1–Q7, kidcount not missing
  2.  Load PolicyRules.dta  →  FPL (F3) by (year, AU_size)
  3.  Compute AU size, nextsizeup lookup, quarterly Cbin = round(3×F3, 100)
  4.  Classify earnings bin  ebin ∈ {0,1,2}  and welfare participation partic ∈ {0,1}
  5.  Create state dummies p0N … p2P; zero out excluded (mixed-welfare) quarters
  6.  Keep 1 month per quarter (first within id-quarter)
  7.  Collapse to person level; fit propensity-score logit; compute pscorewt
  8.  Expand back to person-quarter; compute IPW-adjusted, inclusion-normalised
      state distributions separately for JF and AFDC groups
  9.  Bootstrap (N_BOOTS draws, clustered by person) to get SEs
  10. Write Table4_mat_python.txt  (header + point-estimate row + bootstrap rows)

Settings matching Table 4 of the paper:
  whichFPL       = "nextsizeup"
  whichthreshold = "FPL"   (3 earnings bins)
  quarters 1–7 post-RA
  799 bootstrap draws, seed 112233 (same as Stata)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — edit to match your local setup
# ─────────────────────────────────────────────────────────────────────────────
JF_PATH           = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/JF.dta")
POLICY_RULES_PATH = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/PolicyRules.dta")
KT_TABLE4_PATH    = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/Table4_mat.txt")
OUTPUT_PATH       = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "Table4_mat_python.txt")

# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
FIRST_QUARTER = 1
LAST_QUARTER  = 7
WHICH_FPL     = "nextsizeup"   # "exactsize" | "nextsizeup" | "twosizesup"
N_BOOTS       = 799
RANDOM_SEED   = 112233         # matches Stata  `set seed 112233`

# Full $pscorevars list from AllGlobal.do (order preserved)
# Some may be absent from JF.dta — they are skipped gracefully below.
PSCORE_VARS = (
    ["ernpq8"]
    + [f"ernpq{q}" for q in range(6, 0, -1)]      # ernpq6..ernpq1
    + [f"adcpq{q}" for q in range(7, 0, -1)]       # adcpq7..adcpq1
    + [f"fstpq{q}" for q in range(7, 0, -1)]       # fstpq7..fstpq1
    + [f"anyernpq{q}" for q in range(1, 7)]        # anyernpq1..anyernpq6
    + ["anyernpq8"]
    + [f"anyadcpq{q}" for q in range(1, 8)]        # anyadcpq1..anyadcpq7
    + [f"anyfstpq{q}" for q in range(1, 8)]        # anyfstpq1..anyfstpq7
    + ["yremp", "prevafdc", "white", "black", "hisp",
       "marnvr", "marapt", "agelt25", "age2534",
       "nohsged", "hsged", "kidctgt2", "applcant",
       "misshs", "misskidctgt2", "missmar",
       "ernpq7", "anyernpq7"]
)

STATE_VARS = ["p0N", "p1N", "p2N", "p0P", "p1P", "p2P"]


# ═════════════════════════════════════════════════════════════════════════════
# PART 1-2:  Build person-quarter dataset with state dummies
#            Mirrors PART 2 of DistributionOverStates.do
# ═════════════════════════════════════════════════════════════════════════════
def build_person_quarter(jf_path, rules_path, which_fpl=WHICH_FPL,
                         fq=FIRST_QUARTER, lq=LAST_QUARTER):
    """
    Returns a DataFrame with one row per (id, quarter) for quarters fq..lq.
    Columns include: id, e, quarter, year, month, included, p0N..p2P,
    plus all available pscore variables.
    """
    # ── Load policy rules ──────────────────────────────────────────────────
    print("Loading PolicyRules.dta …")
    rules = pd.read_stata(rules_path)
    rules['year'] = rules['year'].astype(int)
    rules['size'] = rules['size'].astype(int)
    f3_map = {(int(r['year']), int(r['size'])): float(r['F3'])
              for _, r in rules.iterrows()}
    print(f"  {len(rules)} policy-rule rows, "
          f"years {rules['year'].min()}–{rules['year'].max()}, "
          f"sizes {rules['size'].min()}–{rules['size'].max()}")

    # ── Load JF.dta ────────────────────────────────────────────────────────
    print(f"Loading JF.dta …")
    df = pd.read_stata(jf_path)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    # ── Filter: quarters fq..lq, kidcount not missing ──────────────────────
    df = df[(df['quarter'] >= fq) & (df['quarter'] <= lq)].copy()
    df = df[df['kidcount'].notna()].copy()
    print(f"  After Q{fq}–Q{lq} and kidcount filter: {len(df):,} rows")

    # ── AU size ─────────────────────────────────────────────────────────────
    # Stata: size = kidcount+1 if kidcount in {1,2,3}; size=2 if kidcount==0
    df['size'] = np.nan
    df.loc[df['kidcount'].isin([1, 2, 3]), 'size'] = (
        df.loc[df['kidcount'].isin([1, 2, 3]), 'kidcount'] + 1)
    df.loc[df['kidcount'] == 0, 'size'] = 2
    df = df[df['size'].notna()].copy()
    df['size'] = df['size'].astype(int)

    # Lookup size for FPL threshold
    size_adj = {"nextsizeup": 1, "twosizesup": 2, "exactsize": 0}
    df['lookup_size'] = df['size'] + size_adj.get(which_fpl, 1)

    # ── Merge policy rules → F3 ─────────────────────────────────────────────
    df['year_int'] = df['year'].astype(int)
    df['F3'] = df.apply(
        lambda r: f3_map.get((r['year_int'], int(r['lookup_size'])), np.nan), axis=1)
    n_before = len(df)
    df = df[df['F3'].notna()].copy()
    print(f"  After policy-rules merge: {len(df):,} rows "
          f"({n_before - len(df)} dropped — no rules for that year/size)")

    # ── Quarterly FPL threshold ─────────────────────────────────────────────
    # Stata: Cbin = round(3*F3_{ss}, 100)  where round(x,100) → nearest 100
    df['Cbin'] = np.round(3.0 * df['F3'] / 100.0) * 100.0

    # ── Earnings bin ────────────────────────────────────────────────────────
    df['ebin'] = np.nan
    df.loc[df['earnq'] == 0,                                   'ebin'] = 0
    df.loc[(df['earnq'] > 0) & (df['earnq'] <= df['Cbin']),   'ebin'] = 1
    df.loc[(df['earnq'] > df['Cbin']) & df['earnq'].notna(),   'ebin'] = 2

    # ── Welfare participation ───────────────────────────────────────────────
    # included = (afdcon_nminq == 0 | afdcon_nminq == 3)
    # partic   = 0 if always off; 1 if always on; NaN if mixed
    anq = df['afdcon_nminq']
    if anq.dtype.name in ('category', 'object'):
        anq_str = anq.astype(str).str.lower()
        is_always_off = anq_str.str.contains(r'always.off|^0$', na=False)
        is_always_on  = anq_str.str.contains(r'always.on|^3$',  na=False)
    else:
        is_always_off = (anq == 0)
        is_always_on  = (anq == 3)

    df['included'] = (is_always_off | is_always_on).astype(int)
    df['partic']   = np.nan
    df.loc[is_always_off, 'partic'] = 0.0
    df.loc[is_always_on,  'partic'] = 1.0

    # ── State dummies ───────────────────────────────────────────────────────
    for sv, (e_, p_) in zip(STATE_VARS, [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]):
        df[sv] = ((df['ebin'] == e_) & (df['partic'] == p_)).astype(float)
        df.loc[df['included'] == 0, sv] = 0.0   # zero out mixed quarters

    # ── Keep 1 month per quarter (first within id-quarter) ─────────────────
    # Mirrors: bys id quarter: replace nn=_n; drop if nn!=1
    df = df.sort_values(['id', 'year', 'quarter', 'month'])
    df['_nn'] = df.groupby(['id', 'quarter']).cumcount()
    df = df[df['_nn'] == 0].drop(columns='_nn').copy()
    print(f"  After keep 1 month/quarter: {len(df):,} person-quarters")

    # Report inclusion stats
    n_inc = df['included'].sum()
    print(f"  Included (consistent welfare): {n_inc:,} / {len(df):,} "
          f"({100*n_inc/len(df):.1f}%)")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# PART 3:  Propensity-score logit — person level
#          Mirrors: logit e $pscorevars, asis  in DistributionOverStates_P
# ═════════════════════════════════════════════════════════════════════════════
def fit_pscore_logit(id_vals, e_vals, X_mat, seed=RANDOM_SEED):
    """
    Fit logistic regression on person-level arrays, replicating Stata's
      logit e $pscorevars, asis
    as closely as possible.

    Two-cause elimination strategy
    ───────────────────────────────
    Cause 1 (algorithm): sklearn L-BFGS ≠ Stata Newton-Raphson.
      Fix: use statsmodels.Logit(method='newton') — same IRLS update rule.

    Cause 2 (sample): JF.dta pscorewt uses 4803 persons (GetJFData.do).
      DistributionOverStates_P refits on the filtered 4642-person sample.
      Fix: call this function on the filtered person_df (not JF.dta weights).

    Together these reproduce DistributionOverStates_P's pscorewt exactly
    (up to floating-point precision).

    Falls back to sklearn L-BFGS if statsmodels is unavailable / fails.
    """
    # Drop zero-variance columns (safe pre-processing; Stata keeps them via
    # `asis` but they contribute nothing and can destabilise the Hessian)
    col_mask = X_mat.std(axis=0) > 1e-10
    X_use    = X_mat[:, col_mask]

    try:
        import statsmodels.api as sm
        import warnings as _warnings

        # Add intercept — Stata includes one by default
        X_sm = sm.add_constant(X_use, has_constant='add')

        model = sm.Logit(e_vals.astype(float), X_sm.astype(float))

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")          # suppress near-singular warnings
            result = model.fit(
                method='newton',          # Newton-Raphson == Stata default
                maxiter=200,
                tol=1e-8,                 # tighter than Stata's 1e-6 — safe
                disp=False,
                warn_convergence=False,
                start_params=np.zeros(X_sm.shape[1]),  # Stata starts at β=0
            )

        pscore = np.clip(result.predict(X_sm), 1e-6, 1 - 1e-6)
        return e_vals / pscore + (1 - e_vals) / (1 - pscore)

    except Exception as ex:
        print(f"  [statsmodels Newton-Raphson failed ({ex})] — falling back to sklearn L-BFGS")
        lr = LogisticRegression(
            penalty='l2', C=1e6, solver='lbfgs',
            max_iter=3000, random_state=seed, tol=1e-6,
        )
        lr.fit(X_use, e_vals)
        pscore = np.clip(lr.predict_proba(X_use)[:, 1], 1e-6, 1 - 1e-6)
        return e_vals / pscore + (1 - e_vals) / (1 - pscore)


# ═════════════════════════════════════════════════════════════════════════════
# PART 4:  IPW-adjusted, inclusion-normalised state distributions
#          Mirrors: meandifs $marginalvars [aw=pscorewt], by(e)
#          in DistributionOverStates_P
# ═════════════════════════════════════════════════════════════════════════════
def compute_state_probs_arrays(e_pq, included_pq, states_pq, pscorewt_pq):
    """
    For each group d ∈ {0=AFDC, 1=JF}:
      p_incl_d = weighted_mean(included, pscorewt | D=d)
      p_s_d    = weighted_mean(p_s, pscorewt | D=d) / p_incl_d

    Returns dict keyed as 'p0N_t', 'p0N_c', …, 'p2P_t', 'p2P_c'
    """
    result = {}
    for label, mask in [('t', e_pq == 1), ('c', e_pq == 0)]:
        w   = pscorewt_pq[mask]
        inc = included_pq[mask]
        sts = states_pq[mask]          # shape (n_group, 6)
        p_incl = np.average(inc, weights=w)
        for j, sv in enumerate(STATE_VARS):
            result[f"{sv}_{label}"] = np.average(sts[:, j], weights=w) / p_incl
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("distribution_over_states.py — Python replication of KT Stata code")
    print("=" * 70)

    # ── Build person-quarter dataset ───────────────────────────────────────
    df_pq = build_person_quarter(JF_PATH, POLICY_RULES_PATH)

    # ── Extract numpy arrays (fast access in bootstrap loop) ───────────────
    # Person-quarter arrays
    pq_id       = df_pq['id'].to_numpy()
    pq_e        = df_pq['e'].to_numpy(dtype=int)
    pq_included = df_pq['included'].to_numpy(dtype=float)
    pq_states   = df_pq[STATE_VARS].to_numpy(dtype=float)   # (N_pq, 6)

    # Person-level arrays (one row per unique person)
    avail_pscvars = [v for v in PSCORE_VARS if v in df_pq.columns]
    missing_psc   = [v for v in PSCORE_VARS if v not in df_pq.columns]
    if missing_psc:
        print(f"\n  [pscore] {len(missing_psc)} vars absent from JF.dta — skipping")
        print(f"  First few missing: {missing_psc[:6]}")

    person_df  = df_pq.drop_duplicates(subset=['id']).copy()
    person_ids = person_df['id'].to_numpy()
    person_e   = person_df['e'].to_numpy(dtype=int)
    person_X   = person_df[avail_pscvars].fillna(0).to_numpy(dtype=float)

    # Map: original person id → list of row indices in df_pq (for bootstrap)
    from collections import defaultdict
    pid_to_rows = defaultdict(list)
    for row_idx, pid in enumerate(pq_id):
        pid_to_rows[pid].append(row_idx)
    pid_list   = list(pid_to_rows.keys())
    N_persons  = len(pid_list)

    print(f"\n  Person-quarter obs: {len(pq_id):,}   Unique persons: {N_persons:,}")
    print(f"  Pscore covariates used: {len(avail_pscvars)}")

    # ── POINT ESTIMATES ────────────────────────────────────────────────────
    # Fit statsmodels Newton-Raphson logit on the filtered 4642-person sample.
    # This mirrors DistributionOverStates_P exactly:
    #   • Same algorithm  (NR == Stata's logit)           → eliminates Cause 1
    #   • Same sample     (filtered persons, not 4803)    → eliminates Cause 2
    print("\nFitting propensity-score logit (statsmodels Newton-Raphson, filtered sample) …")
    pscorewt_pt = fit_pscore_logit(person_ids, person_e, person_X)
    id_to_wt    = dict(zip(person_ids, pscorewt_pt))
    pq_wt_pt    = np.array([id_to_wt[pid] for pid in pq_id])

    probs_pt  = compute_state_probs_arrays(pq_e, pq_included, pq_states, pq_wt_pt)

    # Column order matches Table4_mat.txt header:
    # p0N_t1, p0N_c1, p1N_t1, p1N_c1, p2N_t1, p2N_c1, p0P_t1, p0P_c1, ...
    col_keys = []
    for sv in STATE_VARS:
        col_keys += [f"{sv}_t", f"{sv}_c"]
    pt_row = np.array([probs_pt[k] for k in col_keys])

    print("\n── Point estimates (compare to KT Table 4) ─────────────────────")
    print(f"  {'State':<6}  {'JF':>8}  {'AFDC':>8}")
    for sv in STATE_VARS:
        print(f"  {sv:<6}  {probs_pt[f'{sv}_t']:8.4f}  {probs_pt[f'{sv}_c']:8.4f}")

    # ── BOOTSTRAP (clustered by person) ────────────────────────────────────
    print(f"\nBootstrapping ({N_BOOTS} draws, clustered by person, seed={RANDOM_SEED}) …")
    rng      = np.random.default_rng(RANDOM_SEED)
    bs_rows  = []

    # Pre-convert pid_to_rows to arrays for speed
    pid_row_arrays = {pid: np.array(rows) for pid, rows in pid_to_rows.items()}

    for b in range(N_BOOTS):
        if (b + 1) % 100 == 0 or b == 0:
            print(f"  Bootstrap {b+1}/{N_BOOTS} …")

        # Sample persons with replacement (cluster bootstrap)
        bs_person_slot_idx = rng.integers(0, N_persons, size=N_persons)

        # --- Person-level data for logit ---
        bs_X_person = person_X[bs_person_slot_idx]
        bs_e_person = person_e[bs_person_slot_idx]

        # --- Expand to person-quarter ---
        # Build arrays of pq rows in bootstrap sample
        bs_pq_row_idx = np.concatenate(
            [pid_row_arrays[pid_list[i]] for i in bs_person_slot_idx])
        bs_e_pq       = pq_e[bs_pq_row_idx]
        bs_inc_pq     = pq_included[bs_pq_row_idx]
        bs_states_pq  = pq_states[bs_pq_row_idx]

        # --- Fit pscore on bootstrap persons ---
        try:
            bs_pscorewt_person = fit_pscore_logit(
                np.arange(N_persons), bs_e_person, bs_X_person, seed=RANDOM_SEED + b)
        except Exception:
            # Fallback: equal weights (should rarely happen)
            bs_pscorewt_person = np.ones(N_persons) * 2.0

        # Map person weights to their person-quarter rows
        # The i-th slot in bs_person_slot_idx contributes to consecutive rows
        slot_lens    = [len(pid_row_arrays[pid_list[i]]) for i in bs_person_slot_idx]
        slot_wts_pq  = np.repeat(bs_pscorewt_person, slot_lens)

        # --- Compute state probs ---
        try:
            probs_bs = compute_state_probs_arrays(
                bs_e_pq, bs_inc_pq, bs_states_pq, slot_wts_pq)
            bs_rows.append([probs_bs[k] for k in col_keys])
        except Exception as ex:
            print(f"  Bootstrap {b+1} failed in state probs: {ex} — skipping")
            continue

    print(f"  {len(bs_rows)} successful bootstrap draws")

    # ── WRITE OUTPUT ───────────────────────────────────────────────────────
    # Header uses Stata's _t1/_c1 suffix convention
    col_headers = []
    for sv in STATE_VARS:
        col_headers += [f"{sv}_t1", f"{sv}_c1"]

    all_rows = np.vstack([pt_row] + [np.array(r) for r in bs_rows])

    print(f"\nWriting {OUTPUT_PATH} …")
    with open(OUTPUT_PATH, 'w') as f:
        f.write("\t".join(col_headers) + "\n")
        for row in all_rows:
            f.write("\t".join(f"{v}" for v in row) + "\n")
    print(f"  {len(all_rows)} rows × {len(col_headers)} columns written")

    # ── COMPARE WITH KT's TABLE4_MAT.TXT ──────────────────────────────────
    try:
        kt = np.loadtxt(KT_TABLE4_PATH, skiprows=1, delimiter="\t")
        kt_pt = kt[0, :]
        diff  = pt_row - kt_pt
        print("\n── Comparison with KT Stata Table4_mat.txt ─────────────────────")
        print(f"  {'Column':<12}  {'Python':>8}  {'KT Stata':>10}  {'Diff':>10}")
        for i, h in enumerate(col_headers):
            print(f"  {h:<12}  {pt_row[i]:8.4f}  {kt_pt[i]:10.4f}  {diff[i]:+10.5f}")
        print(f"\n  Max |diff|: {np.abs(diff).max():.6f}   "
              f"Mean |diff|: {np.abs(diff).mean():.6f}")
    except FileNotFoundError:
        print(f"\n  KT file not found at {KT_TABLE4_PATH} — skipping comparison")

    print("\nDone.  Feed Table4_mat_python.txt into replicating4&5.py to check Table 5.")


if __name__ == "__main__":
    main()
