"""
CLP_final_group.py
==================
Final CLP estimator: GroupKFold cross-fitting (grouped by person).
All quarters of the same person are held out together, preventing
data leakage across person-quarters.

Companion to CLP_final.py (same logic, GroupKFold instead of KFold).

Estimators
──────────
  LASSO  — LassoCV   (L1 penalty, sparse)
  Ridge  — RidgeCV   (L2 penalty, dense)
  OLS    — LinearRegression (no penalty; valid benchmark for base-28;
            expect high variance with the larger econ feature set)

Feature sets
────────────
  "base"  — 28 original baseline covariates (identical to CLP.py)

  "econ"  — 28 base + 227 economically motivated features = 255 total

             ── Squared terms (18) ──────────────────────────────────────
             Meaningful (10): ernpq1^2..ernpq8^2, kidcount^2, yremp^2
             Near-binary (8): adcpq1^2..adcpq8^2
                AFDC quarters may be binary in practice; kept because some
                participants have fractional welfare months.  Lasso will
                zero these if they add no information beyond the linear term.

             ── Tier 1 interactions: always keep (152) ──────────────────
             T1a Earnings×Earnings:  ernpqk × ernpql, k<l        (28)
             T1b AFDC×AFDC:         adcpqk × adcpql, k<l        (28)
             T1c Earnings×AFDC (same quarter): ernpqk × adcpqk   (8)
             T1d Earnings×AFDC (cross-quarter): ernpqk × adcpql, k!=l (56)
             T1e kidcount × ernpqk                                (8)
             T1f kidcount × adcpqk                                (8)
             T1g yngchtru × ernpqk                                (8)
             T1h applcant × ernpqk                                (8)

             ── Tier 2 interactions: moderate, let Lasso select (57) ───
             T2a age2534   × ernpqk                               (8)
             T2b hsged     × ernpqk                               (8)
             T2c nohsged   × ernpqk                               (8)
             T2d hsged     × adcpqk                               (8)
             T2e nohsged   × adcpqk                               (8)
             T2f yremp     × ernpqk                               (8)
             T2g yremp     × adcpqk                               (8)
             T2h kidcount  × yngchtru                             (1)

             ── Dropped (Tier 3) ────────────────────────────────────────
             binary^2 (binary dummies squared — identical to original)
             Dummy×Dummy mutually exclusive (race, marital status)
             Race×marital status interactions (no structural mechanism)
             Age×race interactions
             ~126 remaining peripheral interactions

Configurations
──────────────
  6 combinations: {base, econ} × {LASSO, Ridge, OLS}

All estimators use K=5-fold GroupKFold cross-fitting (by person).
Bootstrap CIs: multiplier-bootstrap clustered by person (N_BOOTSTRAP draws).

Includes diagnostics module (D1–D4) based on Semenova (2026) reviewer suggestions.

Dependencies:  numpy  pandas  scikit-learn  statsmodels
"""

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import lsq_linear
from sklearn.ensemble import RandomForestRegressor

PSCORE_VARS = (
    ["ernpq8"]
    + [f"ernpq{q}" for q in range(6, 0, -1)]
    + [f"adcpq{q}" for q in range(7, 0, -1)]
    + [f"fstpq{q}" for q in range(7, 0, -1)]
    + [f"anyernpq{q}" for q in range(1, 7)]
    + ["anyernpq8"]
    + [f"anyadcpq{q}" for q in range(1, 8)]
    + [f"anyfstpq{q}" for q in range(1, 8)]
    + ["yremp", "prevafdc", "white", "black", "hisp",
       "marnvr", "marapt", "agelt25", "age2534",
       "nohsged", "hsged", "kidctgt2", "applcant",
       "misshs", "misskidctgt2", "missmar",
       "ernpq7", "anyernpq7"]
)


def fit_pscore_logit(person_ids, e_vals, X_mat, seed=42):
    """
    Fit propensity-score logit on person-level data (statsmodels Newton-Raphson).
    Replicates DistributionOverStates_P: logit e $pscorevars, asis
    Returns pscorewt = e/pscore + (1-e)/(1-pscore) for each person.
    """
    col_mask = X_mat.std(axis=0) > 1e-10
    X_use    = X_mat[:, col_mask]
    try:
        import statsmodels.api as sm
        X_sm  = sm.add_constant(X_use, has_constant='add')
        model = sm.Logit(e_vals.astype(float), X_sm.astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(
                method='newton', maxiter=200, tol=1e-8,
                disp=False, warn_convergence=False,
                start_params=np.zeros(X_sm.shape[1]),
            )
        pscore = np.clip(result.predict(X_sm), 1e-6, 1 - 1e-6)
    except Exception as ex:
        print(f"  [statsmodels NR failed ({ex})] — sklearn L-BFGS fallback")
        lr = LogisticRegression(
            penalty='l2', C=1e6, solver='lbfgs',
            max_iter=3000, random_state=seed, tol=1e-6,
        )
        lr.fit(X_use, e_vals)
        pscore = np.clip(lr.predict_proba(X_use)[:, 1], 1e-6, 1 - 1e-6)
    return e_vals / pscore + (1 - e_vals) / (1 - pscore)


# ─────────────────────────────────────────────────────────────────────────────
# ▶  USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
JF_DTA_PATH       = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/JF.dta")
POLICY_RULES_PATH = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/PolicyRules.dta")
TABLE4_MAT_PATH   = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/Table4_mat.txt")

N_BOOTSTRAP = 200    # raise to >=1000 for publication-quality CIs
K_FOLDS     = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configurations to run: (feature_set, estimator)
CONFIGS_TO_RUN = {
    "base_lasso":      ("base", "LASSO"),
    "base_ridge":      ("base", "Ridge"),
    "base_ols":        ("base", "OLS"),
    "base_postlasso":  ("base", "PostLasso"),
    "econ_lasso":      ("econ", "LASSO"),
    "econ_ridge":      ("econ", "Ridge"),
    "econ_ols":        ("econ", "OLS"),
    "econ_postlasso":  ("econ", "PostLasso"),
    "rf_lasso":        ("rf",   "LASSO"),
    "rf_ridge":        ("rf",   "Ridge"),
    "rf_postlasso":    ("rf",   "PostLasso"),
    "disc_lasso":      ("disc", "LASSO"),
    "disc_ridge":      ("disc", "Ridge"),
    "disc_postlasso":  ("disc", "PostLasso"),
}

# Observable state codes
S_0N, S_1N, S_2N, S_0P, S_1P, S_2P = 0, 1, 2, 3, 4, 5

BETA_NAMES = [
    "beta(0n,1r)", "beta(0r,0n)", "beta(2n,1r)", "beta(0r,2n)", "beta(0r,1r)",
    "beta(0r,1n)", "beta(1n,1r)", "beta(0r,2u)", "beta(2u,1r)",
]
PI_NAMES = [
    "pi(0n->1r)", "pi(0r->0n)", "pi(2n->1r)", "pi(0r->2n)", "pi(0r->1r)",
    "pi(0r->1n)", "pi(1n->1r)", "pi(0r->2u)", "pi(2u->1r)",
]

COV_VARS = [
    "age2534", "black", "hisp", "white",
    "marnvr", "marapt", "hsged", "nohsged", "yngchtru", "kidcount",
    "ernpq1", "ernpq2", "ernpq3", "ernpq4",
    "ernpq5", "ernpq6", "ernpq7", "ernpq8",
    "adcpq1", "adcpq2", "adcpq3", "adcpq4",
    "adcpq5", "adcpq6", "adcpq7", "adcpq8",
    "applcant", "yremp",
]

JF_CONFIG = dict(
    first_quarter  = 1,
    last_quarter   = 7,
    which_fpl      = "nextsizeup",
    covariate_vars = COV_VARS,
)


# =============================================================================
# 1.  DATA LOADING
# =============================================================================
def load_table4_mat(path=TABLE4_MAT_PATH):
    data = np.loadtxt(path, skiprows=1)
    row  = data[0]
    p = dict(
        p00_t=row[0],  p00_c=row[1],
        p10_t=row[2],  p10_c=row[3],
        p20_t=row[4],  p20_c=row[5],
        p01_t=row[6],  p01_c=row[7],
        p11_t=row[8],  p11_c=row[9],
        p21_t=row[10], p21_c=row[11],
    )
    return p, data[1:]


def prepare_jf_data(cfg=None, jf_path=JF_DTA_PATH, rules_path=POLICY_RULES_PATH):
    """
    Replicate KT Stata pipeline at individual level.
    Returns D, state_obs, df_incl, person_id, pscorewt.
    (Returns df_incl so engineer_features_econ() can access raw columns.)

    CRITICAL: pscorewt is always recomputed from scratch on the filtered
    sample using fit_pscore_logit + statsmodels Newton-Raphson.
    The pscorewt column in JF.dta (from GetJFData.do) was computed on
    4,803 persons with a different variable set — using it directly gives
    wrong Table 4 frequencies.
    """
    if cfg is None:
        cfg = JF_CONFIG

    fq, lq = cfg['first_quarter'], cfg['last_quarter']

    print(f"\n[JF DATA]  Loading policy rules from {rules_path} ...")
    rules = pd.read_stata(rules_path)[['year', 'size', 'F3']].copy()
    rules['year'] = rules['year'].astype(int)
    rules['size'] = rules['size'].astype(int)
    f3_map = {(r['year'], r['size']): r['F3'] for _, r in rules.iterrows()}
    print(f"  Policy rules: {len(rules)} rows, years {rules['year'].min()}-{rules['year'].max()}, "
          f"sizes {rules['size'].min()}-{rules['size'].max()}")

    print(f"[JF DATA]  Loading {jf_path} ...")
    df = pd.read_stata(jf_path)
    print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")

    df = df[(df['quarter'] >= fq) & (df['quarter'] <= lq)].copy()
    df = df[df['kidcount'].notna()].copy()
    print(f"  After filter Q{fq}-Q{lq} & kidcount not missing: {len(df):,} rows")

    df['size'] = np.nan
    mask_123 = df['kidcount'].isin([1, 2, 3])
    df.loc[mask_123, 'size'] = df.loc[mask_123, 'kidcount'] + 1
    df.loc[df['kidcount'] == 0, 'size'] = 2
    df = df[df['size'].notna()].copy()
    df['size'] = df['size'].astype(int)

    if cfg['which_fpl'] == "nextsizeup":
        df['lookup_size'] = df['size'] + 1
    elif cfg['which_fpl'] == "twosizesup":
        df['lookup_size'] = df['size'] + 2
    else:
        df['lookup_size'] = df['size']

    df['year_int'] = df['year'].astype(int)
    df['F3_nextsizeup'] = df.apply(
        lambda r: f3_map.get((r['year_int'], r['lookup_size']), np.nan), axis=1
    )
    n_before = len(df)
    df = df[df['F3_nextsizeup'].notna()].copy()
    if n_before > len(df):
        print(f"  Dropped {n_before - len(df)} rows with no matching policy rules")

    df['Cbin'] = np.round(3.0 * df['F3_nextsizeup'] / 100.0) * 100.0

    df['ebin'] = np.nan
    df.loc[df['earnq'] == 0, 'ebin'] = 0
    df.loc[(df['earnq'] > 0) & (df['earnq'] <= df['Cbin']), 'ebin'] = 1
    df.loc[(df['earnq'] > df['Cbin']) & df['earnq'].notna(), 'ebin'] = 2

    anq = df['afdcon_nminq']
    if anq.dtype.name == 'category' or anq.dtype == object:
        anq_str = anq.astype(str).str.lower()
        is_always_off = anq_str.str.contains('always off|always_off') | (anq_str == '0')
        is_always_on  = anq_str.str.contains('always on|always_on')  | (anq_str == '3')
    else:
        is_always_off = (anq == 0)
        is_always_on  = (anq == 3)

    df['included'] = (is_always_off | is_always_on).astype(int)
    df['partic'] = np.nan
    df.loc[is_always_on,  'partic'] = 1.0
    df.loc[is_always_off, 'partic'] = 0.0

    df['p0N'] = ((df['ebin'] == 0) & (df['partic'] == 0)).astype(int)
    df['p1N'] = ((df['ebin'] == 1) & (df['partic'] == 0)).astype(int)
    df['p2N'] = ((df['ebin'] == 2) & (df['partic'] == 0)).astype(int)
    df['p0P'] = ((df['ebin'] == 0) & (df['partic'] == 1)).astype(int)
    df['p1P'] = ((df['ebin'] == 1) & (df['partic'] == 1)).astype(int)
    df['p2P'] = ((df['ebin'] == 2) & (df['partic'] == 1)).astype(int)
    for v in ['p0N', 'p1N', 'p2N', 'p0P', 'p1P', 'p2P']:
        df.loc[df['included'] == 0, v] = 0

    df = df.sort_values(['id', 'year', 'quarter', 'month'])
    df['nn'] = df.groupby(['id', 'quarter']).cumcount() + 1
    df = df[df['nn'] == 1].copy()
    print(f"  After keeping 1 month/quarter: {len(df):,} person-quarters")

    n_incl = df['included'].sum()
    n_excl = (df['included'] == 0).sum()
    print(f"  Included (consistent welfare): {n_incl:,}  "
          f"Excluded (mixed): {n_excl:,}  P(included)={n_incl/len(df):.3f}")

    # ── Propensity-score logit — recomputed on this filtered sample (KT method) ──
    # CRITICAL: always recompute from scratch on the filtered 4,642-person sample
    # using the full 59-variable PSCORE_VARS + statsmodels Newton-Raphson.
    # The pscorewt column in JF.dta (from GetJFData.do) was computed on 4,803 persons
    # with a different variable set — using it directly gives wrong Table 4 frequencies.
    avail_psc = [v for v in PSCORE_VARS if v in df.columns]
    missing_psc = [v for v in PSCORE_VARS if v not in df.columns]
    if missing_psc:
        print(f"  [pscore] {len(missing_psc)} vars absent from JF.dta — "
              f"skipping: {missing_psc[:5]}{'...' if len(missing_psc) > 5 else ''}")
    person_df  = df.drop_duplicates(subset=['id']).copy()
    person_ids = person_df['id'].to_numpy()
    person_e   = person_df['e'].to_numpy(dtype=int)
    person_X   = person_df[avail_psc].fillna(0).to_numpy(dtype=float)
    N_persons  = len(person_ids)
    print(f"  Fitting pscore logit on {N_persons:,} filtered persons "
          f"({len(avail_psc)} vars) — statsmodels Newton-Raphson ...")
    pscorewt_person = fit_pscore_logit(person_ids, person_e, person_X)
    id_to_wt = dict(zip(person_ids, pscorewt_person))
    df['pscorewt'] = df['id'].map(id_to_wt)
    print(f"  pscorewt range: [{pscorewt_person.min():.4f}, {pscorewt_person.max():.4f}]")

    # ── Keep included person-quarters ──
    df_incl = df[df['included'] == 1].copy()

    df_incl['state_obs'] = np.nan
    df_incl.loc[df_incl['p0N'] == 1, 'state_obs'] = S_0N
    df_incl.loc[df_incl['p1N'] == 1, 'state_obs'] = S_1N
    df_incl.loc[df_incl['p2N'] == 1, 'state_obs'] = S_2N
    df_incl.loc[df_incl['p0P'] == 1, 'state_obs'] = S_0P
    df_incl.loc[df_incl['p1P'] == 1, 'state_obs'] = S_1P
    df_incl.loc[df_incl['p2P'] == 1, 'state_obs'] = S_2P
    assert df_incl['state_obs'].notna().all(), \
        "Some included obs have no state — check ebin/partic classification."

    D         = df_incl['e'].to_numpy(dtype=int)
    state_obs_arr = df_incl['state_obs'].to_numpy(dtype=int)
    person_id = df_incl['id'].to_numpy()
    pscorewt  = df_incl['pscorewt'].to_numpy(dtype=float)

    n_persons = len(np.unique(person_id))
    print(f"\n  Final dataset: {len(D):,} person-quarters from {n_persons:,} persons")
    print(f"  JF (treated): {(D==1).sum():,}   AFDC (control): {(D==0).sum():,}")
    sc = [(state_obs_arr == s).sum() for s in range(6)]
    print(f"  States: 0n={sc[0]}  1n={sc[1]}  2n={sc[2]}  "
          f"0p={sc[3]}  1p={sc[4]}  2p={sc[5]}")

    print("\n  ── IPW-adjusted state frequencies (conditional on included) ──")
    state_lbls6 = ['0n', '1n', '2n', '0p', '1p', '2p']
    for label, gmask in [("JF", D == 1), ("AFDC", D == 0)]:
        w    = pscorewt[gmask]
        s    = state_obs_arr[gmask]
        wsum = w.sum() if w.sum() > 0 else 1.0
        freqs = [
            f"{sl}={np.sum((s == j).astype(float) * w) / wsum:.4f}"
            for j, sl in enumerate(state_lbls6)
        ]
        print(f"    {label:5s} (N={gmask.sum():,}):  " + "  ".join(freqs))
    print()

    print(f"    Treatment: D=1:{D.sum():,}  D=0:{(D==0).sum():,}")
    return D, state_obs_arr, df_incl, person_id, pscorewt


# =============================================================================
# 2.  FEATURE ENGINEERING  — economically motivated set
# =============================================================================
def engineer_features_econ(df_incl, cov_vars=COV_VARS):
    """
    Build the raw (unscaled) "econ" feature matrix.

    Returns
    -------
    X_base_raw : (N, 28) ndarray  — original 28 covariates, unscaled
    X_econ_raw : (N, M)  ndarray  — economically motivated extra features
    feat_names : list[str]        — names for all 28+M features
    group_info : dict             — feature group labels and column slices

    Scaling is done INSIDE each CV fold in _build_fold_features() to avoid
    data leakage.
    """
    df = df_incl.reset_index(drop=True)
    avail = [v for v in cov_vars if v in df.columns]
    X_base_raw = df[avail].fillna(0).to_numpy(float)   # (N, 28)

    def _col(name):
        return df[name].fillna(0).to_numpy(float) if name in df.columns \
               else np.zeros(len(df))

    ern_cols = [c for c in [f'ernpq{k}' for k in range(1, 9)] if c in df.columns]
    adc_cols = [c for c in [f'adcpq{k}' for k in range(1, 9)] if c in df.columns]
    n_q      = len(ern_cols)   # <= 8

    ern = df[ern_cols].fillna(0).to_numpy(float)   # (N, n_q)
    adc = df[adc_cols].fillna(0).to_numpy(float)   # (N, n_q)

    kidcount = _col('kidcount')
    yngchtru = _col('yngchtru')
    applcant = _col('applcant')
    age2534  = _col('age2534')
    hsged    = _col('hsged')
    nohsged  = _col('nohsged')
    yremp    = _col('yremp')

    blocks = []
    names  = []
    group_info = {}   # group_label -> column slice in X_econ_raw

    def _add(block_arr, block_names, label):
        """Append a block of features and record its column range."""
        start = sum(b.shape[1] for b in blocks)
        blocks.append(block_arr)
        names.extend(block_names)
        end = start + block_arr.shape[1]
        group_info[label] = slice(start, end)

    # ── Squared terms: meaningful continuous variables ────────────────────────
    ern_sq   = ern ** 2
    _add(ern_sq,
         [f'ernpq{k+1}^2' for k in range(n_q)],
         "sq_ern (meaningful)")
    _add(kidcount[:, None] ** 2,       ["kidcount^2"],      "sq_kidcount")
    _add(yremp[:, None]    ** 2,       ["yremp^2"],         "sq_yremp")

    # ── Squared terms: near-binary AFDC — drop column if truly binary ────────
    # If adcpqk takes only values in {0, 1} then adcpqk^2 = adcpqk (perfect
    # collinearity with the base feature).  We check each column individually
    # and only keep those with > 2 unique values (genuinely continuous).
    adc_sq_cols  = []
    adc_sq_names = []
    n_dropped_adc = 0
    for k in range(n_q):
        col = adc[:, k]
        n_uniq = len(np.unique(col))
        if n_uniq <= 2:
            # Binary: x^2 = x — drop to avoid perfect collinearity
            n_dropped_adc += 1
        else:
            adc_sq_cols.append(col ** 2)
            adc_sq_names.append(f'adcpq{k+1}^2')
    if adc_sq_cols:
        _add(np.column_stack(adc_sq_cols), adc_sq_names,
             f"sq_adc (kept {len(adc_sq_cols)}/{n_q}, {n_dropped_adc} binary dropped)")
    else:
        print(f"    sq_adc: all {n_q} adcpq columns are binary -> all dropped")

    # ── T1a: Earnings x Earnings  (all unordered pairs, k < l) ───────────────
    ee_block = []
    ee_names = []
    for k, l in combinations(range(n_q), 2):
        ee_block.append(ern[:, k] * ern[:, l])
        ee_names.append(f'ernpq{k+1}xernpq{l+1}')
    _add(np.column_stack(ee_block), ee_names, "T1a: Ern×Ern")

    # ── T1b: AFDC x AFDC  (all unordered pairs, k < l) ──────────────────────
    aa_block = []
    aa_names = []
    for k, l in combinations(range(n_q), 2):
        aa_block.append(adc[:, k] * adc[:, l])
        aa_names.append(f'adcpq{k+1}xadcpq{l+1}')
    _add(np.column_stack(aa_block), aa_names, "T1b: AFDC×AFDC")

    # ── T1c: Earnings x AFDC — same quarter  (k=1..8) ───────────────────────
    ea_same_block = []
    ea_same_names = []
    for k in range(n_q):
        ea_same_block.append(ern[:, k] * adc[:, k])
        ea_same_names.append(f'ernpq{k+1}xadcpq{k+1}')
    _add(np.column_stack(ea_same_block), ea_same_names, "T1c: Ern×AFDC same-q")

    # ── T1d: Earnings x AFDC — cross-quarter  (all k!=l ordered pairs) ────────
    ea_cross_block = []
    ea_cross_names = []
    for k in range(n_q):
        for l in range(n_q):
            if k != l:
                ea_cross_block.append(ern[:, k] * adc[:, l])
                ea_cross_names.append(f'ernpq{k+1}xadcpq{l+1}')
    _add(np.column_stack(ea_cross_block), ea_cross_names, "T1d: Ern×AFDC cross-q")

    # ── T1e: kidcount x earnpqk  (k=1..8) ───────────────────────────────────
    _add(ern * kidcount[:, None],
         [f'kidcountxernpq{k+1}' for k in range(n_q)],
         "T1e: kidcount×Ern")

    # ── T1f: kidcount x adcpqk  (k=1..8) ────────────────────────────────────
    _add(adc * kidcount[:, None],
         [f'kidcountxadcpq{k+1}' for k in range(n_q)],
         "T1f: kidcount×AFDC")

    # ── T1g: yngchtru x earnpqk  (k=1..8) ───────────────────────────────────
    _add(ern * yngchtru[:, None],
         [f'yngchtruxernpq{k+1}' for k in range(n_q)],
         "T1g: yngchtru×Ern")

    # ── T1h: applcant x earnpqk  (k=1..8) ───────────────────────────────────
    _add(ern * applcant[:, None],
         [f'applcantxernpq{k+1}' for k in range(n_q)],
         "T1h: applcant×Ern")

    # ── T2a: age2534 x earnpqk  (k=1..8) ────────────────────────────────────
    _add(ern * age2534[:, None],
         [f'age2534xernpq{k+1}' for k in range(n_q)],
         "T2a: age2534×Ern")

    # ── T2b: hsged x earnpqk  (k=1..8) ──────────────────────────────────────
    _add(ern * hsged[:, None],
         [f'hsgedxernpq{k+1}' for k in range(n_q)],
         "T2b: hsged×Ern")

    # ── T2c: nohsged x earnpqk  (k=1..8) ────────────────────────────────────
    _add(ern * nohsged[:, None],
         [f'nohsgedxernpq{k+1}' for k in range(n_q)],
         "T2c: nohsged×Ern")

    # ── T2d: hsged x adcpqk  (k=1..8) ───────────────────────────────────────
    _add(adc * hsged[:, None],
         [f'hsgedxadcpq{k+1}' for k in range(n_q)],
         "T2d: hsged×AFDC")

    # ── T2e: nohsged x adcpqk  (k=1..8) ─────────────────────────────────────
    _add(adc * nohsged[:, None],
         [f'nohsgedxadcpq{k+1}' for k in range(n_q)],
         "T2e: nohsged×AFDC")

    # ── T2f: yremp x earnpqk  (k=1..8) ──────────────────────────────────────
    _add(ern * yremp[:, None],
         [f'yrempxernpq{k+1}' for k in range(n_q)],
         "T2f: yremp×Ern")

    # ── T2g: yremp x adcpqk  (k=1..8) ───────────────────────────────────────
    _add(adc * yremp[:, None],
         [f'yrempxadcpq{k+1}' for k in range(n_q)],
         "T2g: yremp×AFDC")

    # ── T2h: kidcount x yngchtru  (1 term) ───────────────────────────────────
    _add((kidcount * yngchtru)[:, None],
         ["kidcountxyngchtru"],
         "T2h: kidcount×yngchtru")

    X_econ_raw = np.column_stack(blocks)   # (N, M)
    feat_names = avail + names

    n_sq_meaningful = n_q + 2                  # ernpq^2 x n_q + kidcount^2 + yremp^2
    n_sq_adc_kept   = len(adc_sq_cols)         # adcpq^2 columns that passed binary check
    n_t1   = (len(ee_block) + len(aa_block) + len(ea_same_block)
              + len(ea_cross_block) + 4 * n_q)  # T1a..T1h
    n_t2   = 7 * n_q + 1                        # T2a..T2h

    print(f"    Econ feature breakdown:")
    print(f"      Base:                       {X_base_raw.shape[1]:>4}")
    print(f"      Squared (meaningful):       {n_sq_meaningful:>4}  "
          f"(ernpq^2 x{n_q}, kidcount^2, yremp^2)")
    print(f"      Squared adcpq^2 kept:       {n_sq_adc_kept:>4}  "
          f"({n_dropped_adc} dropped — binary, x^2=x)")
    print(f"      Binary dummy^2 (age/race/etc): dropped entirely (10 terms, x^2=x)")
    print(f"      Tier 1 interactions:        {n_t1:>4}")
    print(f"        T1a Ern×Ern:              {len(ee_block):>4}")
    print(f"        T1b AFDC×AFDC:            {len(aa_block):>4}")
    print(f"        T1c Ern×AFDC same-q:      {len(ea_same_block):>4}")
    print(f"        T1d Ern×AFDC cross-q:     {len(ea_cross_block):>4}")
    print(f"        T1e-h scalar×Ern/AFDC:   {4*n_q:>4}")
    print(f"      Tier 2 interactions:        {n_t2:>4}")
    print(f"      Tier 3 collinear (excl.×excl.): dropped entirely")
    print(f"      Extra total:                {X_econ_raw.shape[1]:>4}")
    print(f"      Grand total (base+econ):    {X_base_raw.shape[1] + X_econ_raw.shape[1]:>4}")

    return X_base_raw, X_econ_raw, feat_names, group_info


# =============================================================================
# 2b.  FEATURE ENGINEERING  — RF importance-based interactions
# =============================================================================
def engineer_features_rf(df_incl, D, state_obs, pscorewt,
                          cov_vars=COV_VARS, n_top=10, seed=42):
    """
    Data-driven feature construction via random forest importance.

    Steps:
    1. Fit a random forest for each B-component on the 28 base covariates.
    2. Average feature importances across all B-components.
    3. Select top n_top features.
    4. Construct all C(n_top, 2) pairwise interactions among top features.

    The LP dual vertex is piecewise-constant in b_0(x); random-forest importance
    identifies which covariates actually drive vertex selection.
    """
    B_obs = compute_B(D, state_obs, pscorewt=pscorewt)
    k     = B_obs.shape[1]

    df    = df_incl.reset_index(drop=True)
    avail = [v for v in cov_vars if v in df.columns]
    X_base_raw = df[avail].fillna(0).to_numpy(float)

    avg_imp = np.zeros(len(avail))
    for j in range(k):
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=seed, n_jobs=-1,
        )
        rf.fit(X_base_raw, B_obs[:, j])
        avg_imp += rf.feature_importances_
    avg_imp /= k

    top_idx   = np.argsort(avg_imp)[::-1][:n_top]
    top_names = [avail[i] for i in top_idx]

    print(f"  [RF] Top {n_top} features (avg importance across {k} B-components):")
    for rank, (name, imp) in enumerate(zip(top_names, avg_imp[top_idx]), 1):
        print(f"    {rank:2d}. {name:<22s}  imp={imp:.5f}")

    X_top  = X_base_raw[:, top_idx]
    blocks = []
    inames = []
    for i, j in combinations(range(n_top), 2):
        blocks.append(X_top[:, i] * X_top[:, j])
        inames.append(f'{top_names[i]}x{top_names[j]}')

    X_rf_extra = np.column_stack(blocks) if blocks else np.zeros((len(df), 0))
    print(f"  [RF] Generated {len(blocks)} interactions (C({n_top},2) from top-{n_top})")
    return X_rf_extra


# =============================================================================
# 2c.  FEATURE ENGINEERING  — discretized continuous covariates
# =============================================================================
def engineer_features_disc(df_incl, cov_vars=COV_VARS):
    """
    Discretize continuous covariates (earnings, kidcount, yremp) into quartile
    bin dummies.

    Motivation: the LP dual vertex is piecewise-constant in b_0(x), which is
    piecewise-smooth in x.  Linear features poorly approximate this structure;
    quartile bin dummies match it better.
    """
    df    = df_incl.reset_index(drop=True)
    avail = [v for v in cov_vars if v in df.columns]
    X_base_raw = df[avail].fillna(0).to_numpy(float)

    continuous_targets = (
        [f'ernpq{k}' for k in range(1, 9)] + ['kidcount', 'yremp']
    )

    blocks = []
    dnames = []
    for vname in continuous_targets:
        if vname not in avail:
            continue
        col = X_base_raw[:, avail.index(vname)]
        if col.std() < 1e-10:
            continue
        nz = col[col > 0]
        qs = np.nanpercentile(nz, [25, 50, 75]) if len(nz) >= 20 \
             else np.percentile(col, [25, 50, 75])
        for q_idx, thresh in enumerate(qs, 2):
            dummy = (col > thresh).astype(float)
            blocks.append(dummy)
            dnames.append(f'{vname}_q{q_idx}plus')

    X_disc_extra = np.column_stack(blocks) if blocks else np.zeros((len(df_incl), 0))
    print(f"  [Disc] Generated {len(blocks)} quartile bin dummies "
          f"({len(continuous_targets)} continuous vars × 3 thresholds)")
    return X_disc_extra


# =============================================================================
# 3.  A MATRIX  (pure +-1, base 3-bin model)
# =============================================================================
def build_A():
    #               b0n1r b0r0n b2n1r b0r2n b0r1r b0r1n b1n1r b0r2u b2u1r
    return np.array([
        [ -1,    1,    0,    0,    0,    0,    0,    0,    0],   # 0n
        [  0,    0,    0,    0,    0,    1,   -1,    0,    0],   # 1n
        [  0,    0,   -1,    1,    0,    0,    0,    0,    0],   # 2n
        [  0,   -1,    0,   -1,   -1,   -1,    0,   -1,    0],   # 0p
        [  0,    0,    0,    0,    0,    0,    0,    1,   -1],   # 2p
    ], dtype=float)


# =============================================================================
# 5.  B-SCORES
# =============================================================================
def compute_B(D, state_obs, pscorewt=None):
    if pscorewt is not None:
        w = (2.0 * D - 1.0) * pscorewt
    else:
        rho = float(D.mean())
        w   = D / rho - (1.0 - D) / (1.0 - rho)
    B = np.zeros((len(D), 5))
    for j, s in enumerate([S_0N, S_1N, S_2N, S_0P, S_2P]):
        B[:, j] = (state_obs == s) * w
    return B


# =============================================================================
# 6.  DUAL VERTEX ENUMERATION
# =============================================================================
def enumerate_dual_vertices(A, q):
    """Enumerate all C(9,5)=126 candidate vertices of T_q = {nu: A'nu >= q}."""
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


def binding_vertex(b_hat_i, vertices):
    vals = np.array([v @ b_hat_i for v in vertices])
    return vertices[int(np.argmin(vals))]


# =============================================================================
# 7.  CLP PLUG-IN ESTIMATOR
# =============================================================================
def clp_estimate(q, b_hat, B_obs, A_mat):
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


# =============================================================================
# 8.  MULTIPLIER BOOTSTRAP CI  (clustered by person)
# =============================================================================
def multiplier_bootstrap_ci(contribs, n_bs=N_BOOTSTRAP, alpha=0.05, person_id=None):
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
# 9.  FIRST-STAGE MODEL FACTORY
# =============================================================================
def _make_model(estimator_name, seed):
    if estimator_name == "LASSO":
        return LassoCV(
            alphas=np.logspace(-4, 1, 30), cv=5, eps=1e-4,
            max_iter=3000, random_state=seed, n_jobs=1,
        )
    elif estimator_name == "Ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 40))
    elif estimator_name == "OLS":
        return LinearRegression(n_jobs=1)
    elif estimator_name == "PostLasso":
        # Handled specially in estimate_b0_features — marker only
        return None
    else:
        raise ValueError(f"Unknown estimator: '{estimator_name}'")


# =============================================================================
# 10.  FOLD-LEVEL FEATURE BUILDER
# =============================================================================
def _build_fold_features(feature_set, X_base_raw, X_econ_raw,
                          train_idx, test_idx,
                          X_rf_raw=None, X_disc_raw=None):
    """
    Build standardised feature matrices for one CV fold.

    "base": StandardScaler on the 28 base covariates.
    "econ": StandardScaler on [base | econ_extra] jointly.
    "rf":   StandardScaler on [base | RF-importance interactions].
    "disc": StandardScaler on [base | quartile bin dummies].

    Scaling is fit on the training fold only — no data leakage.
    """
    if feature_set == "base":
        X_full = X_base_raw
    elif feature_set == "econ":
        X_full = np.hstack([X_base_raw, X_econ_raw])
    elif feature_set == "rf":
        if X_rf_raw is not None and X_rf_raw.shape[1] > 0:
            X_full = np.hstack([X_base_raw, X_rf_raw])
        else:
            X_full = X_base_raw
    elif feature_set == "disc":
        if X_disc_raw is not None and X_disc_raw.shape[1] > 0:
            X_full = np.hstack([X_base_raw, X_disc_raw])
        else:
            X_full = X_base_raw
    else:
        raise ValueError(f"Unknown feature_set: '{feature_set}'")

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_full[train_idx])
    X_te = sc.transform(X_full[test_idx])
    return X_tr, X_te


# =============================================================================
# 11.  CROSS-FITTED FIRST STAGE  (GroupKFold by person)
# =============================================================================
def estimate_b0_features(D, state_obs, X_base_raw, X_econ_raw,
                          feature_set, estimator_name,
                          K=5, seed=42, pscorewt=None, person_id=None,
                          X_rf_raw=None, X_disc_raw=None):
    """
    K-fold cross-fitted first stage with a given (feature_set, estimator).

    When person_id is provided, uses GroupKFold so that all quarters of each
    person are held out together (prevents data leakage across person-quarters).
    Falls back to plain KFold when person_id is None.

    PostLasso: Lasso selects variables, then OLS refits on selected set.
    Returns b_hat (N, 5) and B_obs (N, 5).
    """
    B_obs = compute_B(D, state_obs, pscorewt=pscorewt)
    n, k  = B_obs.shape
    b_hat = np.zeros((n, k))

    if person_id is not None:
        splitter = GroupKFold(n_splits=K)
        splits = list(splitter.split(X_base_raw, groups=person_id))
        print(f"  [first-stage] GroupKFold by person ({K} folds, "
              f"{len(np.unique(person_id)):,} persons, {n:,} person-quarters)")
    else:
        splitter = KFold(n_splits=K, shuffle=True, random_state=seed)
        splits = list(splitter.split(X_base_raw))
        print(f"  [first-stage] KFold ({K} folds, {n:,} obs)")

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr, X_te = _build_fold_features(
            feature_set, X_base_raw, X_econ_raw, train_idx, test_idx,
            X_rf_raw=X_rf_raw, X_disc_raw=X_disc_raw,
        )
        for j in range(k):
            y = B_obs[train_idx, j]
            if y.std() < 1e-10:
                b_hat[test_idx, j] = y.mean()
                continue
            if estimator_name == "PostLasso":
                lasso = LassoCV(
                    alphas=np.logspace(-4, 1, 30), cv=5, eps=1e-4,
                    max_iter=3000, random_state=seed, n_jobs=1,
                )
                lasso.fit(X_tr, y)
                sel = np.where(np.abs(lasso.coef_) > 1e-10)[0]
                if len(sel) == 0:
                    sel = np.arange(X_tr.shape[1])
                ols = LinearRegression(n_jobs=1)
                ols.fit(X_tr[:, sel], y)
                b_hat[test_idx, j] = ols.predict(X_te[:, sel])
            else:
                model = _make_model(estimator_name, seed)
                model.fit(X_tr, y)
                b_hat[test_idx, j] = model.predict(X_te)

    return b_hat, B_obs


# =============================================================================
# DIAGNOSTICS MODULE
# Based on: Semenova (2026) CLP paper reviewer suggestions
# =============================================================================

def diag_firstsage_r2(B_obs, b_hat):
    """
    Diagnostic 4: Out-of-fold R^2 per B-component.
    R^2_j = 1 - sum(B_j - b_hat_j)^2 / sum(B_j - B_bar_j)^2
    Red flag: R^2_j ~= 0 or negative for most components combined with
    strong tightening -> suggests over-interpretation.
    """
    k = B_obs.shape[1]
    r2 = np.full(k, np.nan)
    for j in range(k):
        ss_tot = np.sum((B_obs[:, j] - B_obs[:, j].mean()) ** 2)
        ss_res = np.sum((B_obs[:, j] - b_hat[:, j]) ** 2)
        if ss_tot > 1e-10:
            r2[j] = 1.0 - ss_res / ss_tot
    return r2


def diag_vertex_margin(b_hat, A_mat, q):
    """
    Diagnostic 2: Vertex-margin m_i = s_{i,(2)} - s_{i,(1)}.
    s_{ij} = v_j' b_hat_i for vertex v_j in T_q = {v: A'v >= q}.
    s_{i,(1)} <= s_{i,(2)} are the two smallest scores.
    m_i = gap between 1st and 2nd smallest score.

    Large m_i -> binding vertex clearly identified (robust).
    Near-zero m_i -> near-tie, tiny first-stage perturbation can flip vertex.
    Red flag: large share of obs with m_i < epsilon.
    """
    vertices = enumerate_dual_vertices(A_mat, q)
    if len(vertices) < 2:
        return np.zeros(len(b_hat))
    n = len(b_hat)
    margins = np.zeros(n)
    for i in range(n):
        scores = np.array([v @ b_hat[i] for v in vertices])
        s_sort = np.sort(scores)
        margins[i] = s_sort[1] - s_sort[0]
    return margins


def diag_binding_vertex_entropy(nu_sel):
    """
    Diagnostic 3: Binding-vertex distribution and entropy.
    p_hat_j = (1/n) sum 1{v_hat_i = v_j}
    H = -sum_j p_hat_j log(p_hat_j)

    Low H + strong tightening -> suspicious (face-selection collapse).
    Credible result: stable, spread vertex distribution across obs.
    """
    from collections import Counter
    keys = [tuple(np.round(nu_sel[i], 5)) for i in range(len(nu_sel))]
    cnt = Counter(keys)
    n = len(keys)
    phat = np.array([v / n for v in cnt.values()])
    H = float(-np.sum(phat * np.log(phat + 1e-12)))
    return H, len(cnt), float(max(cnt.values()) / n)


# ── D5: Calibration plot ───────────────────────────────────────────────────────
def calibration_plot(b_hat, B_obs, n_bins=10):
    """
    Calibration diagnostic: bin observations by predicted decile, compare
    mean(b_hat) vs mean(B_obs) within each bin.

    Slope = 0  → model collapsed to intercept (zero shrinkage dominates).
    Slope = 1  → well calibrated.
    Slope > 1  → overfitting (predictions too spread out).

    Returns dict: component_idx -> (pred_means array, obs_means array).
    """
    k = B_obs.shape[1]
    results = {}
    for j in range(k):
        idx  = np.argsort(b_hat[:, j])
        bins = np.array_split(idx, n_bins)
        pred_means = np.array([b_hat[b, j].mean() for b in bins])
        obs_means  = np.array([B_obs[b, j].mean()  for b in bins])
        results[j] = (pred_means, obs_means)
    return results


def _calibration_slope(pred_means, obs_means):
    """OLS slope of obs_means ~ pred_means across the n_bins bin means."""
    pm = np.asarray(pred_means, dtype=float)
    om = np.asarray(obs_means,  dtype=float)
    if pm.std() < 1e-10:
        return float('nan')
    return float(np.cov(pm, om)[0, 1] / np.var(pm))


# ── D7: Vertex stability test ─────────────────────────────────────────────────
def vertex_stability_test(b_hat, A_mat, q_vec, noise_frac=0.1, seed=7777):
    """
    Perturb b_hat by noise_frac * component-std, recompute binding dual vertex
    for each observation, measure fraction that switch vertex.

    < 5%  switch rate → stable first stage.
    > 50% switch rate → first stage landing near vertex boundaries (fragile).
    ~100% switch rate → first stage collapsed to intercept.

    Uses enumerate_dual_vertices (coarse model).
    """
    rng        = np.random.default_rng(seed)
    noise_std  = np.maximum(b_hat.std(axis=0) * noise_frac, 1e-10)
    b_perturbed = b_hat + rng.normal(0.0, noise_std, b_hat.shape)

    vertices = enumerate_dual_vertices(A_mat, q_vec)
    if len(vertices) < 2:
        return float('nan')

    def _binding_idx(b):
        vals = np.array([v @ b for v in vertices])
        return int(np.argmin(vals))

    orig_idx = np.array([_binding_idx(b) for b in b_hat])
    pert_idx = np.array([_binding_idx(b) for b in b_perturbed])
    return float(np.mean(orig_idx != pert_idx))


# ── D6: Feasibility check ─────────────────────────────────────────────────────
def feasibility_check(b_hat, A_mat):
    """
    For each observation i, solve min_beta ||A beta - b_hat[i]||_2  s.t. beta >= 0.
    Report the L2 residual.  Residual > 1e-6 => b_hat[i] outside the feasible image.

    A large fraction of infeasible predictions means the first stage is predicting
    states that are structurally impossible given the A matrix — a data construction
    or model misspecification issue.
    """
    n_beta = A_mat.shape[1]
    residuals = []
    for b in b_hat:
        try:
            res = lsq_linear(A_mat, b, bounds=(0.0, np.inf), method='bvls')
            residuals.append(np.sqrt(max(2.0 * res.cost, 0.0)))
        except Exception:
            residuals.append(np.inf)
    return np.array(residuals)


def print_config_diagnostics(config_name, B_obs, b_hat, A_mat, beta_names,
                              all_nu_sel_ub, all_results_config):
    """
    Print all diagnostics for one (feature_set, estimator) configuration.
    Called at the end of run_config().

    all_nu_sel_ub: list of (N, n_states) arrays, one per beta param (UB direction).
    """
    k      = B_obs.shape[1]
    n_beta = len(beta_names)
    SEP    = "-" * 72

    print(f"\n{SEP}")
    print(f"  DIAGNOSTICS: {config_name}")
    print(SEP)

    # ── D4: First-stage R^2 ─────────────────────────────────────────────────
    r2 = diag_firstsage_r2(B_obs, b_hat)
    state_labels = ["0n", "1n", "2n", "0p", "2p"][:k]
    print(f"\n  [D4] First-stage out-of-fold R^2 per B-component:")
    for j, sl in enumerate(state_labels):
        val  = f"{r2[j]:+.4f}" if not np.isnan(r2[j]) else "   nan"
        flag = "  <- WARNING zero/neg" if (np.isnan(r2[j]) or r2[j] < 0) else ""
        print(f"       B[{sl}]:  R^2={val}{flag}")
    n_neg = int(np.sum(r2 < 0))
    if n_neg > 0:
        print(f"    WARNING  {n_neg}/{k} components R^2<0 — first stage adds no info for those.")
    red_r2 = (n_neg > k // 2)

    # ── D2: Vertex-margin ───────────────────────────────────────────────────
    print(f"\n  [D2] Vertex-margin diagnostic (UB direction, m_i = s_{{i,(2)}} - s_{{i,(1)}}):")
    print(f"  {'beta':13s}  {'median':>7}  {'p10':>7}  {'%<1e-4':>7}  {'%<1e-3':>7}")
    print(f"  {'─'*48}")
    n_red_margin = 0
    for j, bname in enumerate(beta_names):
        q_up  = np.zeros(n_beta); q_up[j] = 1.0
        m     = diag_vertex_margin(b_hat, A_mat, q_up)
        med   = np.median(m)
        p10   = np.percentile(m, 10)
        pct1  = 100.0 * np.mean(m < 1e-4)
        pct3  = 100.0 * np.mean(m < 1e-3)
        flag  = "  WARNING" if pct1 > 20 else ""
        if pct1 > 20:
            n_red_margin += 1
        print(f"  {bname:13s}  {med:7.4f}  {p10:7.4f}  {pct1:6.1f}%  {pct3:6.1f}%{flag}")
    if n_red_margin > n_beta // 2:
        print(f"  WARNING  >50% beta params: many near-zero margins — vertex selection fragile.")

    # ── D3: Binding-vertex distribution ─────────────────────────────────────
    print(f"\n  [D3] Binding-vertex distribution (UB direction):")
    print(f"  {'beta':13s}  {'H(entropy)':>10}  {'n_vertices':>10}  {'top_freq':>8}")
    print(f"  {'─'*48}")
    for j, bname in enumerate(beta_names):
        H, nv, tf = diag_binding_vertex_entropy(all_nu_sel_ub[j])
        flag = "  WARNING collapse" if tf > 0.95 else ""
        print(f"  {bname:13s}  {H:10.4f}  {nv:10d}  {tf:8.4f}{flag}")

    # ── D5: Calibration plot ────────────────────────────────────────────────
    print(f"\n  [D5] Calibration diagnostic (mean pred vs mean obs, 10 decile bins):")
    calib = calibration_plot(b_hat, B_obs)
    state_labels_used = ["0n", "1n", "2n", "0p", "2p"][:k]
    print(f"  {'comp':6s}  {'slope':>8}  {'min_pred':>10}  {'max_pred':>10}  interpretation")
    print(f"  {'─'*56}")
    for j, sl in enumerate(state_labels_used):
        pm, om = calib[j]
        slope  = _calibration_slope(pm, om)
        if np.isnan(slope):
            interp = "n/a (no variance in pred)"
        elif abs(slope) < 0.2:
            interp = "COLLAPSED (slope≈0: intercept-only model)"
        elif slope > 1.3:
            interp = "OVERFIT   (slope>1: predictions too spread)"
        elif 0.7 <= slope <= 1.3:
            interp = "calibrated"
        else:
            interp = f"moderate (slope={slope:.2f})"
        slope_str = f"{slope:8.3f}" if not np.isnan(slope) else "     nan"
        print(f"  B[{sl}]  {slope_str}  {pm.min():10.5f}  {pm.max():10.5f}  {interp}")

    # ── D6: Feasibility check ───────────────────────────────────────────────
    print(f"\n  [D6] Feasibility check (b_hat in image of A restricted to beta>=0):")
    feas_resid = feasibility_check(b_hat, A_mat)
    frac_inf   = float(np.mean(feas_resid > 1e-6))
    print(f"       Frac infeasible (resid>1e-6):  {frac_inf:.4f}   "
          f"median resid: {np.median(feas_resid):.2e}   "
          f"max resid: {np.max(feas_resid):.2e}")
    if frac_inf > 0.05:
        print(f"       WARNING  {frac_inf*100:.1f}% of b_hat predictions fall outside "
              f"the feasible set — check A matrix or first-stage model.")

    # ── D7: Vertex stability ────────────────────────────────────────────────
    print(f"\n  [D7] Vertex stability under 10% noise perturbation (switch rate):")
    print(f"  {'beta':13s}  {'switch_rate':>12}  interpretation")
    print(f"  {'─'*48}")
    for j, bname in enumerate(beta_names):
        q_up = np.zeros(len(beta_names)); q_up[j] = 1.0
        sr   = vertex_stability_test(b_hat, A_mat, q_up)
        if np.isnan(sr):
            interp = "n/a"
        elif sr < 0.05:
            interp = "stable (<5%)"
        elif sr < 0.20:
            interp = "moderate (5-20%)"
        elif sr < 0.50:
            interp = "fragile (20-50%)  WARNING"
        else:
            interp = "UNSTABLE (>50%)  WARNING"
        sr_str = f"{sr:12.4f}" if not np.isnan(sr) else "         nan"
        print(f"  {bname:13s}  {sr_str}  {interp}")

    # ── Combined red-flag summary ────────────────────────────────────────────
    print(f"\n  Combined red-flag check:")
    print(f"    R^2<0 for >50% components: {'YES WARNING' if red_r2 else 'no'}")
    print(f"    Margin red flags:          {n_red_margin}/{n_beta} beta params")


# =============================================================================
# 11b. COMPOSITE BOUNDS  (KT Bounds.m, lines 525-700)
# =============================================================================
#
# KT Table 5 reports composite linear-combination bounds in addition to the
# 9 individual response probabilities.  Their LP form is
#
#     theta_M = f' pi   (linear combination of the 9 transition probabilities)
#
# KT solve:  min/max f' pi  s.t.  A pi = b,  0 <= pi <= 1,
# and report bounds on theta_M as `LPbound = [lb, ub]`.
#
# In the CLP framework the LP variables are beta = pi * P^a(source), so the
# corresponding linear-combination weight on beta is
#     q_beta_j = f_j / P^a(source_j)
# and the CLP estimator sigma_hat(q_beta) = max{q_beta' beta : A beta = b_0}
# is the SAME bound as KT's.
#
# We implement the three KT composites:
#   (A) Take-Up Work    : Not working under AFDC -> Working under JF
#                          KT line 533:
#                            f = [0 p00_c p01_c 0 p01_c p01_c 0 0 p01_c]
#                                / (p01_c + p00_c)
#                          [In CLP beta-ordering, the same composite is
#                           q_beta = [1, 0, 0, 1, 1, 1, 0, 1, 0]
#                                    / (p00_c + p01_c)
#                           which sums beta over all "not-working source ->
#                           working destination" transitions and divides by
#                           the AFDC mass in not-working source states.]
#                          Note: KT remark this is point-identified.
#
#   (B) Take-Up Welfare : Off welfare under AFDC -> On welfare under JF
#                          KT line 567:
#                            f = [0 p00_c 0 p20_c 0 0 0 p10_c 0]
#                                / (p00_c + p10_c + p20_c)
#                          [In CLP order: q_beta = [1, 0, 1, 0, 0, 0, 1, 0, 0]
#                                              / (p00_c + p10_c + p20_c)
#                           = (beta(0n,1r) + beta(2n,1r) + beta(1n,1r))
#                             / (P^a(0n) + P^a(1n) + P^a(2n))]
#
#   (C) Exit 0r -> n    : On welfare with zero earnings under AFDC -> Off
#                          welfare under JF.  KT line 666 (LP form):
#                            f = [1 0 1 0 0 0 0 0 1]                 (unnorm.)
#                          In CLP order: q_beta = [0, 1, 0, 1, 0, 1, 0, 0, 0]
#                          divided by p01_c to get pi-units.
#                          = (beta(0r,0n) + beta(0r,2n) + beta(0r,1n))
#                            / P^a(0p)
#
# Note the index permutation between KT order (0r0n, 0n1r, 0r2n, 2n1r,
# 0r1r, 0r2u, 2u1r, 1n1r, 0r1n) and our BETA_NAMES order (0n1r, 0r0n, 2n1r,
# 0r2n, 0r1r, 0r1n, 1n1r, 0r2u, 2u1r) -- the q vectors below already account
# for that.
#
# CLP-specific implementation:
#   - We pass q_beta to the existing clp_estimate(q, b_hat, B_obs, A_mat)
#     which solves the per-i LP via vertex enumeration.
#   - Person-clustered multiplier-bootstrap CI from multiplier_bootstrap_ci.
#   - Bounds are reported in pi-units (clip to [0,1] only for display).
# =============================================================================
def _composite_q_vectors(p):
    """
    Return a dict {composite_name: q_beta_vec} for the three KT composites,
    using the AFDC source-state probabilities in `p` (output of
    load_table4_mat).

    q_beta is in BETA_NAMES order:
      ["beta(0n,1r)", "beta(0r,0n)", "beta(2n,1r)", "beta(0r,2n)",
       "beta(0r,1r)", "beta(0r,1n)", "beta(1n,1r)", "beta(0r,2u)",
       "beta(2u,1r)"]
    """
    p00_c = p['p00_c']; p01_c = p['p01_c']
    p10_c = p['p10_c']; p20_c = p['p20_c']

    # (A) Take-Up Work: not working -> working
    # Numerator: sum of "not-working source -> working destination" beta's
    #   beta(0n,1r), beta(0r,2n), beta(0r,1r), beta(0r,1n), beta(0r,2u)
    # Denominator: P^a(0n) + P^a(0p)
    q_TUW = np.array([
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0
    ]) / (p00_c + p01_c)

    # (B) Take-Up Welfare: off welfare -> on welfare
    # Numerator: beta(0n,1r) + beta(2n,1r) + beta(1n,1r)
    # Denominator: P^a(0n) + P^a(1n) + P^a(2n)
    q_TUWelf = np.array([
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
    ]) / (p00_c + p10_c + p20_c)

    # (C) Exit 0r -> n
    # Numerator: beta(0r,0n) + beta(0r,2n) + beta(0r,1n)
    # Denominator: P^a(0p) = p01_c
    q_Exit = np.array([
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ]) / p01_c

    return {
        "Take-Up Work (not working -> working)":      q_TUW,
        "Take-Up Welfare (off -> on welfare)":         q_TUWelf,
        "Exit 0r (on-welfare zero earn -> off welfare)": q_Exit,
    }


def compute_composite_bounds(b_hat, B_obs, A_mat, person_id, p, label=""):
    """
    Compute KT-style composite bounds via the CLP machinery.

    For each of the 3 composite questions:
      - Build q_beta (linear combination weights on beta).
      - Compute UB = sigma_hat(+q_beta) and LB = -sigma_hat(-q_beta) using the
        existing clp_estimate (vertex enumeration over the 9-d dual polytope).
      - Person-clustered multiplier-bootstrap CI.
      - Print [LB, UB] and 95% CI; theta_M is in pi-units already
        (q_beta carries the source-population denominator).

    Returns a dict of {composite_name: result_dict}.
    """
    qs = _composite_q_vectors(p)
    results = {}

    print(f"\n  --- COMPOSITE BOUNDS ({label}) ---")
    print(f"  Composite parameters in pi-units (sums over multiple transitions, "
          f"normalized by source-state mass):")
    print(f"  {'composite':<55}  {'LB_pi':>8}  {'UB_pi':>8}  "
          f"{'width':>8}    {'95% CI':>22}")
    print("  " + "-" * 110)

    for name, q_beta in qs.items():
        ub_hat, c_up, _, _, _ = clp_estimate( q_beta, b_hat, B_obs, A_mat)
        ci_ub = list(multiplier_bootstrap_ci(
            c_up, n_bs=N_BOOTSTRAP, person_id=person_id))
        neg_lb, c_dn, _, _, _ = clp_estimate(-q_beta, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb
        ci_lb = [-x for x in
                 multiplier_bootstrap_ci(
                     c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]

        # Clip to [0,1] for pi-units (theta_M is a probability).
        lb_c = max(0.0, min(1.0, lb_hat))
        ub_c = max(0.0, min(1.0, ub_hat))
        ci_lo = max(0.0, min(1.0, ci_lb[0]))
        ci_hi = max(0.0, min(1.0, ci_ub[1]))

        results[name] = dict(
            lb=lb_hat, ub=ub_hat, width=ub_hat - lb_hat,
            ci_lb=tuple(ci_lb), ci_ub=tuple(ci_ub),
            lb_clipped=lb_c, ub_clipped=ub_c,
            ci_outer=(ci_lo, ci_hi),
        )
        print(f"  {name:<55}  {lb_c:8.4f}  {ub_c:8.4f}  "
              f"{ub_c - lb_c:8.4f}    [{ci_lo:.3f}, {ci_hi:.3f}]")

    return results


# =============================================================================
# 12.  RUN ONE CONFIGURATION
# =============================================================================
_FEAT_DIM = {"base": 28, "econ": 255}   # updated after data load if n_q < 8


def run_config(config_name, feature_set, estimator_name,
               D, state_obs, X_base_raw, X_econ_raw,
               person_id, pscorewt, A_mat, p,
               X_rf_raw=None, X_disc_raw=None):
    """Full CLP pipeline for one (feature_set, estimator) pair.
    Uses GroupKFold by person in the first stage."""
    if feature_set == "base":
        dim = X_base_raw.shape[1]
    elif feature_set == "econ":
        dim = X_base_raw.shape[1] + X_econ_raw.shape[1]
    elif feature_set == "rf":
        dim = X_base_raw.shape[1] + (X_rf_raw.shape[1] if X_rf_raw is not None else 0)
    elif feature_set == "disc":
        dim = X_base_raw.shape[1] + (X_disc_raw.shape[1] if X_disc_raw is not None else 0)
    else:
        dim = X_base_raw.shape[1]

    print(f"\n{'='*68}")
    print(f"  CONFIG: {config_name}")
    print(f"  feature_set={feature_set} ({dim} features)   estimator={estimator_name}")
    if estimator_name == "OLS" and feature_set == "econ":
        print(f"  WARNING  OLS + econ ({dim} features) has no regularization — "
              f"high variance; treat as benchmark only.")
    print(f"{'='*68}")

    b_hat, B_obs = estimate_b0_features(
        D, state_obs, X_base_raw, X_econ_raw,
        feature_set=feature_set, estimator_name=estimator_name,
        K=K_FOLDS, seed=RANDOM_SEED, pscorewt=pscorewt,
        person_id=person_id,
        X_rf_raw=X_rf_raw, X_disc_raw=X_disc_raw,
    )
    b_mean = B_obs.mean(axis=0)
    b_fit  = b_hat.mean(axis=0)
    print(f"  True b0 (sample mean B):  {b_mean.round(4)}")
    print(f"  First-stage b_hat0 (mean):    {b_fit.round(4)}")
    print(f"  Max |bias| in b_hat0 mean:    {np.max(np.abs(b_mean - b_fit)):.2e}")

    # First-stage fit quality
    print(f"  Pearson r (B_obs vs b_hat0) per component:", end="  ")
    for j in range(5):
        if B_obs[:, j].std() > 1e-10 and b_hat[:, j].std() > 1e-10:
            r = float(np.corrcoef(B_obs[:, j], b_hat[:, j])[0, 1])
        else:
            r = float('nan')
        print(f"{r:.3f}", end="  ")
    print()

    n_pi    = len(BETA_NAMES)
    results = []
    nu_sel_ub_list = []

    print(f"\n  {'beta param':<13}  {'LB':>7}  {'UB':>7}  "
          f"{'Width':>7}  {'95% CI LB':>14}  {'95% CI UB':>14}")
    print("  " + "-" * 74)

    for j in range(n_pi):
        q_up = np.zeros(n_pi); q_up[j] =  1.0
        q_dn = np.zeros(n_pi); q_dn[j] = -1.0

        ub_hat, c_up, nu_up, _verts_up, _vidx_up = clp_estimate(q_up, b_hat, B_obs, A_mat)
        nu_sel_ub_list.append(nu_up)
        ci_ub = list(multiplier_bootstrap_ci(
            c_up, n_bs=N_BOOTSTRAP, person_id=person_id))

        neg_lb, c_dn, _nu_dn, _verts_dn, _vidx_dn = clp_estimate(q_dn, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb
        ci_lb  = [-x for x in
                  multiplier_bootstrap_ci(
                      c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]

        lb_hat   = max(0.0, lb_hat)
        ub_hat   = max(0.0, min(1.0, ub_hat))
        ci_lb[0] = max(0.0, ci_lb[0])
        ci_lb[1] = max(0.0, ci_lb[1])
        ci_ub[0] = max(0.0, ci_ub[0])
        ci_ub[1] = min(1.0, ci_ub[1])
        width    = ub_hat - lb_hat

        results.append(dict(
            name=BETA_NAMES[j],
            lb=lb_hat, ub=ub_hat, width=width,
            ci_lb=tuple(ci_lb), ci_ub=tuple(ci_ub),
        ))
        flag = " (*)" if width < -1e-6 else ""
        print(f"  {BETA_NAMES[j]:<13}  {lb_hat:7.4f}  {ub_hat:7.4f}  "
              f"{width:7.4f}{flag}  "
              f"  [{ci_lb[0]:.3f}, {ci_lb[1]:.3f}]  "
              f"  [{ci_ub[0]:.3f}, {ci_ub[1]:.3f}]")

    # beta -> pi back-conversion
    source_pop = np.array([
        p['p00_c'], p['p01_c'], p['p20_c'], p['p01_c'], p['p01_c'],
        p['p01_c'], p['p10_c'], p['p01_c'], p['p21_c'],
    ])

    def _c(x):
        return float(min(1.0, max(0.0, x)))

    print(f"\n  {'pi param':<13}  {'LB_pi':>8}  {'UB_pi':>8}  "
          f"{'95% CI LB_pi':>16}  {'95% CI UB_pi':>16}")
    print("  " + "-" * 70)

    for j, res in enumerate(results):
        sp = source_pop[j]
        if sp <= 1e-12:
            print(f"  {PI_NAMES[j]:<13}  N/A (P^a~=0)")
            continue
        lb_p = _c(res['lb'] / sp);  ub_p = _c(res['ub'] / sp)
        cllo = _c(res['ci_lb'][0] / sp); clhi = _c(res['ci_lb'][1] / sp)
        culo = _c(res['ci_ub'][0] / sp); cuhi = _c(res['ci_ub'][1] / sp)
        print(f"  {PI_NAMES[j]:<13}  {lb_p:8.4f}  {ub_p:8.4f}  "
              f"  [{cllo:.4f}, {clhi:.4f}]    [{culo:.4f}, {cuhi:.4f}]")

    # ── Composite bounds (KT Bounds.m TUW / Take-Up Welfare / Exit) ───────
    composite_results = compute_composite_bounds(
        b_hat, B_obs, A_mat, person_id, p, label=config_name,
    )

    # ── Print diagnostics ──────────────────────────────────────────────────
    print_config_diagnostics(config_name, B_obs, b_hat, A_mat, BETA_NAMES,
                             nu_sel_ub_list, {r['name']: r for r in results})

    # Attach composite results so that the caller can access them.
    out = {r['name']: r for r in results}
    out['__composite__'] = composite_results
    return out


# =============================================================================
# 13.  SUMMARY TABLE
# =============================================================================
def print_summary_table(all_results):
    """
    Side-by-side beta interval widths [UB - LB] across all configurations.
    Organised into feature-set panels for easy reading.
    """
    config_names = list(all_results.keys())
    col_w = 13

    print(f"\n\n{'='*72}")
    print("SUMMARY: beta interval widths  [UB - LB]")
    print("Narrower = tighter identified set.  (*) = LB > UB (convergence issue).")
    print(f"{'='*72}")

    print(f"\n  {'beta param':<13}  " +
          "  ".join(f"{n[:col_w]:>{col_w}}" for n in config_names))
    print("  " + "-" * (13 + (col_w + 2) * len(config_names) + 2))

    any_neg = False
    for beta in BETA_NAMES:
        row = f"  {beta:<13}  "
        cells = []
        for name in config_names:
            res = all_results[name].get(beta)
            if res is None:
                cells.append(f"{'N/A':>{col_w}}")
            else:
                w = res['width']
                if w < -1e-6:
                    any_neg = True
                    cells.append(f"{'(*)'+f'{w:.3f}':>{col_w}}")
                else:
                    cells.append(f"{w:>{col_w}.4f}")
        print(row + "  ".join(cells))

    print("  " + "-" * (13 + (col_w + 2) * len(config_names) + 2))
    means = []
    for name in config_names:
        widths = [v['width'] for k, v in all_results[name].items() if k != '__composite__']
        means.append(np.mean(widths))
    print(f"  {'Mean width':<13}  " +
          "  ".join(f"{m:>{col_w}.4f}" for m in means))

    # — by feature set —
    print(f"\n  Average width by feature set:")
    for fset in ("base", "econ", "rf", "disc"):
        matching = [n for n in config_names
                    if CONFIGS_TO_RUN.get(n, ('', ''))[0] == fset]
        if not matching:
            continue
        avg = np.mean([np.mean([v['width'] for k, v in all_results[m].items() if k != '__composite__'])
                       for m in matching])
        dim = "28" if fset == "base" else "255 (econ)" if fset == "econ" else fset
        print(f"    {fset:<6} ({dim} features):  mean width = {avg:.4f}")

    # — by estimator —
    print(f"\n  Average width by estimator:")
    for est in ("LASSO", "Ridge", "OLS", "PostLasso"):
        matching = [n for n in config_names
                    if CONFIGS_TO_RUN.get(n, ('', ''))[1] == est]
        if not matching:
            continue
        avg = np.mean([np.mean([v['width'] for k, v in all_results[m].items() if k != '__composite__'])
                       for m in matching])
        note = " (WARNING no regularization — benchmark only)" if est == "OLS" else ""
        print(f"    {est:<14}:  mean width = {avg:.4f}{note}")

    # — Delta econ vs base —
    est_list = ["LASSO", "Ridge", "OLS"]
    print(f"\n  Width change: econ vs base (negative = tighter, positive = wider)")
    print(f"  {'beta param':<13}  " +
          "  ".join(f"{'D '+est[:7]:>13}" for est in est_list))
    print("  " + "-" * (13 + 15 * len(est_list) + 2))
    for beta in BETA_NAMES:
        row = f"  {beta:<13}  "
        cells = []
        for est in est_list:
            base_key = f"base_{est.lower()}"
            econ_key = f"econ_{est.lower()}"
            wb = all_results.get(base_key, {}).get(beta, {}).get('width')
            we = all_results.get(econ_key, {}).get(beta, {}).get('width')
            if wb is None or we is None:
                cells.append(f"{'N/A':>13}")
            else:
                cells.append(f"{(we - wb):+13.4f}")
        print(row + "  ".join(cells))
    print("  " + "-" * (13 + 15 * len(est_list) + 2))
    row = f"  {'Mean D':<13}  "
    cells = []
    for est in est_list:
        base_key = f"base_{est.lower()}"
        econ_key = f"econ_{est.lower()}"
        deltas = []
        for beta in BETA_NAMES:
            wb = all_results.get(base_key, {}).get(beta, {}).get('width')
            we = all_results.get(econ_key, {}).get(beta, {}).get('width')
            if wb is not None and we is not None:
                deltas.append(we - wb)
        cells.append(f"{np.mean(deltas):+13.4f}" if deltas else f"{'N/A':>13}")
    print(row + "  ".join(cells))

    print(f"\n  NOTE: K={K_FOLDS}-fold GroupKFold by person cross-fitting, "
          f"N_BOOTSTRAP={N_BOOTSTRAP}.")
    print(f"  OLS + econ (255 features): no regularization — high-variance benchmark.")
    if any_neg:
        print("\n  (*) Negative width = LB > UB.  First-stage convergence failure.")
        print("      Do NOT interpret as valid bounds.")

    # ── D1: Stability of tightening across estimators ──────────────────────────
    print(f"\n  [D1] Stability of tightening (D_r = width per estimator per beta):")
    est_list = ["LASSO", "Ridge", "OLS", "PostLasso"]
    print(f"  {'beta':13s}  {'LASSO':>8}  {'Ridge':>8}  {'OLS':>8}  {'PostLasso':>10}  {'mean':>8}  {'std':>8}  {'signfl':>7}")
    print(f"  {'─'*84}")
    for beta in BETA_NAMES:
        ws = {}
        for cfg, (fset, est) in CONFIGS_TO_RUN.items():
            r = all_results.get(cfg, {}).get(beta)
            if r is not None:
                ws.setdefault(est, []).append(r['width'])
        wvals = {e: np.mean(v) for e, v in ws.items()}
        wl = wvals.get("LASSO",     float('nan'))
        wr = wvals.get("Ridge",     float('nan'))
        wo = wvals.get("OLS",       float('nan'))
        wp = wvals.get("PostLasso", float('nan'))
        arr = [v for v in [wl, wr, wo, wp] if not np.isnan(v)]
        mean_w = np.mean(arr) if arr else float('nan')
        std_w  = np.std(arr)  if len(arr) > 1 else float('nan')
        sf     = int(np.sum(np.array(arr) < 0)) if arr else 0
        flag   = "  WARNING" if sf > 0 or (not np.isnan(std_w) and not np.isnan(mean_w) and mean_w > 1e-4 and std_w / mean_w > 0.5) else ""
        wl_s = f"{wl:8.4f}" if not np.isnan(wl) else "     nan"
        wr_s = f"{wr:8.4f}" if not np.isnan(wr) else "     nan"
        wo_s = f"{wo:8.4f}" if not np.isnan(wo) else "     nan"
        wp_s = f"{wp:10.4f}" if not np.isnan(wp) else "       nan"
        mw_s = f"{mean_w:8.4f}" if not np.isnan(mean_w) else "     nan"
        sw_s = f"{std_w:8.4f}" if not np.isnan(std_w) else "     nan"
        print(f"  {beta:13s}  {wl_s}  {wr_s}  {wo_s}  {wp_s}  {mw_s}  {sw_s}  {sf:7d}{flag}")
    print(f"  NOTE: Trustworthy tightening: D_r>0 consistently, small std/mean ratio,")
    print(f"        similar across learners. Sign flips or learner-specific collapse = red flag.")


# =============================================================================
# 14.  MAIN
# =============================================================================
def main():
    import time
    t_start = time.time()

    print("=" * 68)
    print("CLP Final (GroupKFold)  —  LASSO / Ridge / OLS  (base vs. econ set)")
    print("=" * 68)
    print("Configurations (6 total, no RandomForest, GroupKFold by person):")
    for name, (fset, est) in CONFIGS_TO_RUN.items():
        print(f"  {name:<18}  features={fset:<6}  estimator={est}")
    print(f"\nK_FOLDS={K_FOLDS}  N_BOOTSTRAP={N_BOOTSTRAP}  SEED={RANDOM_SEED}")

    # ── 1. Table4_mat (back-conversion prior) ────────────────────────────────
    print("\nLoading Table4_mat.txt ...")
    p, _ = load_table4_mat()

    # ── 2. Individual-level data ──────────────────────────────────────────────
    print("\nPreparing individual-level data ...")
    D, state_obs, df_incl, person_id, pscorewt = prepare_jf_data()

    # Update p with IPW-adjusted frequencies from individual data
    ctrl, treat = (D == 0), (D == 1)
    for key, mask in [('c', ctrl), ('t', treat)]:
        w = pscorewt[mask]; ws = w.sum()
        p[f'p00_{key}'] = np.sum((state_obs[mask] == S_0N) * w) / ws
        p[f'p10_{key}'] = np.sum((state_obs[mask] == S_1N) * w) / ws
        p[f'p20_{key}'] = np.sum((state_obs[mask] == S_2N) * w) / ws
        p[f'p01_{key}'] = np.sum((state_obs[mask] == S_0P) * w) / ws
        p[f'p11_{key}'] = np.sum((state_obs[mask] == S_1P) * w) / ws
        p[f'p21_{key}'] = np.sum((state_obs[mask] == S_2P) * w) / ws

    print("\n  State probs (IPW-adjusted, control):")
    print(f"    0n={p['p00_c']:.4f}  1n={p['p10_c']:.4f}  2n={p['p20_c']:.4f}  "
          f"0p={p['p01_c']:.4f}  1p={p['p11_c']:.4f}  2p={p['p21_c']:.4f}")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\nEngineering econ feature set ...")
    X_base_raw, X_econ_raw, feat_names, group_info = engineer_features_econ(df_incl)

    print(f"\n  Feature group summary:")
    for label, slc in group_info.items():
        n_feat = slc.stop - slc.start
        print(f"    {label:<35}  {n_feat:>4} features")

    # ── RF importance-based features ────────────────────────────────────────
    print("\nEngineering RF importance-based feature set ...")
    X_rf_raw = engineer_features_rf(df_incl, D, state_obs, pscorewt)

    # ── Discretized features ─────────────────────────────────────────────────
    print("\nEngineering discretized feature set ...")
    X_disc_raw = engineer_features_disc(df_incl)

    # ── 4. A matrix ───────────────────────────────────────────────────────────
    A_mat = build_A()

    # ── 5. Run all configurations ─────────────────────────────────────────────
    all_results = {}
    for config_name, (feature_set, estimator_name) in CONFIGS_TO_RUN.items():
        all_results[config_name] = run_config(
            config_name, feature_set, estimator_name,
            D, state_obs, X_base_raw, X_econ_raw,
            person_id, pscorewt, A_mat, p,
            X_rf_raw=X_rf_raw, X_disc_raw=X_disc_raw,
        )

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── 7. Save results to CSV ────────────────────────────────────────────────
    import os
    rows = []
    for cfg, res in all_results.items():
        fset, est = CONFIGS_TO_RUN[cfg]
        for beta, r in res.items():
            rows.append({
                'config': cfg, 'feature_set': fset, 'estimator': est,
                'beta': beta,
                'lb': r['lb'], 'ub': r['ub'], 'width': r['width'],
                'ci_lb_lo': r['ci_lb'][0], 'ci_lb_hi': r['ci_lb'][1],
                'ci_ub_lo': r['ci_ub'][0], 'ci_ub_hi': r['ci_ub'][1],
            })
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "CLP_final_group_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
