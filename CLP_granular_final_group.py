"""
CLP_granular_final_group.py
============================
CLP estimator (Semenova 2026) applied to the Kline-Tartari (2016) Jobs First
data with a GRANULAR 9-bin income classification, combined with economically
motivated feature engineering.

Runs CLP bounds with 6 configurations:
  {base, econ} x {LASSO, Ridge, OLS}

where:
  "base" = 28 original covariates
  "econ" = 28 base + 227 economically motivated interaction/quadratic features
           (255 total)

All estimators use 5-fold GroupKFold cross-fitting (grouped by person).
All quarters of the same person are held out together, preventing data leakage across person-quarters.
Within-fold StandardScaler (fit on train, apply to test) — NO global scaling.
13 B-score components (granular model).

Income bins (relative to FPL threshold Cbin):
  0  : earnq == 0
  b1 : 0 < earnq <= 0.20 * Cbin
  b2 : 0.20 < earnq <= 0.40 * Cbin
  b3 : 0.40 < earnq <= 0.60 * Cbin
  b4 : 0.60 < earnq <= 0.80 * Cbin
  b5 : 0.80 < earnq <= 1.00 * Cbin
  b6 : 1.00 < earnq <= 1.20 * Cbin
  b7 : 1.20 < earnq <= 1.40 * Cbin
  b8 :         earnq >  1.40 * Cbin

Non-reference states modelled (13 rows in A):
  Row  0 : 0n   (ebin=0, partic=0)
  Row  1 : b1n  (ebin=1, partic=0)
  ...
  Row 12 : b8p  (ebin=8, partic=1)

Dependencies:  numpy  scipy  pandas  scikit-learn
"""

# =============================================================================
# SECTION 1 — IMPORTS & CONSTANTS
# =============================================================================
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize import lsq_linear
from itertools import combinations
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
JF_DTA_PATH       = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/JF.dta")
POLICY_RULES_PATH = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/PolicyRules.dta")
TABLE4_MAT_PATH   = ("/Users/gevorgkhandamiryan/Desktop/Working Folder/"
                     "KT Replication package/AER_Code/DerivedData/Table4_mat.txt")

N_BOOTSTRAP = 200
K_FOLDS     = 5
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# PROPENSITY-SCORE VARIABLES
# -----------------------------------------------------------------------------
PSCORE_VARS = (
    ["ernpq8"]
    + [f"ernpq{q}" for q in range(6, 0, -1)]       # ernpq6..ernpq1
    + [f"adcpq{q}" for q in range(7, 0, -1)]        # adcpq7..adcpq1
    + [f"fstpq{q}" for q in range(7, 0, -1)]        # fstpq7..fstpq1
    + [f"anyernpq{q}" for q in range(1, 7)]         # anyernpq1..anyernpq6
    + ["anyernpq8"]
    + [f"anyadcpq{q}" for q in range(1, 8)]         # anyadcpq1..anyadcpq7
    + [f"anyfstpq{q}" for q in range(1, 8)]         # anyfstpq1..anyfstpq7
    + ["yremp", "prevafdc", "white", "black", "hisp",
       "marnvr", "marapt", "agelt25", "age2534",
       "nohsged", "hsged", "kidctgt2", "applcant",
       "misshs", "misskidctgt2", "missmar",
       "ernpq7", "anyernpq7"]
)

JF_CONFIG = dict(
    first_quarter  = 1,
    last_quarter   = 7,
    which_fpl      = "nextsizeup",
    covariate_vars = [
        "age2534", "black", "hisp", "white",
        "marnvr", "marapt", "hsged", "nohsged", "yngchtru", "kidcount",
        "ernpq1", "ernpq2", "ernpq3", "ernpq4",
        "ernpq5", "ernpq6", "ernpq7", "ernpq8",
        "adcpq1", "adcpq2", "adcpq3", "adcpq4",
        "adcpq5", "adcpq6", "adcpq7", "adcpq8",
        "applcant", "yremp",
    ],
)

# -----------------------------------------------------------------------------
# STATE & BETA-PARAMETER INDEXING
# -----------------------------------------------------------------------------

# Non-reference state codes: (ebin, partic) -> row index in A
# Reference states (not in A): ebin in {1..5} with partic=1  (i.e. b1p..b5p)
STATE_ROWS = {
    (0, 0): 0,   # 0n
    (1, 0): 1,   # b1n
    (2, 0): 2,   # b2n
    (3, 0): 3,   # b3n
    (4, 0): 4,   # b4n
    (5, 0): 5,   # b5n
    (6, 0): 6,   # b6n
    (7, 0): 7,   # b7n
    (8, 0): 8,   # b8n
    (0, 1): 9,   # 0p
    (6, 1): 10,  # b6p
    (7, 1): 11,  # b7p
    (8, 1): 12,  # b8p
}
N_STATES   = 13   # number of non-reference states (= rows of A)
N_BETA     = 33   # number of beta parameters       (= columns of A)

# beta parameter names, in column order
BETA_NAMES = (
    # Group 1: beta(0n -> b_j AFDC-on),  j=1..5   [cols 0-4]
    ["beta(0n,b1r)", "beta(0n,b2r)", "beta(0n,b3r)", "beta(0n,b4r)", "beta(0n,b5r)"]
    # Group 2: beta(0p -> 0n)            [col  5]
    + ["beta(0r,0n)"]
    # Group 3: beta(b_{6+k}n -> 1r),    k=0..2   [cols 6-8]
    + ["beta(b6n,1r)", "beta(b7n,1r)", "beta(b8n,1r)"]
    # Group 4: beta(0p -> b_{6+k}n),    k=0..2   [cols 9-11]
    + ["beta(0r,b6n)", "beta(0r,b7n)", "beta(0r,b8n)"]
    # Group 5: beta(0p -> b_j AFDC-on), j=1..5   [cols 12-16]
    + ["beta(0r,b1r)", "beta(0r,b2r)", "beta(0r,b3r)", "beta(0r,b4r)", "beta(0r,b5r)"]
    # Group 6: beta(0p -> b_j no-AFDC), j=1..5   [cols 17-21]
    + ["beta(0r,b1n)", "beta(0r,b2n)", "beta(0r,b3n)", "beta(0r,b4n)", "beta(0r,b5n)"]
    # Group 7: beta(b_j n -> 1r),       j=1..5   [cols 22-26]
    + ["beta(b1n,1r)", "beta(b2n,1r)", "beta(b3n,1r)", "beta(b4n,1r)", "beta(b5n,1r)"]
    # Group 8: beta(0p -> b_{6+k}p),    k=0..2   [cols 27-29]
    + ["beta(0r,b6u)", "beta(0r,b7u)", "beta(0r,b8u)"]
    # Group 9: beta(b_{6+k}p -> 1r),    k=0..2   [cols 30-32]
    + ["beta(b6u,1r)", "beta(b7u,1r)", "beta(b8u,1r)"]
)
assert len(BETA_NAMES) == N_BETA, f"Expected {N_BETA} beta params, got {len(BETA_NAMES)}"

# pi (transition probability) names — same order as beta
PI_NAMES = [n.replace("beta", "pi") for n in BETA_NAMES]

# Source state probabilities (for beta -> pi back-conversion).
# Keyed by beta-column index; value is (ebin, partic) of the source state.
BETA_SOURCE_STATE = (
    [(0, 0)] * 5          # Group 1: source = 0n
    + [(0, 1)]            # Group 2: source = 0p
    + [(6, 0), (7, 0), (8, 0)]   # Group 3
    + [(0, 1)] * 3        # Group 4: source = 0p
    + [(0, 1)] * 5        # Group 5: source = 0p
    + [(0, 1)] * 5        # Group 6: source = 0p
    + [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]   # Group 7
    + [(0, 1)] * 3        # Group 8: source = 0p
    + [(6, 1), (7, 1), (8, 1)]   # Group 9
)
assert len(BETA_SOURCE_STATE) == N_BETA

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


# =============================================================================
# SECTION 2 — A MATRIX
# =============================================================================
def build_A_granular():
    """
    Build the 13 x 33 A matrix for the granular 9-bin income model.

    Each entry A[row, col] in {-1, 0, +1}.
      +1 : the beta parameter in that column is an INFLOW  to the state in that row
      -1 : the beta parameter in that column is an OUTFLOW from the state in that row

    The structure inherits directly from the original 5 x 9 coarse A matrix:
      - "1" (below-FPL) state expands into a 5-row x 5-col diagonal block
      - "2" (above-FPL) state expands into a 3-row x 3-col diagonal block
      - "0n" and "0p" rows do not split; their column-blocks expand accordingly
    """
    A = np.zeros((N_STATES, N_BETA), dtype=float)

    # -- Row 0: state 0n
    # Outflows to b1r..b5r (reference), one column each
    A[0, 0:5] = -1.0    # cols 0-4: beta(0n,b_j r)
    # Inflow from 0p
    A[0, 5]   = +1.0    # col  5: beta(0r,0n)

    # -- Rows 1-5: states b1n..b5n  (below-FPL no-AFDC sub-bins)
    # Block-diagonal: row k (k=1..5) gets inflow from col 16+k and outflow in col 21+k
    for k in range(5):
        row = 1 + k
        col_in  = 17 + k   # beta(0r,b_{k+1}n)
        col_out = 22 + k   # beta(b_{k+1}n,1r)
        A[row, col_in]  = +1.0
        A[row, col_out] = -1.0

    # -- Rows 6-8: states b6n..b8n  (above-FPL no-AFDC sub-bins)
    # Block-diagonal: row 6+k gets inflow from col 9+k and outflow in col 6+k
    for k in range(3):
        row = 6 + k
        col_out = 6 + k    # beta(b_{6+k}n,1r)
        col_in  = 9 + k    # beta(0r,b_{6+k}n)
        A[row, col_in]  = +1.0
        A[row, col_out] = -1.0

    # -- Row 9: state 0p
    # 0p is a pure-outflow state; all outgoing beta parameters contribute -1 here.
    # col  5: beta(0r,0n)              -> outflow from 0p
    A[9, 5]      = -1.0
    # cols 9-11: beta(0r,b6n..b8n)    -> outflow from 0p to above-FPL no-AFDC
    A[9, 9:12]   = -1.0
    # cols 12-16: beta(0r,b1r..b5r)   -> outflow from 0p to below-FPL AFDC-on (ref)
    A[9, 12:17]  = -1.0
    # cols 17-21: beta(0r,b1n..b5n)   -> outflow from 0p to below-FPL no-AFDC
    A[9, 17:22]  = -1.0
    # cols 27-29: beta(0r,b6u..b8u)   -> outflow from 0p to above-FPL AFDC-on
    A[9, 27:30]  = -1.0

    # -- Rows 10-12: states b6p..b8p  (above-FPL AFDC-on sub-bins)
    # Block-diagonal: row 10+k gets inflow from col 27+k and outflow in col 30+k
    for k in range(3):
        row = 10 + k
        col_in  = 27 + k   # beta(0r,b_{6+k}u)
        col_out = 30 + k   # beta(b_{6+k}u,1r)
        A[row, col_in]  = +1.0
        A[row, col_out] = -1.0

    return A


def _verify_A(A):
    """Quick sanity checks on the A matrix."""
    assert A.shape == (N_STATES, N_BETA), f"Shape mismatch: {A.shape}"
    assert set(np.unique(A)).issubset({-1.0, 0.0, 1.0}), "Non-+-1 entries found"
    # Every column must have at least one non-zero entry
    assert (np.abs(A).sum(axis=0) > 0).all(), "A has a zero column"
    # Every row must have at least one non-zero entry
    assert (np.abs(A).sum(axis=1) > 0).all(), "A has a zero row"
    print(f"  A matrix verified: {N_STATES}x{N_BETA}, entries in {{-1,0,+1}}")
    # Count non-zeros
    nnz = int((A != 0).sum())
    print(f"  Non-zero entries: {nnz}  (density {nnz/(N_STATES*N_BETA):.2%})")


# =============================================================================
# SECTION 3 — DATA LOADING
# =============================================================================
def fit_pscore_logit(person_ids, e_vals, X_mat, seed=RANDOM_SEED):
    """
    Fit a propensity-score logit on PERSON-LEVEL data and return
    pscorewt = e/pscore + (1-e)/(1-pscore)  for each person.

    Replicates DistributionOverStates_P exactly:
      - Algorithm : statsmodels Newton-Raphson  (= Stata logit default)
      - Sample    : filtered 4,642-person dataset (kidcount not missing, Q1-7)
      - Variables : full 59-variable PSCORE_VARS from AllGlobal.do $pscorevars
      - Options   : asis -> no pre-drop; replicated by dropping zero-variance cols.

    Falls back to sklearn L-BFGS if statsmodels is unavailable or fails.
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
                method='newton',
                maxiter=200,
                tol=1e-8,
                disp=False,
                warn_convergence=False,
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


def prepare_jf_data_granular(cfg=None,
                              jf_path=JF_DTA_PATH,
                              rules_path=POLICY_RULES_PATH):
    """
    Replicate KT Stata pipeline (DistributionOverStates.do) with GRANULAR
    9-bin income classification instead of the original 3 bins.

    Follows the exact same steps as CLP.py's prepare_jf_data():
      1. Filter to Q1-Q7, kidcount not missing
      2. AU size = kidcount+1  (kidcount=0 -> size=2)
      3. Merge PolicyRules.dta -> F3, compute Cbin = round(3xF3_{nextsizeup}, 100)
      4. Granular ebin:  0 if earnq=0, 1-5 for fractions of Cbin <= FPL,
                         6-8 for fractions of Cbin > FPL
      5. Welfare: included iff afdcon_nminq in {0,3}; partic=1/0 accordingly
      6. Keep 1 month per quarter (first within id-quarter)
      7. CRITICAL: refit propensity-score logit on the filtered person-level
         sample using the full 59-var $pscorevars (mirrors DistributionOverStates_P)
      8. Keep included only; map to A-matrix row index

    Returns df_incl instead of a pre-scaled X matrix, so engineer_features_econ
    can build feature matrices from the raw DataFrame columns.

    Returns
    -------
    D          : (N,) int      treatment indicator (1=JF, 0=AFDC)
    ebin_arr   : (N,) int      granular income bin 0..8
    partic_arr : (N,) int      AFDC participation 0/1
    state_row  : (N,) int      row index in A matrix; -1 for reference states
    df_incl    : DataFrame     filtered included person-quarters (raw columns)
    person_id  : (N,)          person identifier
    pscorewt   : (N,) float    IPW weight (refitted logit, person-level)
    """
    if cfg is None:
        cfg = JF_CONFIG

    fq, lq = cfg['first_quarter'], cfg['last_quarter']

    # -- 1. Load policy rules
    print(f"\n[JF DATA]  Loading policy rules from {rules_path} ...")
    rules = pd.read_stata(rules_path)[['year', 'size', 'F3']].copy()
    rules['year'] = rules['year'].astype(int)
    rules['size'] = rules['size'].astype(int)
    f3_map = {(r['year'], r['size']): r['F3'] for _, r in rules.iterrows()}
    print(f"  Policy rules: {len(rules)} rows, years {rules['year'].min()}-"
          f"{rules['year'].max()}, sizes {rules['size'].min()}-{rules['size'].max()}")

    # -- 2. Load JF.dta
    print(f"[JF DATA]  Loading {jf_path}  (this may take 30-60 s) ...")
    df = pd.read_stata(jf_path)
    print(f"  Loaded: {len(df):,} rows x {df.shape[1]} columns")

    # -- 3. Filter to outcome quarters, kidcount not missing
    df = df[(df['quarter'] >= fq) & (df['quarter'] <= lq)].copy()
    df = df[df['kidcount'].notna()].copy()
    print(f"  After filter Q{fq}-Q{lq} & kidcount not missing: {len(df):,} rows")

    # -- 4. Compute AU size
    # Stata: size = kidcount+1 if kidcount in {1,2,3}; size=2 if kidcount=0
    df['size'] = np.nan
    mask_123 = df['kidcount'].isin([1, 2, 3])
    df.loc[mask_123, 'size'] = df.loc[mask_123, 'kidcount'] + 1
    df.loc[df['kidcount'] == 0, 'size'] = 2
    df = df[df['size'].notna()].copy()
    df['size'] = df['size'].astype(int)

    # Lookup size for FPL
    if cfg['which_fpl'] == "nextsizeup":
        df['lookup_size'] = df['size'] + 1
    elif cfg['which_fpl'] == "twosizesup":
        df['lookup_size'] = df['size'] + 2
    else:
        df['lookup_size'] = df['size']

    # -- 5. Merge policy rules -> F3
    df['year_int'] = df['year'].astype(int)
    df['F3_nextsizeup'] = df.apply(
        lambda r: f3_map.get((r['year_int'], r['lookup_size']), np.nan), axis=1
    )
    n_before = len(df)
    df = df[df['F3_nextsizeup'].notna()].copy()
    if n_before > len(df):
        print(f"  Dropped {n_before - len(df)} rows with no matching policy rules")
    print(f"  After merge with policy rules: {len(df):,} rows")

    # -- 6. Compute Cbin (quarterly FPL threshold, rounded to $100)
    df['Cbin'] = np.round(3.0 * df['F3_nextsizeup'] / 100.0) * 100.0

    # -- 7. Granular earnings bin
    thresholds = [0.0, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40]
    df['ebin'] = np.nan
    df.loc[df['earnq'] == 0, 'ebin'] = 0
    for k, (lo_frac, hi_frac) in enumerate(zip(thresholds, thresholds[1:]), start=1):
        lo = lo_frac * df['Cbin']
        hi = hi_frac * df['Cbin']
        mask = (df['earnq'] > lo) & (df['earnq'] <= hi) & df['earnq'].notna()
        df.loc[mask, 'ebin'] = k
    # ebin=8: earnq > 1.40 x Cbin
    mask_b8 = (df['earnq'] > 1.40 * df['Cbin']) & df['earnq'].notna()
    df.loc[mask_b8, 'ebin'] = 8

    # -- 8. Welfare participation
    # Stata: included = (afdcon_nminq==0 | afdcon_nminq==3)
    #        partic=1 if afdcon_nminq==3, partic=0 if afdcon_nminq==0
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

    # -- 9. Keep 1 month per quarter (first within id-quarter)
    # Stata: bys id quarter: replace nn = _n; drop if nn != 1
    df = df.sort_values(['id', 'year', 'quarter', 'month'])
    df['nn'] = df.groupby(['id', 'quarter']).cumcount() + 1
    df = df[df['nn'] == 1].copy()

    n_incl = df['included'].sum()
    n_excl = (df['included'] == 0).sum()
    print(f"  After keeping 1 month/quarter: {len(df):,} person-quarters")
    print(f"  Included (consistent welfare): {n_incl:,}  "
          f"Excluded (mixed welfare): {n_excl:,}  "
          f"P(included) = {n_incl / len(df):.3f}")

    # -- 10. Propensity-score logit on filtered person-level sample
    avail_psc   = [v for v in PSCORE_VARS if v in df.columns]
    missing_psc = [v for v in PSCORE_VARS if v not in df.columns]
    if missing_psc:
        print(f"  [pscore] {len(missing_psc)} vars absent — skipping: "
              f"{missing_psc[:5]}{'...' if len(missing_psc) > 5 else ''}")

    person_df  = df.drop_duplicates(subset=['id']).copy()   # one row per person
    person_ids = person_df['id'].to_numpy()
    person_e   = person_df['e'].to_numpy(dtype=int)
    person_X   = person_df[avail_psc].fillna(0).to_numpy(dtype=float)
    N_persons  = len(person_ids)

    print(f"  Fitting pscore logit on {N_persons:,} filtered persons "
          f"({len(avail_psc)} vars) — statsmodels Newton-Raphson ...")
    pscorewt_person = fit_pscore_logit(person_ids, person_e, person_X)
    id_to_wt = dict(zip(person_ids, pscorewt_person))
    df['pscorewt'] = df['id'].map(id_to_wt)
    wt_lo, wt_hi = float(pscorewt_person.min()), float(pscorewt_person.max())
    print(f"  pscorewt range: [{wt_lo:.4f}, {wt_hi:.4f}]")

    # -- 11. Keep included person-quarters only
    df_incl = df[(df['included'] == 1) & df['ebin'].notna()].copy()
    df_incl['ebin']   = df_incl['ebin'].astype(int)
    df_incl['partic'] = df_incl['partic'].astype(int)

    # -- 12. Map to A-matrix row index; -1 for reference states (b1p..b5p)
    df_incl['state_row'] = df_incl.apply(
        lambda r: STATE_ROWS.get((int(r['ebin']), int(r['partic'])), -1), axis=1
    )

    # Sanity check: every included obs must be in a valid state
    bad = df_incl[df_incl['state_row'].isna()]
    assert len(bad) == 0, \
        f"{len(bad)} included obs could not be mapped to any state!"

    # -- 13. Count observations per granular state
    print(f"\n  Final dataset: {len(df_incl):,} person-quarters from "
          f"{df_incl['id'].nunique():,} persons")
    print(f"  JF (treated):   {(df_incl['e'] == 1).sum():,} person-quarters")
    print(f"  AFDC (control): {(df_incl['e'] == 0).sum():,} person-quarters")
    print("\n  Granular state distribution (non-reference):")
    for (eb, pa), row_idx in sorted(STATE_ROWS.items()):
        n = int(((df_incl['ebin'] == eb) & (df_incl['partic'] == pa)).sum())
        suffix = f"b{eb}{'p' if pa else 'n'}" if eb > 0 else f"0{'p' if pa else 'n'}"
        print(f"    Row {row_idx:2d} ({suffix:4s}): {n:6,}")
    n_ref = int((df_incl['state_row'] == -1).sum())
    print(f"    Reference (b1p..b5p):   {n_ref:6,}")

    # -- 14. Extract arrays
    D          = df_incl['e'].to_numpy(dtype=int)
    ebin_arr   = df_incl['ebin'].to_numpy(dtype=int)
    partic_arr = df_incl['partic'].to_numpy(dtype=int)
    state_row  = df_incl['state_row'].to_numpy(dtype=int)
    person_id  = df_incl['id'].to_numpy()
    pscorewt   = df_incl['pscorewt'].to_numpy(dtype=float)

    cov_vars = [v for v in cfg['covariate_vars'] if v in df_incl.columns]
    print(f"  Covariates: {len(cov_vars)}")

    # -- 15. IPW-adjusted state frequencies
    print("\n  -- IPW-adjusted granular state frequencies --")
    for label, gmask in [("JF", D == 1), ("AFDC", D == 0)]:
        w     = pscorewt[gmask]
        eb_g  = ebin_arr[gmask]
        pa_g  = partic_arr[gmask]
        w_sum = w.sum() if w.sum() > 0 else 1.0
        parts = []
        for (eb, pa), row_idx in sorted(STATE_ROWS.items()):
            suf = f"b{eb}{'p' if pa else 'n'}" if eb > 0 else f"0{'p' if pa else 'n'}"
            freq = float(np.sum(((eb_g == eb) & (pa_g == pa)).astype(float) * w) / w_sum)
            if freq > 0.005:
                parts.append(f"{suf}={freq:.4f}")
        print(f"    {label:5s} (N={gmask.sum():,}):  " + "  ".join(parts))
    print()

    return D, ebin_arr, partic_arr, state_row, df_incl, person_id, pscorewt


# =============================================================================
# SECTION 4 — FEATURE ENGINEERING
# =============================================================================
COV_VARS = JF_CONFIG['covariate_vars']


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

    # -- Squared terms: meaningful continuous variables
    ern_sq   = ern ** 2
    _add(ern_sq,
         [f'ernpq{k+1}^2' for k in range(n_q)],
         "sq_ern (meaningful)")
    _add(kidcount[:, None] ** 2,       ["kidcount^2"],      "sq_kidcount")
    _add(yremp[:, None]    ** 2,       ["yremp^2"],         "sq_yremp")

    # -- Squared terms: near-binary AFDC — drop column if truly binary
    # If adcpqk takes only values in {0, 1} then adcpqk^2 == adcpqk (perfect
    # collinearity with the base feature).  We check each column individually
    # and only keep those with > 2 unique values (genuinely continuous).
    adc_sq_cols  = []
    adc_sq_names = []
    n_dropped_adc = 0
    for k in range(n_q):
        col = adc[:, k]
        n_uniq = len(np.unique(col))
        if n_uniq <= 2:
            # Binary: x^2 == x — drop to avoid perfect collinearity
            n_dropped_adc += 1
        else:
            adc_sq_cols.append(col ** 2)
            adc_sq_names.append(f'adcpq{k+1}^2')
    if adc_sq_cols:
        _add(np.column_stack(adc_sq_cols), adc_sq_names,
             f"sq_adc (kept {len(adc_sq_cols)}/{n_q}, {n_dropped_adc} binary dropped)")
    else:
        print(f"    sq_adc: all {n_q} adcpq columns are binary -> all dropped")

    # -- T1a: Earnings x Earnings  (all unordered pairs, k < l)
    ee_block = []
    ee_names = []
    for k, l in combinations(range(n_q), 2):
        ee_block.append(ern[:, k] * ern[:, l])
        ee_names.append(f'ernpq{k+1}xernpq{l+1}')
    _add(np.column_stack(ee_block), ee_names, "T1a: Ern x Ern")

    # -- T1b: AFDC x AFDC  (all unordered pairs, k < l)
    aa_block = []
    aa_names = []
    for k, l in combinations(range(n_q), 2):
        aa_block.append(adc[:, k] * adc[:, l])
        aa_names.append(f'adcpq{k+1}xadcpq{l+1}')
    _add(np.column_stack(aa_block), aa_names, "T1b: AFDC x AFDC")

    # -- T1c: Earnings x AFDC — same quarter  (k=1..8)
    ea_same_block = []
    ea_same_names = []
    for k in range(n_q):
        ea_same_block.append(ern[:, k] * adc[:, k])
        ea_same_names.append(f'ernpq{k+1}xadcpq{k+1}')
    _add(np.column_stack(ea_same_block), ea_same_names, "T1c: Ern x AFDC same-q")

    # -- T1d: Earnings x AFDC — cross-quarter  (all k!=l ordered pairs)
    ea_cross_block = []
    ea_cross_names = []
    for k in range(n_q):
        for l in range(n_q):
            if k != l:
                ea_cross_block.append(ern[:, k] * adc[:, l])
                ea_cross_names.append(f'ernpq{k+1}xadcpq{l+1}')
    _add(np.column_stack(ea_cross_block), ea_cross_names, "T1d: Ern x AFDC cross-q")

    # -- T1e: kidcount x earnpqk  (k=1..8)
    _add(ern * kidcount[:, None],
         [f'kidcountxernpq{k+1}' for k in range(n_q)],
         "T1e: kidcount x Ern")

    # -- T1f: kidcount x adcpqk  (k=1..8)
    _add(adc * kidcount[:, None],
         [f'kidcountxadcpq{k+1}' for k in range(n_q)],
         "T1f: kidcount x AFDC")

    # -- T1g: yngchtru x earnpqk  (k=1..8)
    _add(ern * yngchtru[:, None],
         [f'yngchtruxernpq{k+1}' for k in range(n_q)],
         "T1g: yngchtru x Ern")

    # -- T1h: applcant x earnpqk  (k=1..8)
    _add(ern * applcant[:, None],
         [f'applcantxernpq{k+1}' for k in range(n_q)],
         "T1h: applcant x Ern")

    # -- T2a: age2534 x earnpqk  (k=1..8)
    _add(ern * age2534[:, None],
         [f'age2534xernpq{k+1}' for k in range(n_q)],
         "T2a: age2534 x Ern")

    # -- T2b: hsged x earnpqk  (k=1..8)
    _add(ern * hsged[:, None],
         [f'hsgedxernpq{k+1}' for k in range(n_q)],
         "T2b: hsged x Ern")

    # -- T2c: nohsged x earnpqk  (k=1..8)
    _add(ern * nohsged[:, None],
         [f'nohsgedxernpq{k+1}' for k in range(n_q)],
         "T2c: nohsged x Ern")

    # -- T2d: hsged x adcpqk  (k=1..8)
    _add(adc * hsged[:, None],
         [f'hsgedxadcpq{k+1}' for k in range(n_q)],
         "T2d: hsged x AFDC")

    # -- T2e: nohsged x adcpqk  (k=1..8)
    _add(adc * nohsged[:, None],
         [f'nohsgedxadcpq{k+1}' for k in range(n_q)],
         "T2e: nohsged x AFDC")

    # -- T2f: yremp x earnpqk  (k=1..8)
    _add(ern * yremp[:, None],
         [f'yrempxernpq{k+1}' for k in range(n_q)],
         "T2f: yremp x Ern")

    # -- T2g: yremp x adcpqk  (k=1..8)
    _add(adc * yremp[:, None],
         [f'yrempxadcpq{k+1}' for k in range(n_q)],
         "T2g: yremp x AFDC")

    # -- T2h: kidcount x yngchtru  (1 term)
    _add((kidcount * yngchtru)[:, None],
         ["kidcountxyngchtru"],
         "T2h: kidcount x yngchtru")

    X_econ_raw = np.column_stack(blocks)   # (N, M)
    feat_names = avail + names

    n_sq_meaningful = n_q + 2                  # ernpq^2*n_q + kidcount^2 + yremp^2
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
    print(f"        T1a Ern x Ern:            {len(ee_block):>4}")
    print(f"        T1b AFDC x AFDC:          {len(aa_block):>4}")
    print(f"        T1c Ern x AFDC same-q:    {len(ea_same_block):>4}")
    print(f"        T1d Ern x AFDC cross-q:   {len(ea_cross_block):>4}")
    print(f"        T1e-h scalar x Ern/AFDC: {4*n_q:>4}")
    print(f"      Tier 2 interactions:        {n_t2:>4}")
    print(f"      Tier 3 collinear (excl.xexcl.): dropped entirely")
    print(f"      Extra total:                {X_econ_raw.shape[1]:>4}")
    print(f"      Grand total (base+econ):    {X_base_raw.shape[1] + X_econ_raw.shape[1]:>4}")

    return X_base_raw, X_econ_raw, feat_names, group_info


# =============================================================================
# 2b.  RF IMPORTANCE-BASED FEATURES
# =============================================================================
def engineer_features_rf(df_incl, D, state_row, pscorewt,
                          cov_vars=COV_VARS, n_top=10, seed=42):
    """
    RF-importance-based feature construction for the granular model.
    Uses compute_B_granular for B-scores.
    """
    B_obs = compute_B_granular(D, state_row, pscorewt)
    k     = B_obs.shape[1]   # N_STATES = 13

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
    print(f"  [RF] Generated {len(blocks)} interactions (C({n_top},2))")
    return X_rf_extra


# =============================================================================
# 2c.  DISCRETIZED FEATURES
# =============================================================================
def engineer_features_disc(df_incl, cov_vars=COV_VARS):
    """
    Quartile bin dummies for continuous covariates.
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

    X_disc_extra = np.column_stack(blocks) if blocks else np.zeros((len(df), 0))
    print(f"  [Disc] Generated {len(blocks)} quartile bin dummies")
    return X_disc_extra


# =============================================================================
# SECTION 5 — FOLD-LEVEL FEATURE BUILDER
# =============================================================================
def _build_fold_features(feature_set, X_base_raw, X_econ_raw,
                          train_idx, test_idx,
                          X_rf_raw=None, X_disc_raw=None):
    """
    Build standardised feature matrices for one CV fold.
    "base": 28 base covariates.
    "econ": base + econ interactions.
    "rf":   base + RF-importance interactions.
    "disc": base + quartile bin dummies.
    Scaling fit on train only — no leakage.
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
# SECTION 6 — MODEL FACTORY
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
        return None  # handled specially in estimate_b0_features_granular
    else:
        raise ValueError(f"Unknown estimator: '{estimator_name}'")


# =============================================================================
# SECTION 7 — B-SCORES
# =============================================================================
def compute_B_granular(D, state_row, pscorewt):
    """
    Compute B-score matrix: B_i^s = 1{S_i=s} x w_i
    where w_i = (2D_i - 1) x pscorewt_i.

    Observations in reference states (state_row == -1) contribute 0 to every
    B-score component.

    Returns
    -------
    B : (N, 13) float array   — one column per non-reference state (A matrix row)
    """
    w = (2.0 * D - 1.0) * pscorewt
    N = len(D)
    B = np.zeros((N, N_STATES), dtype=float)
    for s in range(N_STATES):
        B[:, s] = (state_row == s) * w
    return B


# =============================================================================
# SECTION 8 — SOURCE PROBS
# =============================================================================
def compute_source_probs(D, ebin_arr, partic_arr, pscorewt):
    """
    Compute IPW-adjusted probability of being in each state, separately for
    treatment and control groups.

    Returns
    -------
    source_prob_c : (N_BETA,) float  — P(source state | control, IPW-adjusted)
    source_prob_t : (N_BETA,) float  — P(source state | treatment, IPW-adjusted)
    state_prob_c  : dict (ebin, partic) -> probability in control
    """
    ctrl, treat = (D == 0), (D == 1)
    wc, wt      = pscorewt[ctrl], pscorewt[treat]
    wc_sum, wt_sum = wc.sum(), wt.sum()

    state_prob_c = {}
    state_prob_t = {}
    for (eb, pa) in STATE_ROWS:
        mask_c = ctrl & (ebin_arr == eb) & (partic_arr == pa)
        mask_t = treat & (ebin_arr == eb) & (partic_arr == pa)
        state_prob_c[(eb, pa)] = float(pscorewt[mask_c].sum() / wc_sum)
        state_prob_t[(eb, pa)] = float(pscorewt[mask_t].sum() / wt_sum)
    # Also add reference states (ebin 1..5, partic=1)
    for eb in range(1, 6):
        for pa in [1]:
            mask_c = ctrl & (ebin_arr == eb) & (partic_arr == pa)
            mask_t = treat & (ebin_arr == eb) & (partic_arr == pa)
            state_prob_c[(eb, pa)] = float(pscorewt[mask_c].sum() / wc_sum)
            state_prob_t[(eb, pa)] = float(pscorewt[mask_t].sum() / wt_sum)

    source_prob_c = np.array([state_prob_c.get(s, 0.0) for s in BETA_SOURCE_STATE])
    source_prob_t = np.array([state_prob_t.get(s, 0.0) for s in BETA_SOURCE_STATE])

    print("\n  Granular state probabilities (IPW-adjusted, control group):")
    for (eb, pa), prob in sorted(state_prob_c.items()):
        suffix = f"b{eb}{'p' if pa else 'n'}" if eb > 0 else f"0{'p' if pa else 'n'}"
        if prob > 0.001:
            print(f"    ({eb},{pa}) {suffix:4s}: {prob:.4f}")

    return source_prob_c, state_prob_c


# =============================================================================
# SECTION 9 — CROSS-FITTED FIRST STAGE WITH ECON FEATURES
# =============================================================================
def estimate_b0_features_granular(feature_set, X_base_raw, X_econ_raw,
                                   B_obs, D, state_row,
                                   estimator_name,
                                   pscorewt, K=K_FOLDS, seed=RANDOM_SEED,
                                   person_id=None,
                                   X_rf_raw=None, X_disc_raw=None):
    """
    K-fold cross-fitted first stage for granular 13-component B-scores.
    Uses _build_fold_features() for within-fold scaling (no global scaling).

    When person_id is provided, uses GroupKFold so all quarters of the same
    person fall in the same fold (eliminates within-person covariate leakage).
    Falls back to KFold if person_id is None.

    Parameters
    ----------
    feature_set    : "base" (28), "econ" (255), "rf", or "disc"
    estimator_name : "LASSO", "Ridge", "OLS", or "PostLasso"
    person_id      : (N,) array of person identifiers, or None
    """
    n, k  = B_obs.shape   # k = N_STATES = 13
    b_hat = np.zeros((n, k))

    idx_arr = np.arange(n)

    if person_id is not None:
        kf    = GroupKFold(n_splits=K)
        folds = list(kf.split(idx_arr, groups=person_id))
    else:
        kf    = KFold(n_splits=K, shuffle=True, random_state=seed)
        folds = list(kf.split(idx_arr))

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

    if person_id is not None:
        n_persons = len(np.unique(person_id))
        print(f"  [{estimator_name}/{feature_set}] GroupKFold ({K} folds, "
              f"{n_persons:,} persons, {n:,} obs, {dim} features, {k} B-components) "
              f"— within-fold scaling")
    else:
        print(f"  [{estimator_name}/{feature_set}] KFold ({K} folds, {n:,} obs, "
              f"{dim} features, {k} B-components) — within-fold scaling")

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_tr, X_te = _build_fold_features(
            feature_set, X_base_raw, X_econ_raw, train_idx, test_idx,
            X_rf_raw=X_rf_raw, X_disc_raw=X_disc_raw,
        )
        B_tr = B_obs[train_idx]
        print(f"    Fold {fold_i+1}/{K} ...", end="  ", flush=True)
        for j in range(k):
            y = B_tr[:, j]
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
        print("components done")

    b_mean = B_obs.mean(axis=0)
    b_fit  = b_hat.mean(axis=0)
    print(f"  Sample mean B (true b0): {b_mean.round(4)}")
    print(f"  First-stage mean b_hat:  {b_fit.round(4)}")
    print(f"  Max |bias|:              {np.max(np.abs(b_mean - b_fit)):.2e}")
    return b_hat, B_obs


# =============================================================================
# SECTION 10 — LP SOLVER
# =============================================================================

# Universal feasible fallback for any q = +-e_j.
_FALLBACK_NU       = np.full(N_STATES, -1.0)
_FALLBACK_NU[9]    = -2.0   # row 9 = state 0p (the "hub" outflow state)


def _clp_solve_lp(c, A_ub_neg, b_ub_neg, bounds):
    """
    Solve min c'nu  s.t.  (-A')nu <= (-q)  [i.e. A'nu >= q]  for one observation.

    Parameters
    ----------
    c         : (N_STATES,) float  — first-stage prediction b_hat_0(X_i)
    A_ub_neg  : (N_BETA, N_STATES) = -A.T  — pre-built, fixed across observations
    b_ub_neg  : (N_BETA,)          = -q    — pre-built, fixed for a given q
    bounds    : list of (lo, hi) per nu component

    Returns
    -------
    nu   : (N_STATES,) float  — optimal dual vector
    fval : float              — optimal objective value
    """
    res = linprog(
        c=c,
        A_ub=A_ub_neg,
        b_ub=b_ub_neg,
        bounds=bounds,
        method='highs',
        options={'disp': False},
    )
    if res.status == 0:
        return res.x, float(res.fun)
    # Fallback: nu = (-1,...,-1) is always feasible for q = +-e_j
    return _FALLBACK_NU.copy(), float(_FALLBACK_NU @ c)


def clp_estimate_granular(q, b_hat, B_obs, A_mat, verbose=False):
    """
    CLP plug-in estimator using per-observation LP solve.

    sigma_hat(q) = (1/N) sum_i  nu_hat_i' B_i
    where  nu_hat_i = argmin_{A'nu >= q} nu' b_hat_0(X_i)

    Parameters
    ----------
    q      : (N_BETA,) float         — direction vector (e_j or -e_j)
    b_hat  : (N, N_STATES) float     — cross-fitted first-stage predictions
    B_obs  : (N, N_STATES) float     — observed B-scores
    A_mat  : (N_STATES, N_BETA)

    Returns
    -------
    sigma    : float              — sigma_hat(q)
    contribs : (N,) float        — individual contributions nu_hat_i'B_i
    nu_sel   : (N, N_STATES)     — selected dual vectors (for diagnostics)
    """
    n        = len(b_hat)
    # Pre-build constraint matrices — identical for every observation at fixed q
    A_ub_neg = -A_mat.T                          # (N_BETA, N_STATES)
    b_ub_neg = -np.asarray(q, dtype=float)       # (N_BETA,)
    bounds   = [(-5.0, 5.0)] * N_STATES          # box keeps LP bounded

    contribs = np.zeros(n)
    nu_sel   = np.zeros((n, N_STATES))

    for i in range(n):
        nu_i, _      = _clp_solve_lp(b_hat[i], A_ub_neg, b_ub_neg, bounds)
        nu_sel[i]    = nu_i
        contribs[i]  = nu_i @ B_obs[i]

    return contribs.mean(), contribs, nu_sel


# =============================================================================
# SECTION 11 — BOOTSTRAP
# =============================================================================
def multiplier_bootstrap_ci(contribs, n_bs=N_BOOTSTRAP, alpha=0.05,
                             person_id=None):
    """
    Multiplier (exponential) bootstrap confidence interval, clustered by person.
    Re-weights the pre-computed per-observation contributions — no LP re-solves.
    """
    n        = len(contribs)
    sigma_hat= contribs.mean()

    if person_id is not None:
        unique_ids = np.unique(person_id)
        id_to_idx  = {pid: np.where(person_id == pid)[0] for pid in unique_ids}
        bs = np.zeros(n_bs)
        for b in range(n_bs):
            wts   = np.ones(n)
            exp_w = np.random.exponential(1.0, size=len(unique_ids))
            for c, pid in enumerate(unique_ids):
                wts[id_to_idx[pid]] = exp_w[c]
            bs[b] = (contribs * wts).mean()
    else:
        bs = np.array([
            (contribs * np.random.exponential(1.0, n)).mean()
            for _ in range(n_bs)
        ])

    bs_stat = np.sqrt(n) * (bs - sigma_hat)
    q_lo    = np.quantile(bs_stat, alpha / 2)
    q_hi    = np.quantile(bs_stat, 1 - alpha / 2)
    return sigma_hat - q_hi / np.sqrt(n), sigma_hat - q_lo / np.sqrt(n)


# =============================================================================
# SECTION 11b — COMPOSITE BOUNDS (KT Bounds.m, lines 525-700)
# =============================================================================
#
# KT Table 5 reports composite linear-combination bounds in addition to the
# 9 individual response probabilities.  In the granular extension we generalise
# the same three composites to the 33-column model:
#
#   (A) Take-Up Work    : not-working source -> working destination
#       In granular, "not-working" sources are 0n and 0r=0p (zero earnings).
#       "Working" destinations are anything except 0n / 0p / stay.
#       q_beta = 1 for cols whose (source, dest) = (not working, working),
#                0 otherwise; divided by P^a(0n) + P^a(0p) = p00_c + p01_c.
#       In our 33-col layout this means cols in Groups 1, 4, 5, 6, 8 = 1.
#
#   (B) Take-Up Welfare : off-welfare source -> on-welfare destination
#       Sources off-welfare: 0n, b_jn (j=1..5), b_kn (k=6..8).
#       Destinations on-welfare: b_jr (1r) and b_ku (2u underreporters).
#       q_beta = 1 for cols whose (source, dest) = (off, on); divided by
#       P^a(0n) + sum_j P^a(b_jn) + sum_k P^a(b_kn) = p00_c + p10_c + p20_c
#       (the COARSE off-welfare marginals from Table 4 -- granular sub-bins
#        sum to these by construction).
#       In our 33-col layout: cols in Groups 1, 3, 7 = 1.
#
#   (C) Exit 0r        : 0r source -> off-welfare destination
#       q_beta = 1 for cols whose (source, dest) = (0r, off); divided by
#       P^a(0p) = p01_c.  In our layout: cols in Groups 2, 4, 6 = 1.
#
# These composites are written to MATCH the coarse 5x9 versions exactly --
# every coarse-level coefficient simply expands to its granular descendants
# with the same denominator (the coarse marginals).  This keeps the granular
# composite numerically comparable to the coarse one in CLP_final_group.py.
#
# Implementation: build q_beta in the 33-vector layout, pass to the existing
# clp_estimate_granular() (per-i scipy.linprog) for upper and lower bounds,
# bootstrap for CIs, clip pi-units to [0,1] for display.
# =============================================================================
def _granular_composite_q_vectors(p):
    """
    Return q_beta vectors for the three KT composites, in BETA_NAMES order
    (33 entries).  `p` is the dict from load_table4_mat (coarse marginals).
    """
    p00_c = p['p00_c']; p01_c = p['p01_c']
    p10_c = p['p10_c']; p20_c = p['p20_c']

    # ---------- (A) Take-Up Work ----------
    # Cols in Groups 1, 4, 5, 6, 8 = 1; others 0; / (p00_c + p01_c)
    q_TUW = np.zeros(N_BETA, dtype=float)
    q_TUW[ 0: 5] = 1.0    # G1: beta(0n,b_jr)         (5 cols)
    q_TUW[ 9:12] = 1.0    # G4: beta(0r,b_kn)         (3 cols)
    q_TUW[12:17] = 1.0    # G5: beta(0r,b_jr)         (5 cols)
    q_TUW[17:22] = 1.0    # G6: beta(0r,b_jn)         (5 cols)
    q_TUW[27:30] = 1.0    # G8: beta(0r,b_ku)         (3 cols)
    q_TUW /= (p00_c + p01_c)

    # ---------- (B) Take-Up Welfare ----------
    # Cols in Groups 1, 3, 7 = 1; others 0; / (p00_c + p10_c + p20_c)
    q_TUWelf = np.zeros(N_BETA, dtype=float)
    q_TUWelf[ 0: 5] = 1.0    # G1: beta(0n,b_jr)      (5 cols)
    q_TUWelf[ 6: 9] = 1.0    # G3: beta(b_kn,1r)      (3 cols)
    q_TUWelf[22:27] = 1.0    # G7: beta(b_jn,1r)      (5 cols)
    q_TUWelf /= (p00_c + p10_c + p20_c)

    # ---------- (C) Exit 0r ----------
    # Cols in Groups 2, 4, 6 = 1; others 0; / p01_c
    q_Exit = np.zeros(N_BETA, dtype=float)
    q_Exit[ 5]      = 1.0    # G2: beta(0r,0n)        (1 col)
    q_Exit[ 9:12]   = 1.0    # G4: beta(0r,b_kn)      (3 cols)
    q_Exit[17:22]   = 1.0    # G6: beta(0r,b_jn)      (5 cols)
    q_Exit /= p01_c

    return {
        "Take-Up Work (not working -> working)":      q_TUW,
        "Take-Up Welfare (off -> on welfare)":         q_TUWelf,
        "Exit 0r (on-welfare zero earn -> off welfare)": q_Exit,
    }


def compute_composite_bounds_granular(b_hat, B_obs, A_mat, person_id, p,
                                       label=""):
    """
    KT-style composite bounds for the 33-column granular model, computed via
    the same clp_estimate_granular machinery used for individual betas.

    Returns dict {composite_name: result_dict}.
    """
    qs = _granular_composite_q_vectors(p)
    results = {}

    print(f"\n  --- COMPOSITE BOUNDS ({label}) ---")
    print(f"  Granular composites in pi-units (numerator: sum of beta-cols whose")
    print(f"  source/destination pattern matches; denominator: AFDC source-state mass):")
    print(f"  {'composite':<55}  {'LB_pi':>8}  {'UB_pi':>8}  "
          f"{'width':>8}    {'95% CI':>22}")
    print("  " + "-" * 110)

    for name, q_beta in qs.items():
        ub_hat, c_up, _ = clp_estimate_granular( q_beta, b_hat, B_obs, A_mat)
        ci_ub = list(multiplier_bootstrap_ci(
            c_up, n_bs=N_BOOTSTRAP, person_id=person_id))
        neg_lb, c_dn, _ = clp_estimate_granular(-q_beta, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb
        ci_lb = [-x for x in
                 multiplier_bootstrap_ci(
                     c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]

        # Clip to [0,1] (theta_M is a probability).
        lb_c  = max(0.0, min(1.0, lb_hat))
        ub_c  = max(0.0, min(1.0, ub_hat))
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
# SECTION 12 — CLP ESTIMATION LOOP
# =============================================================================
def run_clp_granular(b_hat, B_obs, A_mat, person_id, source_prob_c):
    """
    Run CLP bounds for all N_BETA=33 beta parameters.

    For each beta_j:
      UB = sigma_hat( e_j)
      LB = -sigma_hat(-e_j)

    Back-converts to pi_j = beta_j / P^a(source_j).

    Returns dict {beta_name: {'lb', 'ub', 'width', 'ci_lb', 'ci_ub'}}.
    """
    import time

    n_beta = A_mat.shape[1]   # 33
    results = []

    print(f"\n{'='*76}")
    print("  CLP ESTIMATION  —  granular 9-bin income model  (33 beta parameters)")
    print(f"{'='*76}")
    print(f"\n  Note: each beta requires 2xN LP solves (UB + LB) + bootstrap reweighting.")
    print(f"  Expect ~1-3 min per beta parameter; ~30-100 min total.\n")
    print(f"  {'beta param':<18}  {'LB':>7}  {'UB':>7}  {'Width':>7}  "
          f"{'95% CI LB':>14}  {'95% CI UB':>14}  {'time':>6}")
    print("  " + "-" * 82)

    t_all = time.time()

    for j in range(n_beta):
        t0 = time.time()

        q_up = np.zeros(n_beta); q_up[j] =  1.0
        q_dn = np.zeros(n_beta); q_dn[j] = -1.0

        # Upper bound: sigma_hat(e_j)
        ub_hat, c_up, _ = clp_estimate_granular(q_up, b_hat, B_obs, A_mat)
        ci_ub = list(multiplier_bootstrap_ci(
            c_up, n_bs=N_BOOTSTRAP, person_id=person_id
        ))

        # Lower bound: -sigma_hat(-e_j)
        neg_lb, c_dn, _ = clp_estimate_granular(q_dn, b_hat, B_obs, A_mat)
        lb_hat = -neg_lb
        ci_lb  = [-x for x in
                  multiplier_bootstrap_ci(
                      c_dn, n_bs=N_BOOTSTRAP, person_id=person_id
                  )[::-1]]

        # Clip to [0, 1]
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

        elapsed = time.time() - t0
        print(f"  {BETA_NAMES[j]:<18}  {lb_hat:7.4f}  {ub_hat:7.4f}  "
              f"{width:7.4f}  "
              f"  [{ci_lb[0]:.3f},{ci_lb[1]:.3f}]  "
              f"  [{ci_ub[0]:.3f},{ci_ub[1]:.3f}]  "
              f"{elapsed:5.1f}s")

    print(f"\n  Total CLP estimation time: {(time.time()-t_all)/60:.1f} min")

    # -- beta -> pi back-conversion
    print(f"\n{'─'*76}")
    print("  beta -> pi  (divide by P^a of source state in control group, IPW-adj.)")
    print(f"{'─'*76}")
    print(f"\n  {'pi param':<18}  {'P^a(src)':>9}  {'LB_pi':>8}  {'UB_pi':>8}  "
          f"{'CI LB_pi':>12}  {'CI UB_pi':>12}")
    print("  " + "-" * 70)

    def _clip(x):
        return float(min(1.0, max(0.0, x)))

    for j, res in enumerate(results):
        sp = source_prob_c[j]
        if sp <= 1e-12:
            print(f"  {PI_NAMES[j]:<18}  {'N/A':>9}  (source prob ~= 0)")
            continue
        lb_p  = _clip(res['lb'] / sp)
        ub_p  = _clip(res['ub'] / sp)
        cllo  = _clip(res['ci_lb'][0] / sp)
        clhi  = _clip(res['ci_lb'][1] / sp)
        culo  = _clip(res['ci_ub'][0] / sp)
        cuhi  = _clip(res['ci_ub'][1] / sp)
        print(f"  {PI_NAMES[j]:<18}  {sp:9.4f}  {lb_p:8.4f}  {ub_p:8.4f}  "
              f"  [{cllo:.4f},{clhi:.4f}]  [{culo:.4f},{cuhi:.4f}]")

    return {r['name']: r for r in results}


# =============================================================================
# SECTION 13 — SUMMARY
# =============================================================================
def print_summary(results):
    """Print interval widths grouped by the original coarse beta parameter."""
    beta_list = list(results.values())

    # Corresponds to the 9 groups in BETA_NAMES / A matrix column structure
    groups = [
        ("beta(0n,.r)  [orig beta(0n,1r)x5]",    list(range( 0,  5))),
        ("beta(0r,0n)  [unchanged]",               [5]),
        ("beta(b6-8n,1r) [orig beta(2n,1r)x3]",   list(range( 6,  9))),
        ("beta(0r,b6-8n) [orig beta(0r,2n)x3]",   list(range( 9, 12))),
        ("beta(0r,.r)  [orig beta(0r,1r)x5]",     list(range(12, 17))),
        ("beta(0r,b1-5n) [orig beta(0r,1n)x5]",   list(range(17, 22))),
        ("beta(b1-5n,1r) [orig beta(1n,1r)x5]",   list(range(22, 27))),
        ("beta(0r,b6-8u) [orig beta(0r,2u)x3]",   list(range(27, 30))),
        ("beta(b6-8u,1r) [orig beta(2u,1r)x3]",   list(range(30, 33))),
    ]

    print(f"\n\n{'='*76}")
    print("SUMMARY: granular beta identified intervals  [UB - LB]")
    print("(each group corresponds to one of the original 9 coarse beta parameters)")
    print(f"{'='*76}")

    all_widths = []
    for grp_name, idxs in groups:
        widths = [beta_list[i]['width'] for i in idxs]
        all_widths.extend(widths)
        print(f"\n  -- {grp_name}")
        for i in idxs:
            r = beta_list[i]
            print(f"     {r['name']:<18}  LB={r['lb']:.4f}  UB={r['ub']:.4f}  "
                  f"Width={r['width']:.4f}")
        print(f"     -> group mean width: {np.mean(widths):.4f}  "
              f"(min={min(widths):.4f}, max={max(widths):.4f})")

    print(f"\n  {'─'*50}")
    print(f"  Overall mean width  : {np.mean(all_widths):.4f}")
    print(f"  Overall median width: {np.median(all_widths):.4f}")
    print(f"\n  NOTE: The granular model has {N_BETA} beta parameters vs 9 in the original.")
    print(f"  Each fine beta is bounded within the coarse beta's identified interval.")
    print(f"  Tighter fine-grained widths -> more information from granular data.")


def print_summary_table(all_results):
    """
    Side-by-side beta interval widths [UB - LB] across all 6 configurations,
    using granular BETA_NAMES (33 parameters) and CONFIGS_TO_RUN.
    Prints interval widths, averages by feature set and estimator,
    and Delta econ vs base for each estimator.
    Also prints D1 stability-of-tightening diagnostics.
    """
    config_names = list(all_results.keys())
    col_w = 13

    print(f"\n\n{'='*72}")
    print("SUMMARY: granular beta interval widths  [UB - LB]  —  all 6 configs")
    print("Narrower = tighter identified set.  (*) = LB > UB (convergence issue).")
    print(f"{'='*72}")

    print(f"\n  {'beta param':<18}  " +
          "  ".join(f"{n[:col_w]:>{col_w}}" for n in config_names))
    print("  " + "-" * (18 + (col_w + 2) * len(config_names) + 2))

    any_neg = False
    for beta in BETA_NAMES:
        row = f"  {beta:<18}  "
        cells = []
        for name in config_names:
            res = all_results[name].get(beta)
            if res is None:
                cells.append(f"{'N/A':>{col_w}}")
            else:
                w = res['width']
                if w < -1e-6:
                    any_neg = True
                    cells.append(f"{'(*)'+ f'{w:.3f}':>{col_w}}")
                else:
                    cells.append(f"{w:>{col_w}.4f}")
        print(row + "  ".join(cells))

    print("  " + "-" * (18 + (col_w + 2) * len(config_names) + 2))
    means = []
    for name in config_names:
        widths = [v['width'] for k, v in all_results[name].items() if k != '__composite__']
        means.append(np.mean(widths))
    print(f"  {'Mean width':<18}  " +
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
        note = " (no regularization — benchmark only)" if est == "OLS" else ""
        print(f"    {est:<14}:  mean width = {avg:.4f}{note}")

    # — Delta econ vs base —
    est_list = ["LASSO", "Ridge", "OLS", "PostLasso"]
    print(f"\n  Width change: econ vs base (negative = tighter, positive = wider)")
    print(f"  {'beta param':<18}  " +
          "  ".join(f"{'D '+est[:7]:>13}" for est in est_list))
    print("  " + "-" * (18 + 15 * len(est_list) + 2))
    for beta in BETA_NAMES:
        row = f"  {beta:<18}  "
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
    print("  " + "-" * (18 + 15 * len(est_list) + 2))
    row = f"  {'Mean D':<18}  "
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

    # D1: Stability of tightening
    print(f"\n  [D1] Stability of tightening (Delta_r = width per estimator):")
    print(f"  {'beta param':<18}  {'LASSO':>8}  {'Ridge':>8}  {'OLS':>8}  {'PostLasso':>10}  "
          f"{'mean':>8}  {'std':>8}  {'sf':>4}")
    print(f"  {'─'*80}")
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
        std_w  = np.std(arr) if len(arr) > 1 else float('nan')
        sf     = int(np.sum(np.array(arr) < 0)) if arr else 0
        flag   = "  *" if sf > 0 else ""
        wl_s = f"{wl:8.4f}" if not np.isnan(wl) else "     nan"
        wr_s = f"{wr:8.4f}" if not np.isnan(wr) else "     nan"
        wo_s = f"{wo:8.4f}" if not np.isnan(wo) else "     nan"
        wp_s = f"{wp:10.4f}" if not np.isnan(wp) else "       nan"
        print(f"  {beta:<18}  {wl_s}  {wr_s}  {wo_s}  {wp_s}  "
              f"{mean_w:8.4f}  {std_w:8.4f}  {sf:4d}{flag}")

    print(f"\n  NOTE: K={K_FOLDS}-fold GroupKFold by person, N_BOOTSTRAP={N_BOOTSTRAP}.")
    print(f"  GroupKFold eliminates within-person covariate leakage across folds.")
    print(f"  OLS + econ (255 features): no regularization — high-variance benchmark.")
    if any_neg:
        print("\n  (*) Negative width = LB > UB.  First-stage convergence failure.")
        print("      Do NOT interpret as valid bounds.")


# =============================================================================
# DIAGNOSTICS MODULE (LP-based, granular 9-bin)
# =============================================================================

def diag_firstsage_r2(B_obs, b_hat):
    """Out-of-fold R^2 per B-component."""
    k = B_obs.shape[1]
    r2 = np.full(k, np.nan)
    for j in range(k):
        ss_tot = np.sum((B_obs[:, j] - B_obs[:, j].mean()) ** 2)
        ss_res = np.sum((B_obs[:, j] - b_hat[:, j]) ** 2)
        if ss_tot > 1e-10:
            r2[j] = 1.0 - ss_res / ss_tot
    return r2


def diag_vertex_margin_lp(b_hat, A_mat, q, fallback_nu):
    """
    LP-based proxy for vertex-margin.
    Computes score at fallback vertex minus optimal LP score:
    m_approx_i = fallback_nu @ b_hat[i] - opt_lp_i >= 0

    If m_approx_i ~= 0, the LP solution is near the fallback (possible near-tie).
    If m_approx_i >> 0, the LP found a clearly better vertex.

    Note: this is a lower bound on the true margin (the gap to the nearest
    non-optimal vertex could be smaller or larger).
    """
    A_ub_neg = -A_mat.T
    b_ub_neg = -np.asarray(q, dtype=float)
    bounds   = [(-5.0, 5.0)] * A_mat.shape[0]

    n = len(b_hat)
    margins = np.zeros(n)
    fb_score = b_hat @ fallback_nu  # (N,) score at fallback for each obs

    for i in range(n):
        res = linprog(b_hat[i], A_ub=A_ub_neg, b_ub=b_ub_neg, bounds=bounds, method='highs')
        opt_val = float(res.fun) if res.status == 0 else float(fallback_nu @ b_hat[i])
        margins[i] = fb_score[i] - opt_val
    return margins


def diag_binding_vertex_entropy(nu_sel):
    """Binding-vertex distribution entropy."""
    from collections import Counter
    keys = [tuple(np.round(nu_sel[i], 4)) for i in range(len(nu_sel))]
    cnt  = Counter(keys)
    n    = len(keys)
    phat = np.array([v / n for v in cnt.values()])
    H    = float(-np.sum(phat * np.log(phat + 1e-12)))
    return H, len(cnt), float(max(cnt.values()) / n)


# Fallback vertex (always feasible for q = +-e_j): nu[9]=-2 (state 0p hub)
_DIAG_FALLBACK_NU = np.full(N_STATES, -1.0)
_DIAG_FALLBACK_NU[9] = -2.0


# ── D5: Calibration plot ───────────────────────────────────────────────────────
def calibration_plot(b_hat, B_obs, n_bins=10):
    """
    Calibration diagnostic: bin by predicted decile, compare mean pred vs mean obs.
    Slope = 0 → collapsed. Slope = 1 → calibrated. Slope > 1 → overfit.
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
    """OLS slope of obs_means ~ pred_means across bin means."""
    pm = np.asarray(pred_means, dtype=float)
    om = np.asarray(obs_means,  dtype=float)
    if pm.std() < 1e-10:
        return float('nan')
    return float(np.cov(pm, om)[0, 1] / np.var(pm))


# ── D7: Vertex stability test (LP version for granular model) ──────────────────
def vertex_stability_test_granular(b_hat, A_mat, q_vec, noise_frac=0.1, seed=7777):
    """
    Perturb b_hat by noise_frac * component-std, recompute binding dual vector
    via LP for each observation, measure fraction that change significantly.

    Uses the LP solver (_clp_solve_lp) since granular model has too many vertices
    to enumerate.

    < 5%  switch rate → stable first stage.
    > 50% switch rate → first stage landing near boundaries (fragile).
    """
    rng        = np.random.default_rng(seed)
    noise_std  = np.maximum(b_hat.std(axis=0) * noise_frac, 1e-10)
    b_perturbed = b_hat + rng.normal(0.0, noise_std, b_hat.shape)

    A_ub_neg = -A_mat.T
    b_ub_neg = -np.asarray(q_vec, dtype=float)
    bounds   = [(-5.0, 5.0)] * A_mat.shape[0]

    orig_nus = np.array([_clp_solve_lp(b, A_ub_neg, b_ub_neg, bounds)[0]
                         for b in b_hat])
    pert_nus = np.array([_clp_solve_lp(b, A_ub_neg, b_ub_neg, bounds)[0]
                         for b in b_perturbed])

    switch_rate = float(np.mean([
        not np.allclose(o, p, atol=1e-4)
        for o, p in zip(orig_nus, pert_nus)
    ]))
    return switch_rate


# ── D6: Feasibility check ──────────────────────────────────────────────────────
def feasibility_check(b_hat, A_mat):
    """
    For each i, solve min_beta ||A beta - b_hat[i]||_2  s.t. beta >= 0.
    L2 residual > 1e-6 => b_hat[i] outside the feasible image of A.
    """
    residuals = []
    for b in b_hat:
        try:
            res = lsq_linear(A_mat, b, bounds=(0.0, np.inf), method='bvls')
            residuals.append(np.sqrt(max(2.0 * res.cost, 0.0)))
        except Exception:
            residuals.append(np.inf)
    return np.array(residuals)


def print_config_diagnostics_granular(config_name, B_obs, b_hat, A_mat,
                                       beta_names, all_nu_sel_ub,
                                       all_results_config):
    """Print all diagnostics for one configuration (granular LP model)."""
    k      = B_obs.shape[1]
    n_beta = len(beta_names)
    SEP    = "-" * 76

    print(f"\n{SEP}")
    print(f"  DIAGNOSTICS (granular 9-bin): {config_name}")
    print(SEP)

    # D4: R^2 per B-component
    r2 = diag_firstsage_r2(B_obs, b_hat)
    state_labels = [
        "0n", "b1n", "b2n", "b3n", "b4n", "b5n",
        "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"
    ][:k]
    print(f"\n  [D4] First-stage out-of-fold R^2 per B-component:")
    for j, sl in enumerate(state_labels):
        val  = f"{r2[j]:+.4f}" if not np.isnan(r2[j]) else "   nan"
        flag = "  <- WARNING" if (np.isnan(r2[j]) or r2[j] < 0) else ""
        print(f"       B[{sl}]:  R^2={val}{flag}")
    n_neg = int(np.sum(r2 < 0))
    if n_neg > 0:
        print(f"    WARNING: {n_neg}/{k} components R^2<0 — first stage adds no info.")
    red_r2 = n_neg > k // 2

    # D2: LP proxy vertex-margin (sample 10 beta params to keep runtime manageable)
    print(f"\n  [D2] LP vertex-margin proxy (fallback score - optimal LP score):")
    print(f"  (Positive value = LP found better vertex than fallback; near-zero = suspect)")
    sample_betas = list(range(min(10, n_beta)))  # first 10 for speed
    print(f"  {'beta':<18}  {'median':>7}  {'p10':>7}  {'%<0.001':>8}  {'%<0.01':>7}")
    print(f"  {'─'*55}")
    for j in sample_betas:
        q_up = np.zeros(n_beta); q_up[j] = 1.0
        m    = diag_vertex_margin_lp(b_hat, A_mat, q_up, _DIAG_FALLBACK_NU)
        med  = np.median(m); p10 = np.percentile(m, 10)
        pct1 = 100.0 * np.mean(m < 0.001)
        pct2 = 100.0 * np.mean(m < 0.01)
        flag = "  WARNING" if pct1 > 30 else ""
        print(f"  {beta_names[j]:<18}  {med:7.4f}  {p10:7.4f}  {pct1:7.1f}%  {pct2:6.1f}%{flag}")
    if n_beta > 10:
        print(f"  (Showing first 10 of {n_beta} beta params for speed)")

    # D3: Binding-vertex distribution
    print(f"\n  [D3] Binding-vertex distribution (rounded LP solutions, UB direction):")
    print(f"  {'beta':<18}  {'H':>10}  {'n_regions':>9}  {'top_frac':>8}")
    print(f"  {'─'*52}")
    for j in range(min(10, n_beta)):
        H, nv, tf = diag_binding_vertex_entropy(all_nu_sel_ub[j])
        flag = "  WARNING collapse" if tf > 0.90 else ""
        print(f"  {beta_names[j]:<18}  {H:10.4f}  {nv:9d}  {tf:8.4f}{flag}")
    if n_beta > 10:
        print(f"  (Showing first 10 of {n_beta} beta params)")

    # ── D5: Calibration ─────────────────────────────────────────────────────
    # Use up to first 13 state labels for the granular model
    state_labels_g = [f's{j}' for j in range(k)]
    print(f"\n  [D5] Calibration diagnostic (mean pred vs mean obs, 10 decile bins):")
    calib = calibration_plot(b_hat, B_obs)
    print(f"  {'comp':7s}  {'slope':>8}  {'min_pred':>10}  {'max_pred':>10}  interpretation")
    print(f"  {'─'*58}")
    for j in range(min(k, 13)):
        sl = state_labels_g[j]
        pm, om = calib[j]
        slope  = _calibration_slope(pm, om)
        if np.isnan(slope):
            interp = "n/a"
        elif abs(slope) < 0.2:
            interp = "COLLAPSED"
        elif slope > 1.3:
            interp = "OVERFIT"
        elif 0.7 <= slope <= 1.3:
            interp = "calibrated"
        else:
            interp = f"moderate"
        slope_str = f"{slope:8.3f}" if not np.isnan(slope) else "     nan"
        print(f"  B[{sl}]  {slope_str}  {pm.min():10.5f}  {pm.max():10.5f}  {interp}")

    # ── D6: Feasibility ──────────────────────────────────────────────────────
    print(f"\n  [D6] Feasibility check (b_hat in image of A restricted to beta>=0):")
    feas_resid = feasibility_check(b_hat, A_mat)
    frac_inf   = float(np.mean(feas_resid > 1e-6))
    print(f"       Frac infeasible (resid>1e-6):  {frac_inf:.4f}   "
          f"median resid: {np.median(feas_resid):.2e}   "
          f"max resid: {np.max(feas_resid):.2e}")
    if frac_inf > 0.05:
        print(f"       WARNING  {frac_inf*100:.1f}% of b_hat predictions outside feasible set.")

    # ── D7: Vertex stability (LP-based) ─────────────────────────────────────
    # D7: Vertex stability — sample to keep runtime manageable
    sample_size = min(len(b_hat), 500)
    rng_d7 = np.random.default_rng(999)
    sample_idx = rng_d7.choice(len(b_hat), size=sample_size, replace=False)
    b_hat_samp = b_hat[sample_idx]
    print(f"\n  [D7] Vertex stability under 10% noise perturbation "
          f"(LP, N_sample={sample_size}, switch rate):")
    print(f"  {'beta':15s}  {'switch_rate':>12}  interpretation")
    print(f"  {'─'*50}")
    for j, bname in enumerate(beta_names):
        q_up = np.zeros(len(beta_names)); q_up[j] = 1.0
        sr   = vertex_stability_test_granular(b_hat_samp, A_mat, q_up)
        if np.isnan(sr):
            interp = "n/a"
        elif sr < 0.05:
            interp = "stable (<5%)"
        elif sr < 0.20:
            interp = "moderate"
        elif sr < 0.50:
            interp = "fragile  WARNING"
        else:
            interp = "UNSTABLE  WARNING"
        sr_str = f"{sr:12.4f}" if not np.isnan(sr) else "         nan"
        print(f"  {bname:15s}  {sr_str}  {interp}")

    print(f"\n  Combined: {'R^2<0 majority: YES WARNING' if red_r2 else 'R^2<0 majority: no'}")


# =============================================================================
# SECTION 14 — RUN CONFIG
# =============================================================================
def run_config_granular(config_name, feature_set, estimator_name,
                        D, ebin_arr, partic_arr, state_row,
                        X_base_raw, X_econ_raw,
                        person_id, pscorewt, A_mat, source_prob_c,
                        X_rf_raw=None, X_disc_raw=None):
    """Full CLP pipeline for one (feature_set, estimator) configuration."""
    dim = X_base_raw.shape[1] if feature_set == "base" \
          else X_base_raw.shape[1] + X_econ_raw.shape[1]
    print(f"\n{'='*68}")
    print(f"  CONFIG: {config_name}")
    print(f"  feature_set={feature_set} ({dim} features)   estimator={estimator_name}")
    if estimator_name == "OLS" and feature_set == "econ":
        print(f"  OLS + econ ({dim} features): no regularization — benchmark only.")
    print(f"{'='*68}")

    B_obs = compute_B_granular(D, state_row, pscorewt)

    b_hat, B_obs = estimate_b0_features_granular(
        feature_set, X_base_raw, X_econ_raw,
        B_obs, D, state_row,
        estimator_name=estimator_name,
        pscorewt=pscorewt, K=K_FOLDS, seed=RANDOM_SEED,
        person_id=person_id,
        X_rf_raw=X_rf_raw, X_disc_raw=X_disc_raw,
    )

    n_beta = N_BETA
    nu_sel_ub_list = []
    for j in range(n_beta):
        q_up = np.zeros(n_beta); q_up[j] = 1.0
        ub_hat, c_up, nu_up = clp_estimate_granular(q_up, b_hat, B_obs, A_mat)
        nu_sel_ub_list.append(nu_up)

    results = run_clp_granular(b_hat, B_obs, A_mat, person_id, source_prob_c)
    print_summary(results)

    # ── KT-style composite bounds (Take-Up Work / Take-Up Welfare / Exit) ──
    # `p` is loaded once at module/main scope and passed to run_config; we
    # re-load it here cheaply if it isn't in scope.
    try:
        p_for_composite = p
    except NameError:
        p_for_composite, _ = load_table4_mat()
    composite_results = compute_composite_bounds_granular(
        b_hat, B_obs, A_mat, person_id, p_for_composite, label=config_name,
    )

    print_config_diagnostics_granular(config_name, B_obs, b_hat, A_mat,
                                       BETA_NAMES, nu_sel_ub_list,
                                       {r['name']: r for r in results.values()})

    # Attach composite results so the caller can access them; key is namespaced
    # to avoid clashing with beta-name keys.
    results = dict(results)
    results['__composite__'] = composite_results
    return results


# =============================================================================
# SECTION 15 — MAIN
# =============================================================================
def main():
    import time
    import os
    t_start = time.time()

    print("=" * 76)
    print("CLP Granular Final (GroupKFold)  —  6 configurations")
    print("  {base, econ} x {LASSO, Ridge, OLS}")
    print("  GroupKFold cross-fitting: all quarters of the same person held out together")
    print("=" * 76)
    print("Configurations:")
    for name, (fset, est) in CONFIGS_TO_RUN.items():
        print(f"  {name:<18}  features={fset:<6}  estimator={est}")
    print(f"\nK_FOLDS={K_FOLDS}  N_BOOTSTRAP={N_BOOTSTRAP}  SEED={RANDOM_SEED}")
    print(f"States: {N_STATES} non-reference  |  beta params: {N_BETA}")
    print(f"Cross-fitting: GroupKFold by person_id (no within-person leakage)")
    print(f"Within-fold StandardScaler — NO global scaling")

    # -- 1. Prepare granular data
    print("\n\nStep 1 — Prepare granular data (9-bin income classification)")
    print("-" * 40)
    (D, ebin_arr, partic_arr, state_row,
     df_incl, person_id, pscorewt) = prepare_jf_data_granular()

    # -- 2. Feature engineering
    print("\n\nStep 2 — Engineer econ feature set")
    print("-" * 40)
    X_base_raw, X_econ_raw, feat_names, group_info = engineer_features_econ(
        df_incl, cov_vars=JF_CONFIG['covariate_vars']
    )

    print(f"\n  Feature group summary:")
    for label, slc in group_info.items():
        n_feat = slc.stop - slc.start
        print(f"    {label:<50}  {n_feat:>4} features")

    print("\nEngineering RF importance-based feature set ...")
    X_rf_raw = engineer_features_rf(df_incl, D, state_row, pscorewt)

    print("\nEngineering discretized feature set ...")
    X_disc_raw = engineer_features_disc(df_incl)

    # -- 3. Build and verify A matrix
    print("\n\nStep 3 — Build granular A matrix (13 x 33)")
    print("-" * 40)
    A_mat = build_A_granular()
    _verify_A(A_mat)
    print(f"\n  A matrix (first 5 rows, first 12 cols):")
    print("  " + str(A_mat[:5, :12].astype(int)))

    # -- 4. Source probabilities
    print("\n\nStep 4 — Source population probabilities")
    print("-" * 40)
    source_prob_c, state_prob_c = compute_source_probs(
        D, ebin_arr, partic_arr, pscorewt
    )

    # -- 5. Run all configurations
    print(f"\n\nStep 5 — Run all {len(CONFIGS_TO_RUN)} CLP configurations")
    print("-" * 40)
    all_results = {}
    for config_name, (feature_set, estimator_name) in CONFIGS_TO_RUN.items():
        all_results[config_name] = run_config_granular(
            config_name, feature_set, estimator_name,
            D, ebin_arr, partic_arr, state_row,
            X_base_raw, X_econ_raw,
            person_id, pscorewt, A_mat, source_prob_c,
            X_rf_raw=X_rf_raw, X_disc_raw=X_disc_raw,
        )

    # -- 6. Summary table
    print_summary_table(all_results)

    # -- 7. Save results to CSV
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
                            "granular_final_group_results.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f} minutes")
    print("Done.")


if __name__ == "__main__":
    main()
