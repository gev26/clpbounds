# Adaptive Estimation of Aggregated Values of Conditional Linear Programs

This repository includes the replication package of the empirical section and simulation exercise of the paper "Adaptive Estimation of Aggregated Values of Conditional Linear Programs".

The empirical section analyzes the Jobs First Connecticut experiment, analyzed earlier in Kline and Tartari (2016). Their results are used as benchmarks to compare the conditional linear program methodology results to. 

## 1. Methodology

## 2. Different Regimes

Each axis below is varied independently across the files; the full set of
results is a Cartesian product of the relevant axes.

---

### 2.1  Covariate Regimes

The first-stage nuisance b̂₀(X) = Ê[B | X] is fit on a covariate matrix X
whose construction varies across regimes. All regimes use the same 28
base KT covariates as a starting point.

| Regime | # cols | What's included | Files that use it |
|--------|------:|-----------------|--------------------|
| **base** | 28 | The KT covariate set: demographics (age2534, black, hisp, white, marnvr, marapt, hsged, nohsged), kid status (yngchtru, kidcount), 8 pre-baseline quarters of earnings (ernpq1…8), 8 pre-baseline quarters of AFDC receipt (adcpq1…8), applicant flag, prior employment years (yremp). | All CLP files (`CLP_final.py`, `CLP_final_group.py`, `clp_ols.py`, `clp_lasso_appendix.py`, `clp_granular.py`) |
| **econ** | ~255 | base + economically motivated engineered features built by `engineer_features_econ`: squared continuous terms (ernpq²×8, kidcount², yremp², and adcpq² where non-binary), pairwise interactions T1a–T1h (Ern×Ern, AFDC×AFDC, Ern×AFDC same/cross quarter, kidcount/yngchtru/applcant × Ern/AFDC), and demographic-by-economic interactions T2a–T2h (age2534/hsged/nohsged × Ern, hsged/nohsged × AFDC, yremp × Ern/AFDC, kidcount × yngchtru). | Same files as **base** — both feature sets are run in parallel |
| **rf** | base + C(10,2) | base + 45 pairwise interactions among the top-10 base covariates ranked by random-forest importance (averaged across the 5 B-components). | `CLP_final_group.py` only (auxiliary robustness comparator) |
| **disc** | base + ~30 | base + quartile-bin dummies for the continuous targets (ernpq1…8, kidcount, yremp), 3 thresholds each. | `CLP_final_group.py` only |

**Why both base and econ?** Sparsity-inducing first stages (LASSO,
PostLasso) often produce wider bounds on the small base set because they
struggle to recover heterogeneity from raw covariates alone. The econ
set gives them enough basis functions to capture earnings/welfare
nonlinearities without exploding dimensionality.

---

### 2.2  Cross-fitting Regimes

Defines how the first-stage estimator avoids own-observation
contamination of b̂₀(Xᵢ).

| Regime | Split rule | Where person-quarters end up | Files that use it |
|--------|------------|-------------------------------|--------------------|
| **full** | No split — fit on all N rows, predict in-sample | (n/a, the same fit is used for every i) | `clp_ols.py` only |
| **kfold** | `KFold(n_splits=5, shuffle=True, random_state=42)` | Person-quarter rows from the same individual can land in *different* folds (train vs. test) | `CLP_final.py`, `clp_lasso_appendix.py` (`SPLIT_MODES[0]`) |
| **groupkfold** | `GroupKFold(n_splits=5)` with `groups=person_id` | All quarters of a given individual stay together in either train *or* test | `CLP_final_group.py`, `clp_lasso_appendix.py` (`SPLIT_MODES[1]`), `clp_granular.py` |

**Trade-off.** `full` produces the smallest predictive variance but
introduces overfitting bias into the CLP, which can shrink bounds toward
the in-sample mean — only safe with OLS on a small base feature set.
`kfold` is the textbook DR cross-fit and works when person-level
dependence is mild. `groupkfold` is the conservative choice: it removes
all within-person leakage, slightly inflates first-stage MSE, and is
the only one whose CIs (via person-clustered multiplier bootstrap) are
consistent under arbitrary within-person dependence.

---

### 2.3  Vertex-handling Regimes (LP solution approach)

Once b̂₀(Xᵢ) is in hand, the CLP plug-in step requires solving
`min_{ν ∈ T_q} ν' b̂₀(Xᵢ)` for every observation i, where
`T_q = {ν : Aᵀν ≥ q}`. How this is done depends on the polytope size.

| Regime | Description | When it's used |
|--------|-------------|----------------|
| **Vertex enumeration** | Enumerate all C(d, k) candidate vertices of `T_q` by solving the basic feasible problem on every k-subset of d columns (drop singular, drop infeasible). For each i, score every vertex and take the argmin. Exact LP optimum. | Coarse 5×9 KT polytope: `clp_ols.py`, `clp_lasso_appendix.py`, `CLP_final.py`, `CLP_final_group.py`. C(9,5) = 126 candidates, < 100 unique vertices in practice. |
| **Per-i LP, box `[−5, 5]`** | Solve `linprog(b̂ᵢ, A_ub=−Aᵀ, b_ub=−q, bounds=[(−5,5)]ᵏ)` for each i. The narrow box bounds the LP and rules out unbounded recession directions, at the cost of clipping the true optimum when the LP wants a long ν. | Granular specs in `clp_granular.py`. Modes A and E. |
| **Per-i LP, box `[−200, 200]`** | Same as above but with a much wider box. Closer to the true LP, but more observations end up at the box face (cap-binders). | Granular specs in `clp_granular.py`. Modes B, C, D. |
| **Constant fallback** | If the per-i LP errors or comes back with a zero vector, fall back to ν ≡ 0 (contributes 0 to that observation). Keeps N constant. | Modes A and B. |
| **Drop-fail** | If the per-i LP errors, drop that observation entirely. Lowers N but avoids biasing toward zero. | Modes C, D, E. |
| **Drop cap-binder** | Drop observations where the LP solution lies at the box face (`max(|ν|) ≥ box − ε`). These are the observations whose true optimum was in the recession cone. | Modes D, E. |
| **IQR-trim outliers** | After computing contributions cᵢ = νᵢ' Bᵢ, trim contributions outside 1.5×IQR of the empirical distribution. | Mode C. |

**The 5 named modes in `clp_granular.py`** combine these primitives:

| Mode | Box | Fallback | Drop fail | Drop cap-binder | IQR-trim |
|------|-----|----------|-----------|------------------|----------|
| **A** | [−5, 5]    | constant | no  | no  | no |
| **B** | [−200, 200] | constant | no  | no  | no |
| **C** | [−200, 200] | (n/a)    | yes | no  | yes |
| **D** | [−200, 200] | (n/a)    | yes | yes | no |
| **E** | [−5, 5]    | (n/a)    | yes | yes | no |

A vs. B isolates the impact of the box width. C and D both drop failed
LPs but address recession unboundedness in different ways: C trims at
the contribution stage; D drops the underlying observation. E is the
strictest — drop everything that touches the narrow [−5, 5] face.
Comparing bounds across A–E reveals how sensitive the estimate is to
recession-cone observations.

---

### 2.4  Estimator Regimes

The choice of first-stage learner for b̂₀(X) = Ê[B | X].

| Estimator | Implementation | Hyperparameters | Files |
|-----------|----------------|-----------------|-------|
| **OLS** | `sklearn.linear_model.LinearRegression` | none | `CLP_final.py`, `CLP_final_group.py`, `clp_ols.py`, `simulation.py` |
| **LASSO** | `LassoCV` | `alphas=np.logspace(-4, 1, 30)`, `cv=5`, `eps=1e-4`, `max_iter=4000`, `random_state=42` | `CLP_final.py`, `CLP_final_group.py`, `clp_lasso_appendix.py`, `clp_granular.py`, `simulation.py` |
| **Ridge** | `RidgeCV` | `alphas=np.logspace(-4, 4, 40)` | same as LASSO |
| **PostLasso** | `LassoCV` for variable selection, then OLS refit on the selected set | same LASSO settings; if zero coefs selected, fall back to all variables | `CLP_final.py`, `CLP_final_group.py` only |

**Per-component fitting.** All estimators are fit *per B-component*
(one regression per state j = 0…4). Within each fold, a `StandardScaler`
is fit on the training-fold X and used to transform both train and
test rows; the response is never scaled.

**Why three flavours of linear?** OLS is the high-variance,
low-bias benchmark and stays well-behaved on the base feature set.
LASSO regularises toward sparsity, controlling variance on the econ
set, but can over-shrink real signal. Ridge spreads shrinkage across
all coefficients and is most stable when many features are mildly
informative. PostLasso decouples selection from estimation: LASSO
picks the variables, OLS refits without shrinkage bias. Comparing
the four exposes how much of the bound width is driven by the
regulariser vs. the data.

---

### 2.5  Granularity Design Regimes

The granularity of the dual polytope — the shape (number of rows ×
number of cols) of A — controls how many transitions the LP can resolve
and the underlying row-coarsening / column-pooling assumptions.

| Regime | Shape | Pooling | Files |
|--------|------:|---------|-------|
| **KT coarse 5×9** | 5 rows × 9 cols | Single row for each of {0n, 1n, 2n, 0p, 2p}; 9 transition β parameters spanning the 9 (source, destination) groups G1…G9 with both sides pooled. | `clp_ols.py`, `clp_lasso_appendix.py`, `CLP_final.py`, `CLP_final_group.py` |
| **Granular sub-bin (8 specs)** | 5×13 to 13×53 | Spec-by-spec choices over which of the 9 groups get sub-bin splits on the source side, destination side, or both. See the granularity summary table. | `clp_granular.py` (runs all 8: spec1, spec5, spec8, spec11, spec13, spec14, spec14_alt, spec18) |

**Spec-by-spec granularity (clp_granular.py).** The 8 specs are
ordered roughly from "richest column structure" to "sparsest":

- **spec1 (13×33)** — canonical: every income bin in 1n and 2n is its own row; G1, G5, G6 destination-1r and G3, G7 source plus G8/G9 sub-bin all active.
- **spec5 (11×53)** — pools b6p–b8p into 2p but maximally splits the destination 1r side, yielding the widest column set.
- **spec8 (7×25)** — collapses 1n into a single row and pools 2p, retaining only the 2n source × 1r destination split.
- **spec11 (5×13)** — KT 5-row layout with a single fine grain (G7 destination split into b1r…b5r).
- **spec13 (11×29)** — spec1 with the underreporter side (2u) pooled, b6p–b8p collapsed to 2p.
- **spec14 (11×29)** — spec1 with 1n replaced by 3 low-tail sub-bins (low_b1n, low_b2n, low_b3n).
- **spec14_alt (9×25)** — spec13 + 1n low-tail.
- **spec18 (7×11)** — sparsest: KT 5-row + 2 extra constraint rows (low_b1p, low_b2p) that point-identify two β columns in G3.

**Why both KT coarse and granular?** The coarse 5×9 spec is what
KT and the original CLP paper report; it gives the most interpretable
bounds and the cleanest comparison against Table 4. The granular
specs trade interpretability for tighter bounds: each extra constraint
row narrows T_q and may reduce the LP optimum, but only if the
underlying cell-level data supports the disaggregation. Cross-spec
agreement is the central robustness check — if two specs with very
different sub-bin choices land on similar bounds for the headline
transitions (β(2n,1r), β(1n,1r), and the KT composites), then the
result is not an artefact of any single granularity choice.


## 3. Files

- **`distribution_over_states.py`** (`replication/01_data_pipeline/`).
  Python port of
   `DistributionOverStates.do + DistributionOverStates_programs.do`
  files in Kline and Tartari (2016) replication package.
  Loads `JF.dta` (pre-cleaned dataset from Kline and Tartari (2016):
  please consult their replication package to arrive to JF.dta,
  this may require data cleaning in STATA, we simply used their
  STATA files to get this version), applies the Q1–Q7 / `kidcount` filter, builds
  `Cbin`, classifies `ebin` and `partic`, fits the pscore logit, runs
  the person-clustered bootstrap, and writes
  `Table4_mat_python.txt`.
  This is the file that replicates KT's data pipeline and arrives at
  the same sample, and also replicates KT's Table 4
  (latent-type proportions with bootstrap SEs).

- **`replicating5.py`** (`replication/02_kt_table5_check/`). Loads
  `Table4_mat_python.txt`, sets up 5×9 conservation matrix and the
  three composite questions (TUW, TUWelf, Exit), and runs
  `scipy.optimize.linprog` on the aggregates to reproduce **Kline and Tartari's
  Table 5**. This is a sanity check to make sure we work with the same
  sample in the next steps.

  - **`CLP_synthetic_simulation.py`** (`replication/03_simulation/`).
  Numerical illustration of the CLP estimator on synthetic data with
  known latent flows `β_0`. Generates a 5×9-design DGP, runs the full
  CLP pipeline, and compares the result to the **analytical KT-style LP
  bounds** on the same population. Uses **OLS, Ridge, and Lasso** as
  three alternative first stages. Verifies (i) that the CLP bounds
  bracket the true `β_0`, (ii) that they are tighter than the
  analytical bounds (per the Jensen-inequality argument. More details
  on DGP and results in Section 5 of the paper.



- **`CLP_final_OLS_variants.py`** (`replication/04_coarse_5x9/`).
  Empirical CLP on the real JF sample at the coarse 5×9 KT design,
  with an **OLS first stage** and **no cross-fitting** (full
  in-sample fit), under four sub-specifications: full-panel OLS,
  cross-sectional OLS, GroupKFold OLS (K = 3, 5, 7), and four
  covariate subsets. This replicates **Table X** in the paper.


  - **`CLP_lasso_appendix.py`**
  (`replication/04_coarse_5x9/`). Implements the bounds estimator for
  two estimators (**LASSO, Ridge**), two covariate sets (**base, extended**), under
  KFold cross-fitting and GroupKFold cross-fitting by
  person, overall 8 specifications. This replicates **Table X** in the Appendix.


  - **`CLP_granular.py`**
  (`replication/04_coarse_5x9/`).  This replicates **Table X** in the paper.
  Implements the bounds estimator for
  **LASSO** estimator, two covariate sets (**base, extended**), under
  GroupKFold cross-fitting by person, but under 8 granular design matrix specifications,
  and across 5 vertex handling regimes. First specification with 13x33 design matrix,
  with Lasso estimator and GroupKFold cross-fitting by person across two cvoariate
  regimes corresponds to **Table X** in the paper.

  The full breakdown of the specifications included is below:

  ### Summary table

`✓` = sub-bin split active; `−` = pooled at this granularity.

| Spec | Rows | G1 dest 1r | G3 src 2n | G3 dest 1r | G4 dest 2n | G5 dest 1r | G6 dest 1n | G7 src 1n | G7 dest 1r | G8/G9 src/dest 2u | Cols |
|------|------|------------|-----------|------------|------------|------------|------------|-----------|------------|-------------------|------|
| 1      | 13 (full sub-bin)        | ✓ (5) | ✓ (3) | −     | ✓ (3) | ✓ (5) | ✓ (5) | ✓ (5) | −     | ✓ (3 each) | 33 |
| 5      | 11 (b6p–b8p pooled)      | −     | ✓ (3) | ✓ (5) | ✓ (3) | −     | ✓ (5) | ✓ (5) | ✓ (5) | −          | 53 |
| 8      | 7 (1n + 2p pooled)       | −     | ✓ (3) | ✓ (5) | ✓ (3) | −     | −     | −     | −     | −          | 25 |
| 11     | 5 (KT coarse)            | −     | −     | −     | −     | −     | −     | −     | ✓ (5) | −          | 13 |
| 13     | 11 (2u pooled)           | ✓ (5) | ✓ (3) | −     | ✓ (3) | ✓ (5) | ✓ (5) | ✓ (5) | −     | −          | 29 |
| 14     | 11 (1n low-tail split)   | ✓ (5) | ✓ (3) | −     | ✓ (3) | ✓ (5) | ✓ (3) | ✓ (3) | −     | ✓ (3 each) | 29 |
| 14_alt | 9 (spec13 + 1n low-tail) | ✓ (5) | ✓ (3) | −     | ✓ (3) | ✓ (5) | ✓ (3) | ✓ (3) | −     | −          | 25 |
| 18     | 7 (spec15 + low_b1p/low_b2p constraints) | − | − | ✓ (3) | − | − | − | − | − | − | 11 |

#### Notes

- **Spec 1** is the canonical 13×33 layout: 1n source split into 5 sub-bins (b1n…b5n), 2n source split into 3 (b6n…b8n), and 2u source/destination split into 3 (b6u…b8u).
- **Spec 5** pools the above-FPL on-welfare rows (b6p…b8p) into a single `2p` row but *also* splits the destination 1r in G3 and G7 into b1r…b5r, giving the widest column count (53).
- **Spec 8** keeps the G3 dest split but collapses 1n entirely (single `1n` row, no G6/G7 splits) and pools 2p.
- **Spec 11** is the only KT-coarse-row spec (5 rows) that activates a destination split (G7 dest 1r → b1r…b5r); G3 stays pooled.
- **Spec 13** is Spec 1 with the 2u column pair pooled (G8/G9 → 1 each) and the b6p…b8p rows collapsed to `2p`.
- **Spec 14** and **Spec 14_alt** replace the 5 base 1n sub-bins (b1n…b5n) with 3 low-tail sub-bins (low_b1n, low_b2n, low_b3n), shrinking both G6 and G7 src-1n columns from 5 to 3; 14_alt additionally pools 2u.
- **Spec 18** is the most aggressive simplification: KT 5-row layout + two extra constraint rows (low_b1p, low_b2p) that point-identify β(2n, low_b1r) and β(2n, low_b2r) under the assumption that pooled-1r flows in G1/G5/G7/G9 land entirely in low_b3r.

The 9 "G" groups partition the columns of the granular CLP design matrix.
Each column is a β parameter β(s → d) where s is the source state (under
AFDC counterfactual) and d is the destination state (under JF). Each
group fixes the (source pattern, destination pattern) up to optional
sub-bin splitting on either side.

**State label conventions** (matched to `STATE_DEF` in `CLP_granular_final_combined.py`):

| Suffix | Cells (ebin, partic) | Meaning |
|--------|----------------------|---------|
| `n`    | partic = 0           | off welfare, with stated earnings bin |
| `r`    | partic = 1, ebin ∈ {0…5} | on welfare, in- or below-FPL earnings ("truthful" on-welfare cell) |
| `u`    | partic = 1, ebin ∈ {6,7,8} | on welfare with above-FPL stated earnings ("underreporter" cell) |
| `p`    | partic = 1           | on welfare, any earnings bin — used interchangeably with `r`/`u` when row-coarseness allows |

Earnings bins: `0` = zero earnings; `b1…b5` = below FPL (5 sub-bins of width 0.2·FPL); `b6…b8` = above FPL (3 sub-bins above 1.0·FPL); pooled labels `1n`/`1r` cover `b1…b5`, `2n`/`2u` cover `b6…b8`. Low-tail variants: `low_b1n…low_b3n` (and analogous `_r`/`_p`) split the in-FPL range as {b1}, {b2}, {b3 ∪ b4 ∪ b5}.

**The 9 groups** (source → destination, in canonical Spec 1 form):

| Group | Source → Destination | Default split (Spec 1) | Economic interpretation |
|-------|----------------------|------------------------|-------------------------|
| **G1** | `0n` → `1r`  | dest 1r split into `b1r…b5r` (5)              | Take-Up Welfare from zero-earnings off-welfare baseline (destination earnings sub-bin) |
| **G2** | `0r` → `0n`  | single col (no split possible)                | Welfare exit without working (`0p → 0n`) — Exit 0r composite |
| **G3** | `2n` → `1r`  | src 2n split into `b6n…b8n` (3); dest 1r pooled in Spec 1, may be split into `b1r…b5r` (5) in finer specs | Take-Up Welfare from above-FPL off-welfare (welfare entry with earnings drop) |
| **G4** | `0r` → `2n`  | dest 2n split into `b6n…b8n` (3)              | Welfare exit into above-FPL work |
| **G5** | `0r` → `1r`  | dest 1r split into `b1r…b5r` (5)              | Take-up of in-FPL work while keeping welfare |
| **G6** | `0r` → `1n`  | dest 1n split into `b1n…b5n` (5) or `low_b1n…low_b3n` (3) | Welfare exit into in-FPL work |
| **G7** | `1n` → `1r`  | src 1n split into `b1n…b5n` (5) or `low_b1n…low_b3n` (3); dest 1r pooled in Spec 1, may be split into `b1r…b5r` (5) in finer specs | Take-Up Welfare from in-FPL off-welfare |
| **G8** | `0r` → `2u`  | dest 2u split into `b6u…b8u` (3), or pooled to single `2u` | Welfare exit-and-re-entry path through above-FPL on-welfare ("underreporter destination") |
| **G9** | `2u` → `1r`  | src 2u split into `b6u…b8u` (3); dest 1r pooled | "Underreporter" → truthful on-welfare in-FPL cell |

The three KT composites are built from G1, G3, G7:
- **TUW (Take-Up Work)** mixes flows with destination `1r` and earnings-bin destinations under G1/G5/G7.
- **TUWelf (Take-Up Welfare)** sums β(0n,1r) + β(2n,1r) + β(1n,1r) — the lead columns of G1, G3, G7.
- **Exit 0r** is simply β(0r, 0n) — the single column of G2.


  - **`CLP_welfare.py`**
  (`replication/04_coarse_5x9/`).  This replicates **Table X** in the paper.
  Implements the welfare bounds with
  **LASSO** estimator, base covariate set, under
  GroupKFold cross-fitting by person, with 13x33 design. 



## 4. Important Functions





## 5. Software requirements

* **Python** ≥ 3.10
* **numpy** ≥ 1.24
* **pandas** ≥ 2.0
* **scipy** ≥ 1.10 (LP solver `scipy.optimize.linprog`, HiGHS backend)
* **scikit-learn** ≥ 1.3 (`LassoCV`, `GroupKFold`)
* **reportlab** ≥ 4.0 (only required to rebuild `CLP_code_manual.pdf`)


