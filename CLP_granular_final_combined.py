"""
CLP_granular_final_combined.py
==============================
Combined CLP estimation file that consolidates the work in:
  - CLP_granular_final_group.py  (the canonical 13x33 setup with multiple
    estimators and feature regimes)
  - CLP_granular_correct.py      (multi-spec LASSO with base covariates)
  - CLP_granular_correct_econ.py (same with econ feature set)

PIPELINE STRUCTURE
==================
PHASE 1: Canonical 13x33 spec1 with full estimator/regime sweep
  - {LASSO, Ridge, OLS} x {base, econ} = 6 configs
  - GroupKFold by person, all configs cross-fit
  - Reports all 33 individual betas + KT composites + 2n->1r / 1n->1r / 0n->1r
    coarse-from-granular composites
  - Saves all_results to CSV

PHASE 2: Multi-spec scan with LASSO + GroupKFold
  - 21 specs total: 9 from CLP_granular_correct (specs 1-9), plus 12 new
    specs (10-20 with two flavors of "spec 14")
  - Each spec runs with {base, econ} covariate sets
  - Reports individual beta bounds + composites + 2n->1r / 1n->1r / 0n->1r
  - Cross-spec comparison summary table for the three coarse composites
  - Saves all_results to CSV

PHASE 3: OLS + base + full-sample (no cross-fitting) sensitivity
  - Specs 1-17 only (not 18-20, which use the more delicate constraint rows)
  - Reports individual beta bounds + composites
  - Saves separate CSV

NEW SPECIFICATIONS (10-20)
==========================
  spec 10: G3 + G7 destination 1r split into b1r..b5r (5), all else pooled
  spec 11: G7 destination 1r split into b1r..b5r (5), all else pooled
  spec 12: 2n source split into b6n..b8n (already in spec1) but DESTINATION
           1r in G3 stays pooled; 2n FURTHER split into 5 sub-bins by an
           extra cut so we get 5 sub-bins of 2n in G3, G4. Result: 9 rows
           (0n,1n,b6an..b6en,0p,2p where b6an..b6en are 5 above-FPL sub-bins)
           x 17 cols.
  spec 13: spec1 with 2u NOT granularized (G8 pooled, G9 pooled, b6p..b8p
           rows replaced by single 2p row). 11 rows x 29 cols.
  spec 14: spec1 + 1n granularized with HEAVY LOW TAIL (3 sub-bins:
           <0.20*FPL, 0.20-0.40*FPL, >=0.40*FPL).  11 rows x 29 cols.
  spec 14_alt: spec13 + 1n low-tail split.  9 rows x 25 cols.
  spec 15: G3 destination 1r split with low tail (3 sub-bins). All else
           pooled. KT 5-row base. 5 rows x 11 cols.
  spec 16: G7 destination 1r split with low tail (3 sub-bins). All else
           pooled. KT 5-row base. 5 rows x 11 cols.
  spec 17: G3 + G7 destination 1r split with low tail. KT 5-row. 5 x 13.
  spec 18: spec 15 + low_b1p, low_b2p constraint rows; "minimally
           necessary" extra cols means we keep spec 15's cols AND treat
           the constraint rows as referencing only G3's split cols.
           7 rows x 11 cols. Drop low_b3p (potential underreporter
           contamination).
  spec 19: spec 16 + same constraint rows. 7 rows x 11 cols.
  spec 20: spec 17 + same constraint rows. 7 rows x 13 cols.

NOTE on specs 18-20: "minimally necessary columns" interpreted as adding
NO new cols beyond what already exists in the parent spec.  The new rows
contain only the +1 entries from cols whose latent destination is
explicitly low_b1r or low_b2r.  Pooled-1r cols (G1, G5, G7, G9 if not
already split) contribute 0 to the new rows -- this implicitly assumes
those flows land entirely in low_b3r (the dropped bin).  This is a
strong assumption but matches the "drop upper bin to avoid latent
underreporter contamination" framing.

BUDGET / RUNTIME
================
With ~30k person-quarters per phase, the full pipeline does:
  PHASE 1: 6 configs x 33 betas x 2 directions x ~30k LP solves ~= 12M
  PHASE 2: 21 specs x 2 covariate sets x avg 18 betas x 2 dirs x 30k ~= 45M
  PHASE 3: 17 specs x 1 config x avg 18 betas x 2 dirs x 30k ~= 18M
Total ~= 75M LP solves.  At ~1 ms each via scipy.linprog HiGHS, expect
20-25 hours of LP time.  Set RUN_PHASE_{1,2,3} = False to skip phases,
or reduce N_BOOTSTRAP / SPECS_TO_RUN for quicker iteration.

Outputs
-------
  combined_phase1_13x33.csv      (PHASE 1 individual + composite)
  combined_phase2_multispec.csv  (PHASE 2 individual + composite)
  combined_phase3_ols_full.csv   (PHASE 3 individual + composite)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import time
import warnings
from typing import Dict, List, Tuple, FrozenSet, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# Reuse data prep, propensity score, bootstrap, and engineered-features from
# the existing granular file.
from CLP_granular_final_group import (
    PSCORE_VARS, JF_CONFIG, COV_VARS,
    JF_DTA_PATH, POLICY_RULES_PATH, TABLE4_MAT_PATH,
    K_FOLDS, RANDOM_SEED,
    fit_pscore_logit, load_table4_mat,
    prepare_jf_data_granular,
    multiplier_bootstrap_ci,
    engineer_features_econ,
)


# =============================================================================
# 2. CONSTANTS / RUN CONFIG
# =============================================================================
N_BOOTSTRAP = 200
np.random.seed(RANDOM_SEED)

OUT_DIR = "/Users/gevorgkhandamiryan/Desktop/cursorclp/check/finalcheck"
PHASE1_CSV = os.path.join(OUT_DIR, "combined_phase1_13x33.csv")
PHASE2_CSV = os.path.join(OUT_DIR, "combined_phase2_multispec.csv")
PHASE3_CSV = os.path.join(OUT_DIR, "combined_phase3_ols_full.csv")

# Phase toggles (set to False to skip)
RUN_PHASE_1 = True
RUN_PHASE_2 = True
RUN_PHASE_3 = True

# nu box for the per-i LP.  The dual LP solves
#   min nu' b_hat[i]  s.t.  A' nu >= q,  nu_lo <= nu <= nu_hi
# When q has large entries (e.g., 1/p_a(s_a) for small p_a like p21_c~0.009
# giving q ~ 100, or coarse-flow composites for pi(2n,1r) giving q ~ 10),
# a tight box [-5, 5] makes the LP infeasible and the fallback nu corrupts
# the bound.  We use a wide box [-200, 200] which accommodates all q
# scales encountered in this file's specs.  The LP_DIAG['binding_nu_count']
# counter will warn if any solve hits the cap; if so, widen further.
NU_LO, NU_HI = -200.0, 200.0


# =============================================================================
# 3. STATE_DEF (extended for low-tail bins)
# =============================================================================
# Each label maps to a frozenset of (ebin, partic) cells it covers.  Aliases:
#   "0r" == "0p"  (zero-earnings on welfare = truthful)
#   "1r" == pooled b1r..b5r  (below-FPL on-welfare = truthful)
#   "2u" == pooled b6u..b8u  (above-FPL on-welfare = underreporter)
#   "b6u" == "b6p", etc.

def _fc(*pairs):
    return frozenset(pairs)


STATE_DEF: Dict[str, FrozenSet[Tuple[int, int]]] = {
    # Single-cell labels
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
    # Pooled coarse labels
    "1n":  _fc((1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
    "2n":  _fc((6, 0), (7, 0), (8, 0)),
    "1r":  _fc((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
    "1p":  _fc((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
    "2u":  _fc((6, 1), (7, 1), (8, 1)),
    "2p":  _fc((6, 1), (7, 1), (8, 1)),
    # Low-tail 1n splits: < 0.20 FPL, 0.20-0.40 FPL, >= 0.40 FPL
    "low_b1n": _fc((1, 0)),
    "low_b2n": _fc((2, 0)),
    "low_b3n": _fc((3, 0), (4, 0), (5, 0)),
    # Low-tail 1r splits (truthful below-FPL on welfare, low/mid/high)
    "low_b1r": _fc((1, 1)),
    "low_b2r": _fc((2, 1)),
    "low_b3r": _fc((3, 1), (4, 1), (5, 1)),
    # Low-tail 1p observable rows (used as new constraint rows in specs 18-20)
    "low_b1p": _fc((1, 1)),
    "low_b2p": _fc((2, 1)),
    # Spec 12: 5-bin split of 2n.  In our 9-bin data we have b6n, b7n, b8n
    # (3 sub-bins above FPL).  To create 5 sub-bins from 3 cells, we split
    # b8n in half by the implicit threshold 1.6 x FPL.  Since the dataset
    # uses ebin in {0..8} we can't redo the cuts post-hoc; for spec 12 we
    # instead take 5 hypothetical sub-bins constructed via auxiliary
    # midpoint ranges.  Below we name them "s2_a", "s2_b", ..., "s2_e";
    # each currently covers the same observable cells as the existing
    # b6n..b8n trio with a heuristic mapping (a=b6n, b=b6n, c=b7n, d=b7n,
    # e=b8n), so spec 12 is a structural placeholder that may produce
    # near-identical bounds across some sub-bins until the data is rebuilt
    # with a 5-bin above-FPL classifier.  See SPEC_12_NOTE below.
    "s2_a": _fc((6, 0)),
    "s2_b": _fc((6, 0)),
    "s2_c": _fc((7, 0)),
    "s2_d": _fc((7, 0)),
    "s2_e": _fc((8, 0)),
    # On-welfare counterparts for s2_*
    "s2u_a": _fc((6, 1)),
    "s2u_b": _fc((6, 1)),
    "s2u_c": _fc((7, 1)),
    "s2u_d": _fc((7, 1)),
    "s2u_e": _fc((8, 1)),
}

SPEC_12_NOTE = (
    "Spec 12 splits 2n into 5 nominal sub-bins {s2_a..s2_e}, but the underlying "
    "data only resolves 3 sub-bins (b6n,b7n,b8n in the 9-bin classifier). "
    "Without a true 5-way data partition, s2_a and s2_b both map to b6n, "
    "s2_c and s2_d both map to b7n, and s2_e maps to b8n -- producing "
    "potentially redundant cols.  Treat spec 12 results as an upper bound "
    "on what a true 5-bin 2n classifier could achieve."
)


# =============================================================================
# 4. CRITERIA HELPERS  (used by composite/target builders)
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
# 5. SPEC DEFINITIONS  (specs 1-9 from CLP_granular_correct + new 10-20)
# =============================================================================

# ---- Specs 1-9 (re-using CLP_granular_correct definitions verbatim) ----
def _spec1():
    return dict(
        name="Spec 1: 13 x 33 (canonical CLP_granular_final_group)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", f"b{j}r") for j in range(1, 6)]                          # G1: 5
            + [("0r", "0n")]                                                  # G2: 1
            + [(f"b{j}n", "1r") for j in range(6, 9)]                         # G3: 3
            + [("0r", f"b{j}n") for j in range(6, 9)]                         # G4: 3
            + [("0r", f"b{j}r") for j in range(1, 6)]                         # G5: 5
            + [("0r", f"b{j}n") for j in range(1, 6)]                         # G6: 5
            + [(f"b{j}n", "1r") for j in range(1, 6)]                         # G7: 5
            + [("0r", f"b{j}u") for j in range(6, 9)]                         # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                         # G9: 3
        ),
        expected_shape=(13, 33),
    )


def _spec2():
    return dict(
        name="Spec 2: 13 x 25 (1r pooled in G1 and G5)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{j}n", "1r") for j in range(6, 9)]
            + [("0r", f"b{j}n") for j in range(6, 9)]
            + [("0r", "1r")]
            + [("0r", f"b{j}n") for j in range(1, 6)]
            + [(f"b{j}n", "1r") for j in range(1, 6)]
            + [("0r", f"b{j}u") for j in range(6, 9)]
            + [(f"b{j}u", "1r") for j in range(6, 9)]
        ),
        expected_shape=(13, 25),
    )


def _spec3():
    return dict(
        name="Spec 3: 13 x 37 (G3 fully split by destination)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                    # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]
            + [("0r", "1r")]
            + [("0r", f"b{j}n") for j in range(1, 6)]
            + [(f"b{j}n", "1r") for j in range(1, 6)]
            + [("0r", f"b{j}u") for j in range(6, 9)]
            + [(f"b{j}u", "1r") for j in range(6, 9)]
        ),
        expected_shape=(13, 37),
    )


def _spec4():
    return dict(
        name="Spec 4: 13 x 57 (G3 and G7 fully split by destination)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                    # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]
            + [("0r", "1r")]
            + [("0r", f"b{j}n") for j in range(1, 6)]
            + [(f"b{j}n", f"b{i}r")
               for j in range(1, 6) for i in range(1, 6)]                    # G7: 25
            + [("0r", f"b{j}u") for j in range(6, 9)]
            + [(f"b{j}u", "1r") for j in range(6, 9)]
        ),
        expected_shape=(13, 57),
    )


def _spec5():
    return dict(
        name="Spec 5: 11 x 53 (above-FPL on-welfare pooled into 2p)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]
            + [("0r", f"b{j}n") for j in range(6, 9)]
            + [("0r", "1r")]
            + [("0r", f"b{j}n") for j in range(1, 6)]
            + [(f"b{j}n", f"b{i}r")
               for j in range(1, 6) for i in range(1, 6)]
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(11, 53),
    )


def _spec6():
    return dict(
        name="Spec 6: 9 x 29 (1n pooled, G3 split, G7 pooled)",
        rows=["0n", "1n", "b6n", "b7n", "b8n",
              "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]
            + [("0r", f"b{j}n") for j in range(6, 9)]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "1r")]
            + [("0r", f"b{j}u") for j in range(6, 9)]
            + [(f"b{j}u", "1r") for j in range(6, 9)]
        ),
        expected_shape=(9, 29),
    )


def _spec7():
    return dict(
        name="Spec 7: 9 x 33 (1n source pooled, G7 destination split)",
        rows=["0n", "1n", "b6n", "b7n", "b8n",
              "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]
            + [("0r", f"b{j}n") for j in range(6, 9)]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", f"b{i}r") for i in range(1, 6)]
            + [("0r", f"b{j}u") for j in range(6, 9)]
            + [(f"b{j}u", "1r") for j in range(6, 9)]
        ),
        expected_shape=(9, 33),
    )


def _spec8():
    return dict(
        name="Spec 8: 7 x 25 (1n + 2p pooled; G3 split)",
        rows=["0n", "1n", "b6n", "b7n", "b8n", "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]
            + [("0r", f"b{j}n") for j in range(6, 9)]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "1r")]
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(7, 25),
    )


def _spec9():
    return dict(
        name="Spec 9: 5 x 13 (KT 5-row + 2n -> b1r..b5r dest split)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [("2n", f"b{i}r") for i in range(1, 6)]
            + [("0r", "2n")]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "1r")]
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(5, 13),
    )


# ---- New specs 10-20 ----
def _spec10():
    """G3 + G7 destinations split into b1r..b5r (5 each), all else pooled.
    Base = spec 9 (KT 5 rows) extended with G3 dest split (already in spec9)
    and G7 dest split."""
    return dict(
        name="Spec 10: 5 x 17 (G3 + G7 destinations split into b1r..b5r)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=(
            [("0n", "1r")]                                                    # G1: 1
            + [("0r", "0n")]                                                  # G2: 1
            + [("2n", f"b{i}r") for i in range(1, 6)]                         # G3: 5
            + [("0r", "2n")]                                                  # G4: 1
            + [("0r", "1r")]                                                  # G5: 1
            + [("0r", "1n")]                                                  # G6: 1
            + [("1n", f"b{i}r") for i in range(1, 6)]                         # G7: 5
            + [("0r", "2u")]                                                  # G8: 1
            + [("2u", "1r")]                                                  # G9: 1
        ),
        expected_shape=(5, 17),
    )


def _spec11():
    """G7 destination split into b1r..b5r (5), all else pooled."""
    return dict(
        name="Spec 11: 5 x 13 (G7 destination split, G3 source pooled)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [("2n", "1r")]                                                  # G3: 1 pooled
            + [("0r", "2n")]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", f"b{i}r") for i in range(1, 6)]                         # G7: 5
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(5, 13),
    )


def _spec12():
    """2n split into 5 sub-bins {s2_a..s2_e} both as source (G3) and as
    destination of G4.  Other groups stay coarse.  Note: STATE_DEF maps
    s2_a..s2_e onto the existing b6n..b8n cells with a heuristic 1-1
    mapping, so several rows may be linearly dependent or have identical
    data signatures.  See SPEC_12_NOTE."""
    return dict(
        name="Spec 12: 9 x 17 (2n split into 5 sub-bins in G3 & G4)",
        rows=["0n", "1n",
              "s2_a", "s2_b", "s2_c", "s2_d", "s2_e",
              "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [(f"s2_{c}", "1r") for c in "abcde"]                            # G3: 5
            + [("0r", f"s2_{c}") for c in "abcde"]                            # G4: 5
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "1r")]
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(9, 17),
    )


def _spec13():
    """Spec 1 (13x33) but 2u not granularized: G8 pooled, G9 pooled,
    b6p..b8p rows replaced by single 2p row.  Result: 11x29."""
    return dict(
        name="Spec 13: 11 x 29 (spec1 with 2u pooled)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "2p"],
        cols=(
            [("0n", f"b{j}r") for j in range(1, 6)]                           # G1: 5
            + [("0r", "0n")]                                                  # G2: 1
            + [(f"b{j}n", "1r") for j in range(6, 9)]                         # G3: 3
            + [("0r", f"b{j}n") for j in range(6, 9)]                         # G4: 3
            + [("0r", f"b{j}r") for j in range(1, 6)]                         # G5: 5
            + [("0r", f"b{j}n") for j in range(1, 6)]                         # G6: 5
            + [(f"b{j}n", "1r") for j in range(1, 6)]                         # G7: 5
            + [("0r", "2u")]                                                  # G8: 1
            + [("2u", "1r")]                                                  # G9: 1
        ),
        expected_shape=(11, 29),
    )


def _spec14():
    """Spec 1 + 1n split into 3 low-tail bins (low_b1n,low_b2n,low_b3n).
    Replaces b1n..b5n rows with low_b1n/low_b2n/low_b3n.  Cols in G6 and
    G7 collapse from 5 to 3 each."""
    return dict(
        name="Spec 14: 11 x 29 (spec1 with 1n low-tail split)",
        rows=["0n", "low_b1n", "low_b2n", "low_b3n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", f"b{j}r") for j in range(1, 6)]                           # G1: 5
            + [("0r", "0n")]                                                  # G2: 1
            + [(f"b{j}n", "1r") for j in range(6, 9)]                         # G3: 3
            + [("0r", f"b{j}n") for j in range(6, 9)]                         # G4: 3
            + [("0r", f"b{j}r") for j in range(1, 6)]                         # G5: 5
            + [("0r", "low_b1n"), ("0r", "low_b2n"), ("0r", "low_b3n")]       # G6: 3
            + [("low_b1n", "1r"), ("low_b2n", "1r"), ("low_b3n", "1r")]       # G7: 3
            + [("0r", f"b{j}u") for j in range(6, 9)]                         # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                         # G9: 3
        ),
        expected_shape=(11, 29),
    )


def _spec14_alt():
    """Spec 13 (2u pooled) + 1n low-tail split.  9 rows x 25 cols.
    Combination of the two -- aggressive simplification of both 1n and 2u."""
    return dict(
        name="Spec 14_alt: 9 x 25 (spec13 with 1n low-tail split)",
        rows=["0n", "low_b1n", "low_b2n", "low_b3n",
              "b6n", "b7n", "b8n", "0p", "2p"],
        cols=(
            [("0n", f"b{j}r") for j in range(1, 6)]                           # G1: 5
            + [("0r", "0n")]                                                  # G2: 1
            + [(f"b{j}n", "1r") for j in range(6, 9)]                         # G3: 3
            + [("0r", f"b{j}n") for j in range(6, 9)]                         # G4: 3
            + [("0r", f"b{j}r") for j in range(1, 6)]                         # G5: 5
            + [("0r", "low_b1n"), ("0r", "low_b2n"), ("0r", "low_b3n")]       # G6: 3
            + [("low_b1n", "1r"), ("low_b2n", "1r"), ("low_b3n", "1r")]       # G7: 3
            + [("0r", "2u")]                                                  # G8: 1
            + [("2u", "1r")]                                                  # G9: 1
        ),
        expected_shape=(9, 25),
    )


def _spec15():
    """KT 5-row + G3 destination 1r low-tail split (3 sub-bins).
    All other groups have pooled 1r destinations."""
    return dict(
        name="Spec 15: 5 x 11 (G3 dest 1r low-tail split, all else pooled)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=(
            [("0n", "1r")]                                                    # G1: 1
            + [("0r", "0n")]                                                  # G2: 1
            + [("2n", "low_b1r"), ("2n", "low_b2r"), ("2n", "low_b3r")]       # G3: 3
            + [("0r", "2n")]                                                  # G4: 1
            + [("0r", "1r")]                                                  # G5: 1
            + [("0r", "1n")]                                                  # G6: 1
            + [("1n", "1r")]                                                  # G7: 1
            + [("0r", "2u")]                                                  # G8: 1
            + [("2u", "1r")]                                                  # G9: 1
        ),
        expected_shape=(5, 11),
    )


def _spec16():
    """KT 5-row + G7 destination 1r low-tail split."""
    return dict(
        name="Spec 16: 5 x 11 (G7 dest 1r low-tail split, all else pooled)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [("2n", "1r")]
            + [("0r", "2n")]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "low_b1r"), ("1n", "low_b2r"), ("1n", "low_b3r")]       # G7: 3
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(5, 11),
    )


def _spec17():
    """KT 5-row + G3 AND G7 destination 1r low-tail split."""
    return dict(
        name="Spec 17: 5 x 13 (G3 + G7 dest 1r low-tail split)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [("2n", "low_b1r"), ("2n", "low_b2r"), ("2n", "low_b3r")]       # G3: 3
            + [("0r", "2n")]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "low_b1r"), ("1n", "low_b2r"), ("1n", "low_b3r")]       # G7: 3
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(5, 13),
    )


def _spec18():
    """Spec 15 + low_b1p, low_b2p constraint rows.

    EXPLICIT ROW DERIVATION (verified by strict cell-equality rule in
    build_A):

        p^J(low_b1p) - p^A(low_b1p) = beta(2n, low_b1r)
        p^J(low_b2p) - p^A(low_b2p) = beta(2n, low_b2r)

    Each constraint row pins down a SINGLE beta column of G3.  This is
    because spec 15 has G3 destinations explicitly split into low_b1r /
    low_b2r / low_b3r, but G1, G5, G7, G9 still pool to "1r"; under the
    strict-equality rule for sub-bin constraint rows (see build_A), the
    pooled-1r cols contribute 0 to the new rows.  Implicit assumption:
    pooled-1r flows land entirely in low_b3r (the dropped third bin) --
    a strong but transparent assumption.

    Result: spec 18's bounds for beta(2n, low_b1r) and beta(2n, low_b2r)
    are POINT-IDENTIFIED via the new rows, modulo the underlying flow-
    balance constraints from rows 0n..2p.
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


def _spec19():
    """Spec 16 + low_b1p, low_b2p constraint rows."""
    return dict(
        name="Spec 19: 7 x 11 (spec 16 + low_b1p/low_b2p constraints)",
        rows=["0n", "1n", "2n", "0p", "2p", "low_b1p", "low_b2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [("2n", "1r")]
            + [("0r", "2n")]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "low_b1r"), ("1n", "low_b2r"), ("1n", "low_b3r")]
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(7, 11),
    )


def _spec20():
    """Spec 17 + low_b1p, low_b2p constraint rows.

    EXPLICIT ROW DERIVATION (most informative of the three; verified):

        p^J(low_b1p) - p^A(low_b1p) = beta(2n, low_b1r) + beta(1n, low_b1r)
        p^J(low_b2p) - p^A(low_b2p) = beta(2n, low_b2r) + beta(1n, low_b2r)

    Each constraint row pins down the SUM of one G3 col + one G7 col.
    Pooled-1r cols (G1, G5, G9) contribute 0 via the strict-equality
    rule.  Among specs 18-20, this is the most informative because both
    G3 and G7 sub-bin destinations contribute to each constraint."""
    return dict(
        name="Spec 20: 7 x 13 (spec 17 + low_b1p/low_b2p constraints)",
        rows=["0n", "1n", "2n", "0p", "2p", "low_b1p", "low_b2p"],
        cols=(
            [("0n", "1r")]
            + [("0r", "0n")]
            + [("2n", "low_b1r"), ("2n", "low_b2r"), ("2n", "low_b3r")]
            + [("0r", "2n")]
            + [("0r", "1r")]
            + [("0r", "1n")]
            + [("1n", "low_b1r"), ("1n", "low_b2r"), ("1n", "low_b3r")]
            + [("0r", "2u")]
            + [("2u", "1r")]
        ),
        expected_shape=(7, 13),
    )


SPECS = {
    "spec1":  _spec1(),  "spec2":  _spec2(),  "spec3":  _spec3(),
    "spec4":  _spec4(),  "spec5":  _spec5(),  "spec6":  _spec6(),
    "spec7":  _spec7(),  "spec8":  _spec8(),  "spec9":  _spec9(),
    "spec10": _spec10(), "spec11": _spec11(), "spec12": _spec12(),
    "spec13": _spec13(), "spec14": _spec14(), "spec14_alt": _spec14_alt(),
    "spec15": _spec15(), "spec16": _spec16(), "spec17": _spec17(),
    "spec18": _spec18(), "spec19": _spec19(), "spec20": _spec20(),
}


# =============================================================================
# 6. A-MATRIX BUILDER (cell-overlap rule, generic over any spec)
# =============================================================================
# Rows whose cells are a STRICT SUBSET of some parent pool's cells are
# treated as "sub-bin constraint rows" and use a stricter cell-equality
# rule for inflows/outflows.  This prevents pooled-destination cols (e.g.
# β(0n, "1r")) from spuriously contributing +1 to multiple sub-bin
# constraint rows (which would over-count: the same pooled mass appearing
# as +1 in both low_b1p and low_b2p constraints simultaneously).
#
# Specifically: rows {low_b1p, low_b2p} (used in specs 18-20) get +1
# only when the col's destination cells EXACTLY EQUAL the row's cells.
# A col with pooled dst "1r" (cells {(1,1)..(5,1)}) does NOT match any
# single sub-bin row -- those flows are implicitly assumed to land in
# low_b3r (the dropped third-bin row).  This is the "minimally necessary
# columns" interpretation: only cols with explicit sub-bin destinations
# contribute to the new constraints.
SUBBIN_CONSTRAINT_ROWS = {"low_b1p", "low_b2p"}


def build_A(spec) -> np.ndarray:
    """For each row r and each col (s_src, s_dst):
        +1 if dst's cells overlap r's cells AND src's don't (inflow)
        -1 if src's cells overlap r's cells AND dst's don't (outflow)
         0 otherwise

    Special handling for SUBBIN_CONSTRAINT_ROWS: use strict equality
    (cells exactly equal) instead of overlap.  This prevents pooled-
    destination cols from over-counting (see comment above)."""
    rows, cols = spec["rows"], spec["cols"]
    A = np.zeros((len(rows), len(cols)), dtype=float)
    for r_idx, r_label in enumerate(rows):
        r_cells = STATE_DEF[r_label]
        is_subbin = r_label in SUBBIN_CONSTRAINT_ROWS
        for c_idx, (src, dst) in enumerate(cols):
            src_cells = STATE_DEF[src]
            dst_cells = STATE_DEF[dst]
            if is_subbin:
                # strict equality: only count if cells exactly match
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


def verify_A(A, spec, verbose=True):
    rows, cols = spec["rows"], spec["cols"]
    expected = spec["expected_shape"]
    assert A.shape == (len(rows), len(cols)), \
        f"shape {A.shape} vs ({len(rows)}, {len(cols)})"
    assert set(np.unique(A)).issubset({-1.0, 0.0, 1.0}), \
        f"Non-+-1 entries: {set(np.unique(A))}"
    rank = int(np.linalg.matrix_rank(A))
    nnz = int((A != 0).sum())
    zero_rows = list(np.where(np.abs(A).sum(axis=1) == 0)[0])
    zero_cols = list(np.where(np.abs(A).sum(axis=0) == 0)[0])
    if verbose:
        print(f"    {spec['name']}")
        print(f"      shape={A.shape} expected={expected}  rank={rank}  "
              f"nnz={nnz} (density {nnz/A.size:.2%})")
        if A.shape != expected:
            print(f"      [WARN] shape mismatch")
        if zero_rows:
            print(f"      [WARN] zero rows: {zero_rows}")
        if zero_cols:
            print(f"      [WARN] zero cols: {zero_cols}")
    return dict(shape=A.shape, rank=rank, nnz=nnz,
                zero_rows=zero_rows, zero_cols=zero_cols)


# =============================================================================
# 7. B-VECTOR + SOURCE-PROBS (generic over spec)
# =============================================================================
def compute_B(spec, D, ebin_arr, partic_arr, pscorewt) -> np.ndarray:
    """B[i, r] = 1{S_i in row r's cells} * (2D_i - 1) * pscorewt_i.

    For SUBBIN_CONSTRAINT_ROWS, the cells are the strict sub-bin cells
    (e.g. row low_b1p uses only ebin=1, partic=1).  For other rows, the
    cells are whatever STATE_DEF assigns (may be a pool)."""
    rows = spec["rows"]
    N = len(D)
    w = (2.0 * D - 1.0) * pscorewt
    B = np.zeros((N, len(rows)), dtype=float)
    for r_idx, r_label in enumerate(rows):
        cells = STATE_DEF[r_label]   # already the strict sub-bin cells
        mask = np.zeros(N, dtype=bool)
        for (eb, pa) in cells:
            mask |= (ebin_arr == eb) & (partic_arr == pa)
        B[:, r_idx] = mask.astype(float) * w
    return B


def compute_source_probs(spec, D, ebin_arr, partic_arr, pscorewt) -> np.ndarray:
    """For each col, return P^a(source state) under IPW (control arm)."""
    cols = spec["cols"]
    ctrl = (D == 0)
    wc = pscorewt[ctrl]
    wc_sum = wc.sum() or 1.0
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
# 8. FIRST-STAGE ESTIMATOR (LASSO/Ridge/OLS, GroupKFold or full)
# =============================================================================
def _make_model(estimator: str, seed: int):
    if estimator == "LASSO":
        return LassoCV(alphas=np.logspace(-4, 1, 30), cv=5, eps=1e-4,
                       max_iter=4000, random_state=seed, n_jobs=1)
    if estimator == "Ridge":
        return RidgeCV(alphas=np.logspace(-4, 4, 40))
    if estimator == "OLS":
        return LinearRegression(n_jobs=1)
    raise ValueError(f"Unknown estimator: {estimator!r}")


def estimate_b0(B_obs: np.ndarray, X_raw: np.ndarray,
                person_id: Optional[np.ndarray],
                estimator: str, mode: str = "cv",
                K: int = K_FOLDS, seed: int = RANDOM_SEED,
                verbose: bool = True) -> np.ndarray:
    """First-stage estimator with two modes:
      mode='cv'   : GroupKFold by person (fallback to KFold if person_id None)
      mode='full' : in-sample fit on the whole sample (no cross-fitting)
    Within-fold (or full-sample) StandardScaler.  Returns b_hat (N, k)."""
    n, k = B_obs.shape
    b_hat = np.zeros((n, k))

    if mode == "full":
        sc = StandardScaler()
        Xs = sc.fit_transform(X_raw)
        for j in range(k):
            y = B_obs[:, j]
            if y.std() < 1e-10:
                b_hat[:, j] = y.mean()
                continue
            model = _make_model(estimator, seed)
            model.fit(Xs, y)
            b_hat[:, j] = model.predict(Xs)
        if verbose:
            print(f"      [{estimator}/full] {n:,} obs x {X_raw.shape[1]} "
                  f"features, {k} B-components -- in-sample fit")
        return b_hat

    if mode == "cv":
        if person_id is not None:
            gkf = GroupKFold(n_splits=K)
            folds = list(gkf.split(np.arange(n), groups=person_id))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=K, shuffle=True, random_state=seed)
            folds = list(kf.split(np.arange(n)))
        if verbose:
            print(f"      [{estimator}/cv] {n:,} obs x {X_raw.shape[1]} "
                  f"features, {k} B-components, K={K} folds")
        for fold_i, (tr, te) in enumerate(folds, 1):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_raw[tr])
            X_te = sc.transform(X_raw[te])
            for j in range(k):
                y = B_obs[tr, j]
                if y.std() < 1e-10:
                    b_hat[te, j] = y.mean()
                    continue
                model = _make_model(estimator, seed)
                model.fit(X_tr, y)
                b_hat[te, j] = model.predict(X_te)
        return b_hat

    raise ValueError(f"Unknown mode: {mode!r}")


# =============================================================================
# 9. LP SOLVER + CLP ESTIMATOR
# =============================================================================
LP_DIAG = {'fallback_count': 0, 'binding_nu_count': 0, 'total_solves': 0}


def reset_lp_diagnostics():
    for k in LP_DIAG:
        LP_DIAG[k] = 0


def _solve_lp(c, A_ub_neg, b_ub_neg, bounds, fallback, nu_abs_cap=None):
    """One scipy.linprog call, with diagnostic counters.  Falls back to
    `fallback` if solver fails.  Increments binding_nu_count if any |nu_i|
    is within 1e-6 of the box cap."""
    LP_DIAG['total_solves'] += 1
    res = linprog(c=c, A_ub=A_ub_neg, b_ub=b_ub_neg, bounds=bounds,
                  method='highs', options={'disp': False})
    if res.status != 0:
        LP_DIAG['fallback_count'] += 1
        return fallback
    if nu_abs_cap is not None and np.any(np.abs(res.x) > nu_abs_cap - 1e-6):
        LP_DIAG['binding_nu_count'] += 1
    return res.x


def clp_estimate(q, b_hat, B_obs, A_mat, hub_row=None,
                 nu_lo=NU_LO, nu_hi=NU_HI):
    """sigma_hat(q) = (1/N) sum_i nu_hat_i' B_i,
        where nu_hat_i = argmin_{A^T nu >= q, lo<=nu<=hi} nu' b_hat[i]."""
    n, k = b_hat.shape
    A_ub_neg = -A_mat.T
    b_ub_neg = -np.asarray(q, dtype=float)
    bounds = [(nu_lo, nu_hi)] * k

    fallback = np.full(k, -1.0)
    if hub_row is not None and 0 <= hub_row < k:
        fallback[hub_row] = -2.0

    nu_cap = max(abs(nu_lo), abs(nu_hi)) if (nu_lo is not None and
                                              nu_hi is not None) else None
    contribs = np.zeros(n)
    for i in range(n):
        nu_i = _solve_lp(b_hat[i], A_ub_neg, b_ub_neg, bounds, fallback,
                         nu_abs_cap=nu_cap)
        contribs[i] = float(nu_i @ B_obs[i])
    return contribs.mean(), contribs


def _bounds_for_q(q, b_hat, B_obs, A_mat, person_id, hub_row=None,
                  n_bs=N_BOOTSTRAP):
    """Compute LB, UB, bootstrap CIs for a single q-direction."""
    ub_hat, c_up = clp_estimate(q, b_hat, B_obs, A_mat, hub_row=hub_row)
    ci_ub = list(multiplier_bootstrap_ci(c_up, n_bs=n_bs, person_id=person_id))
    neg_lb, c_dn = clp_estimate(-q, b_hat, B_obs, A_mat, hub_row=hub_row)
    lb_hat = -neg_lb
    ci_lb = [-x for x in
             multiplier_bootstrap_ci(c_dn, n_bs=n_bs, person_id=person_id)[::-1]]
    return dict(lb=lb_hat, ub=ub_hat, width=ub_hat - lb_hat,
                ci_lb=tuple(ci_lb), ci_ub=tuple(ci_ub))


def _bounds_for_col(j, spec, b_hat, B_obs, A_mat, person_id,
                    hub_row=None, n_bs=N_BOOTSTRAP):
    n_cols = len(spec["cols"])
    q = np.zeros(n_cols); q[j] = 1.0
    return _bounds_for_q(q, b_hat, B_obs, A_mat, person_id,
                         hub_row=hub_row, n_bs=n_bs)


# =============================================================================
# 10. KT-STYLE COARSE-COMPOSITE Q VECTORS
#     (TUW / TUWelf / Exit) and the three coarse flow probabilities
#     (2n -> 1r, 1n -> 1r, 0n -> 1r) requested by the user.
# =============================================================================
def kt_composite_q_vectors(spec, p):
    """Three KT composites in beta-form."""
    cols = spec["cols"]
    n = len(cols)
    p00_c = p['p00_c']; p01_c = p['p01_c']
    p10_c = p['p10_c']; p20_c = p['p20_c']

    q_TUW = np.zeros(n)
    q_TUWelf = np.zeros(n)
    q_Exit = np.zeros(n)
    for c_idx, (src, dst) in enumerate(cols):
        if _is_not_working(src) and _is_working(dst):
            q_TUW[c_idx] = 1.0
        if _is_off_welfare(src) and _is_on_welfare(dst):
            q_TUWelf[c_idx] = 1.0
        if STATE_DEF[src] == STATE_DEF["0r"] and _is_off_welfare(dst):
            q_Exit[c_idx] = 1.0
    q_TUW    /= max(1e-12, p00_c + p01_c)
    q_TUWelf /= max(1e-12, p00_c + p10_c + p20_c)
    q_Exit   /= max(1e-12, p01_c)

    return {
        "TUW (Take-Up Work)":      q_TUW,
        "TUWelf (Take-Up Welfare)": q_TUWelf,
        "Exit 0r (welfare-zero -> off welfare)": q_Exit,
    }


def coarse_flow_q_vectors(spec, p):
    """Coarse 2n->1r, 1n->1r, 0n->1r flow probabilities.

    Returns dict {coarse_name: (q_unit, denom)} where q_unit has +1 at every
    col j whose source is a subset of the coarse source AND destination is a
    subset of the coarse dest, 0 elsewhere.  Caller divides by `denom`
    AFTER solving the LP to obtain pi-units.

    Why unit q (not q/denom): the dual LP has constraint A' nu >= q.  Large
    q (e.g., 1/p20_c ~ 10) forces large |nu| and can exhaust the nu box,
    leading to LP infeasibility and silent fallback corruption.  Solving
    with unit q keeps |nu| in a sensible range, then we divide by denom in
    the caller (pi = beta / P^a)."""
    cols = spec["cols"]
    n = len(cols)
    p00_c = p['p00_c']
    p10_c = p['p10_c']
    p20_c = p['p20_c']

    flows = [
        ("pi(2n, 1r)", "2n", "1r", p20_c),
        ("pi(1n, 1r)", "1n", "1r", p10_c),
        ("pi(0n, 1r)", "0n", "1r", p00_c),
    ]
    out = {}
    for name, src_label, dst_label, denom in flows:
        q_unit = np.zeros(n)
        for c_idx, (src, dst) in enumerate(cols):
            if _is_subset_of(src, src_label) and _is_subset_of(dst, dst_label):
                q_unit[c_idx] = 1.0
        out[name] = (q_unit, denom)
    return out


def find_target_cols(spec, target_src_label: str, target_dst_label: str):
    """Return list of column indices in spec whose (src, dst) is a subset
    of the target labels."""
    target_src_cells = STATE_DEF[target_src_label]
    target_dst_cells = STATE_DEF[target_dst_label]
    matching = []
    for c_idx, (src, dst) in enumerate(spec["cols"]):
        if (STATE_DEF[src] <= target_src_cells
                and STATE_DEF[dst] <= target_dst_cells):
            matching.append(c_idx)
    return matching


# =============================================================================
# 11. PER-SPEC PIPELINE  (one spec, one config)
# =============================================================================
def run_spec_config(spec_key: str, spec: dict, data: tuple, p: dict,
                    estimator: str, feature_set: str, mode: str = "cv",
                    n_bs: int = N_BOOTSTRAP, verbose: bool = True):
    """Run the full pipeline for one (spec, estimator, feature_set, mode).
    Returns dict with 'individual', 'composite', 'flow', 'A_diag'."""
    D, ebin_arr, partic_arr, _state_row, df_incl, person_id, pscorewt = data

    # -- A and basic objects
    A_mat = build_A(spec)
    A_diag = verify_A(A_mat, spec, verbose=False)
    B_obs = compute_B(spec, D, ebin_arr, partic_arr, pscorewt)
    src_probs = compute_source_probs(spec, D, ebin_arr, partic_arr, pscorewt)

    # -- features
    avail = [v for v in COV_VARS if v in df_incl.columns]
    X_base = df_incl[avail].fillna(0).to_numpy(float)
    if feature_set == "base":
        X_raw = X_base
    elif feature_set == "econ":
        X_b, X_e, _, _ = engineer_features_econ(df_incl, cov_vars=COV_VARS)
        X_raw = np.hstack([X_b, X_e]) if X_e.size else X_b
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    if verbose:
        print(f"\n    [{spec_key} / {estimator}-{feature_set}-{mode}]")
        print(f"      A: {A_diag['shape']} rank={A_diag['rank']}  "
              f"X: {X_raw.shape}")

    # -- first stage
    b_hat = estimate_b0(B_obs, X_raw, person_id, estimator=estimator,
                        mode=mode, verbose=verbose)
    if verbose:
        print(f"      max|E[B]-E[b_hat]| = "
              f"{np.max(np.abs(B_obs.mean(axis=0) - b_hat.mean(axis=0))):.2e}")

    # -- hub row (state 0p) for LP fallback
    hub_row = None
    for r_idx, r_label in enumerate(spec["rows"]):
        if r_label in ("0p", "0r"):
            hub_row = r_idx
            break

    reset_lp_diagnostics()

    # -- individual beta bounds (every column)
    individual = {}
    for j in range(len(spec["cols"])):
        r = _bounds_for_col(j, spec, b_hat, B_obs, A_mat, person_id,
                            hub_row=hub_row, n_bs=n_bs)
        sp = src_probs[j]
        if sp > 1e-12:
            r['lb_pi'] = max(0., min(1., r['lb'] / sp))
            r['ub_pi'] = max(0., min(1., r['ub'] / sp))
        else:
            r['lb_pi'] = float('nan'); r['ub_pi'] = float('nan')
        r['source_pa'] = float(sp)
        beta_name = f"beta({spec['cols'][j][0]},{spec['cols'][j][1]})"
        individual[beta_name] = r

    # -- KT composite bounds
    composite = {}
    for cname, q_beta in kt_composite_q_vectors(spec, p).items():
        r = _bounds_for_q(q_beta, b_hat, B_obs, A_mat, person_id,
                          hub_row=hub_row, n_bs=n_bs)
        r['lb_clip'] = max(0., min(1., r['lb']))
        r['ub_clip'] = max(0., min(1., r['ub']))
        composite[cname] = r

    # -- coarse flow bounds (pi(2n,1r), pi(1n,1r), pi(0n,1r))
    # Solve LP with UNIT q (beta-units), then divide everything by denom
    # to get pi-units.  We keep both beta-sum and pi forms in the result.
    flow = {}
    for fname, (q_unit, denom) in coarse_flow_q_vectors(spec, p).items():
        if q_unit.sum() == 0 or denom <= 1e-12:
            flow[fname] = dict(lb=float('nan'), ub=float('nan'),
                               width=float('nan'),
                               ci_lb=(float('nan'),)*2,
                               ci_ub=(float('nan'),)*2,
                               lb_pi=float('nan'), ub_pi=float('nan'),
                               ci_lb_pi=(float('nan'),)*2,
                               ci_ub_pi=(float('nan'),)*2,
                               denom=float(denom), n_active=0)
            continue
        # LP in beta-units (q_unit has just 0s and 1s)
        r = _bounds_for_q(q_unit, b_hat, B_obs, A_mat, person_id,
                          hub_row=hub_row, n_bs=n_bs)
        # r['lb'], r['ub'] are beta-sum bounds; r['ci_*'] also in beta-units.
        # Rescale to pi-units (divide by denom).
        r['lb_beta_sum'] = r['lb']
        r['ub_beta_sum'] = r['ub']
        r['width_beta_sum'] = r['width']
        r['ci_lb_beta'] = r['ci_lb']
        r['ci_ub_beta'] = r['ci_ub']
        r['lb_pi'] = max(0., min(1., r['lb'] / denom))
        r['ub_pi'] = max(0., min(1., r['ub'] / denom))
        r['ci_lb_pi'] = (max(0., min(1., r['ci_lb'][0] / denom)),
                         max(0., min(1., r['ci_lb'][1] / denom)))
        r['ci_ub_pi'] = (max(0., min(1., r['ci_ub'][0] / denom)),
                         max(0., min(1., r['ci_ub'][1] / denom)))
        r['denom'] = float(denom)
        r['n_active'] = int(q_unit.astype(bool).sum())
        flow[fname] = r

    if verbose:
        fb = LP_DIAG['fallback_count']
        nucap = LP_DIAG['binding_nu_count']
        tot = LP_DIAG['total_solves']
        print(f"      LP: {tot:,} solves, fallback={fb}, |nu|=cap={nucap}")
        if fb > 0:
            print(f"      *** WARNING: LP fallback fired {fb} times "
                  f"({100*fb/max(1,tot):.1f}%) - bounds may be biased ***")
        if nucap > 0:
            print(f"      *** WARNING: nu cap binding in {nucap} solves "
                  f"({100*nucap/max(1,tot):.1f}%) - widen NU_LO/NU_HI ***")

    return dict(spec_key=spec_key, spec_name=spec['name'],
                A_shape=A_diag['shape'], rank=A_diag['rank'],
                individual=individual, composite=composite, flow=flow,
                lp_diag=dict(LP_DIAG))


# =============================================================================
# 12. CSV WRITER
# =============================================================================
def results_to_rows(spec_key: str, config_label: str, res: dict):
    """Flatten one spec-config result into a list of rows for CSV output."""
    rows = []
    base_meta = dict(
        spec_key=spec_key, config=config_label,
        spec_name=res['spec_name'],
        A_rows=res['A_shape'][0], A_cols=res['A_shape'][1],
        rank=res['rank'],
        lp_total=res['lp_diag']['total_solves'],
        lp_fallback=res['lp_diag']['fallback_count'],
        lp_nu_cap=res['lp_diag']['binding_nu_count'],
    )
    # individual betas
    for beta_name, r in res['individual'].items():
        rows.append(dict(
            kind="beta", target=beta_name, **base_meta,
            lb_beta=r['lb'], ub_beta=r['ub'], width_beta=r['width'],
            ci_lb_lo=r['ci_lb'][0], ci_lb_hi=r['ci_lb'][1],
            ci_ub_lo=r['ci_ub'][0], ci_ub_hi=r['ci_ub'][1],
            source_pa=r.get('source_pa', float('nan')),
            lb_pi=r['lb_pi'], ub_pi=r['ub_pi'],
        ))
    # composite bounds
    for cname, r in res['composite'].items():
        rows.append(dict(
            kind="composite", target=cname, **base_meta,
            lb_beta=r['lb'], ub_beta=r['ub'], width_beta=r['width'],
            ci_lb_lo=r['ci_lb'][0], ci_lb_hi=r['ci_lb'][1],
            ci_ub_lo=r['ci_ub'][0], ci_ub_hi=r['ci_ub'][1],
            source_pa=float('nan'),
            lb_pi=r['lb_clip'], ub_pi=r['ub_clip'],
        ))
    # coarse flows (both beta-sum and pi units)
    for fname, r in res['flow'].items():
        # For flow rows, lb_beta = sum of beta cols (LP output in beta-units),
        # CIs are also in beta-units; pi-unit columns lb_pi/ub_pi/ci_*_pi are
        # the corresponding /denom values, clipped to [0, 1].
        ci_lb_pi = r.get('ci_lb_pi', (float('nan'), float('nan')))
        ci_ub_pi = r.get('ci_ub_pi', (float('nan'), float('nan')))
        rows.append(dict(
            kind="flow", target=fname, **base_meta,
            lb_beta=r.get('lb_beta_sum', r['lb']),
            ub_beta=r.get('ub_beta_sum', r['ub']),
            width_beta=r.get('width_beta_sum', r['width']),
            ci_lb_lo=r['ci_lb'][0], ci_lb_hi=r['ci_lb'][1],
            ci_ub_lo=r['ci_ub'][0], ci_ub_hi=r['ci_ub'][1],
            source_pa=r.get('denom', float('nan')),
            lb_pi=r['lb_pi'], ub_pi=r['ub_pi'],
            ci_lb_pi_lo=ci_lb_pi[0], ci_lb_pi_hi=ci_lb_pi[1],
            ci_ub_pi_lo=ci_ub_pi[0], ci_ub_pi_hi=ci_ub_pi[1],
        ))
    return rows


# =============================================================================
# 13. SUMMARY PRINT (per phase)
# =============================================================================
def print_flow_summary(all_results, label="Phase"):
    """Cross-config summary for the three coarse flow bounds."""
    print(f"\n\n{'='*92}")
    print(f"{label} -- coarse flow bound comparison "
          f"(pi-units, clipped to [0,1])")
    print('='*92)
    print(f"  {'config':<28}  {'spec':<14}  {'shape':<8}  "
          f"{'pi(2n,1r)':<22}  {'pi(1n,1r)':<22}  {'pi(0n,1r)':<22}")
    print('-'*120)
    for cfg_label, res in all_results.items():
        spec_key = res['spec_key']
        sh = f"{res['A_shape'][0]}x{res['A_shape'][1]}"
        flow = res['flow']

        def _fmt(name):
            d = flow.get(name)
            if d is None or np.isnan(d.get('lb_pi', float('nan'))):
                return "[--]"
            return f"[{d['lb_pi']:.3f},{d['ub_pi']:.3f}] w={d['width']:.3f}"
        f1 = _fmt("pi(2n, 1r)")
        f2 = _fmt("pi(1n, 1r)")
        f3 = _fmt("pi(0n, 1r)")
        print(f"  {cfg_label:<28}  {spec_key:<14}  {sh:<8}  "
              f"{f1:<22}  {f2:<22}  {f3:<22}")


# =============================================================================
# 14. PHASE 1: 13x33 with {LASSO, Ridge, OLS} x {base, econ}, GroupKFold
# =============================================================================
PHASE1_CONFIGS = [
    ("LASSO", "base", "cv"),
    ("LASSO", "econ", "cv"),
    ("Ridge", "base", "cv"),
    ("Ridge", "econ", "cv"),
    ("OLS",   "base", "cv"),
    ("OLS",   "econ", "cv"),
]


def run_phase1(data, p):
    print("\n\n" + "#"*88)
    print("# PHASE 1 -- 13x33 spec1 with {LASSO, Ridge, OLS} x {base, econ}")
    print("# GroupKFold cross-fitting by person_id")
    print("#"*88)
    spec_key = "spec1"
    spec = SPECS[spec_key]
    all_csv_rows = []
    flow_summary = {}
    for est, fset, mode in PHASE1_CONFIGS:
        cfg_label = f"phase1_{est}_{fset}_{mode}"
        print(f"\n  ---- {cfg_label} ----")
        res = run_spec_config(spec_key, spec, data, p,
                              estimator=est, feature_set=fset, mode=mode)
        flow_summary[cfg_label] = res
        all_csv_rows.extend(results_to_rows(spec_key, cfg_label, res))

    pd.DataFrame(all_csv_rows).to_csv(PHASE1_CSV, index=False)
    print(f"\n  Wrote PHASE 1 CSV: {PHASE1_CSV}")
    print_flow_summary(flow_summary, label="PHASE 1")
    return flow_summary, all_csv_rows


# =============================================================================
# 15. PHASE 2: All 21 specs with LASSO x {base, econ}, GroupKFold
# =============================================================================
PHASE2_SPECS = [
    "spec1", "spec2", "spec3", "spec4", "spec5", "spec6", "spec7", "spec8",
    "spec9", "spec10", "spec11", "spec12", "spec13", "spec14", "spec14_alt",
    "spec15", "spec16", "spec17", "spec18", "spec19", "spec20",
]
PHASE2_CONFIGS = [("LASSO", "base", "cv"), ("LASSO", "econ", "cv")]


def run_phase2(data, p):
    print("\n\n" + "#"*88)
    print(f"# PHASE 2 -- {len(PHASE2_SPECS)} specs x LASSO x {{base, econ}} "
          f"GroupKFold")
    print("#"*88)
    print("Specs to run:")
    for sk in PHASE2_SPECS:
        print(f"  {sk:<14}: {SPECS[sk]['name']}")
    if "spec12" in PHASE2_SPECS:
        print(f"\n  NOTE on spec 12: {SPEC_12_NOTE}")

    # Pre-verify all specs build cleanly
    print("\n  [Pre-flight] verifying all spec A matrices build cleanly:")
    for sk in PHASE2_SPECS:
        verify_A(build_A(SPECS[sk]), SPECS[sk], verbose=True)

    all_csv_rows = []
    flow_summary = {}
    for sk in PHASE2_SPECS:
        spec = SPECS[sk]
        for est, fset, mode in PHASE2_CONFIGS:
            cfg_label = f"phase2_{sk}_{est}_{fset}_{mode}"
            print(f"\n  ---- {cfg_label} ----")
            res = run_spec_config(sk, spec, data, p,
                                  estimator=est, feature_set=fset, mode=mode)
            flow_summary[cfg_label] = res
            all_csv_rows.extend(results_to_rows(sk, cfg_label, res))

    pd.DataFrame(all_csv_rows).to_csv(PHASE2_CSV, index=False)
    print(f"\n  Wrote PHASE 2 CSV: {PHASE2_CSV}")
    print_flow_summary(flow_summary, label="PHASE 2")
    return flow_summary, all_csv_rows


# =============================================================================
# 16. PHASE 3: Specs 1-17 with OLS + base + full sample (no cross-fitting)
# =============================================================================
PHASE3_SPECS = [
    "spec1", "spec2", "spec3", "spec4", "spec5", "spec6", "spec7", "spec8",
    "spec9", "spec10", "spec11", "spec12", "spec13", "spec14", "spec14_alt",
    "spec15", "spec16", "spec17",
]


def run_phase3(data, p):
    print("\n\n" + "#"*88)
    print(f"# PHASE 3 -- {len(PHASE3_SPECS)} specs x OLS x base x FULL sample "
          f"(no cross-fitting)")
    print("#"*88)
    all_csv_rows = []
    flow_summary = {}
    for sk in PHASE3_SPECS:
        spec = SPECS[sk]
        cfg_label = f"phase3_{sk}_OLS_base_full"
        print(f"\n  ---- {cfg_label} ----")
        res = run_spec_config(sk, spec, data, p,
                              estimator="OLS", feature_set="base",
                              mode="full")
        flow_summary[cfg_label] = res
        all_csv_rows.extend(results_to_rows(sk, cfg_label, res))

    pd.DataFrame(all_csv_rows).to_csv(PHASE3_CSV, index=False)
    print(f"\n  Wrote PHASE 3 CSV: {PHASE3_CSV}")
    print_flow_summary(flow_summary, label="PHASE 3")
    return flow_summary, all_csv_rows


# =============================================================================
# 17. MAIN
# =============================================================================
def main():
    t0 = time.time()
    print("="*88)
    print("CLP_granular_final_combined.py")
    print("="*88)
    print(f"  N_BOOTSTRAP={N_BOOTSTRAP}  K_FOLDS={K_FOLDS}  SEED={RANDOM_SEED}")
    print(f"  RUN_PHASE_1={RUN_PHASE_1}  RUN_PHASE_2={RUN_PHASE_2}  "
          f"RUN_PHASE_3={RUN_PHASE_3}")

    print("\n[STAGE 0] Load data via prepare_jf_data_granular() (one-time)")
    data = prepare_jf_data_granular()
    p, _ = load_table4_mat()

    if RUN_PHASE_1:
        run_phase1(data, p)
    if RUN_PHASE_2:
        run_phase2(data, p)
    if RUN_PHASE_3:
        run_phase3(data, p)

    print(f"\nTotal runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
