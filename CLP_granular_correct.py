"""
CLP_granular_correct.py
=======================
Multi-specification CLP file -- runs 9 different (rows x cols) layouts to
explore the granularity vs. identification trade-off.

==================================================================================
WHY MULTIPLE SPECIFICATIONS
==================================================================================
The trade-off:
  - More columns (finer transition splits) -> more parameters -> wider bounds.
  - Fewer rows (more pooling of observable states) -> fewer constraints.
  - Best (rows, cols) layout is empirical: tighter individual bounds with
    LP-slack noise still controlled.  This file evaluates 9 candidate layouts
    side-by-side using the same data, propensity score, and first-stage LASSO.

==================================================================================
THE 9 SPECIFICATIONS
==================================================================================
Each spec is identified by a name like "spec1" .. "spec9".  Listed below are
the rows (conservation equations), the transitions (beta columns), and the
expected (rows x cols) shape.  Note three small discrepancies between the
user's stated dimensions and my count of the listed transitions; I proceed
with the listed-transition count and flag the discrepancy in the verification.

Convention: "0r" = "0p" (zero-earnings on welfare); "2p" = "2u" (above-FPL
on-welfare = latent underreporter under AFDC; effectively zero under JF by
Lemma 2).  See CLP_granular_correct documentation about why pooling 1r as
the reference is essential to avoid the underreporter-contamination problem.

  Spec 1 (13 x 33) -- the original CLP_granular_final_group layout
    Rows: 0n, b1n..b5n, b6n..b8n, 0p, b6p, b7p, b8p
    Transitions:
       0n  -> b_jr   (j=1..5)        [G1, 5]
       0r  -> 0n                      [G2, 1]
       b_jn-> 1r     (j=6,7,8)        [G3, 3]
       0r  -> b_jn   (j=6,7,8)        [G4, 3]
       0r  -> b_jr   (j=1..5)         [G5, 5]
       0r  -> b_jn   (j=1..5)         [G6, 5]
       b_jn-> 1r     (j=1..5)         [G7, 5]
       0r  -> b_ju   (j=6,7,8)        [G8, 3]
       b_ju-> 1r     (j=6,7,8)        [G9, 3]
    Total: 5+1+3+3+5+5+5+3+3 = 33 cols.

  Spec 2 (13 x 25) -- 1r pooled in G1, G5
    Same rows.  Transitions:
       0n  -> 1r                      [G1, 1 -- POOLED]
       0r  -> 0n                      [G2, 1]
       b_jn-> 1r    (j=6,7,8)         [G3, 3]
       0r  -> b_jn  (j=6,7,8)         [G4, 3]
       0r  -> 1r                      [G5, 1 -- POOLED]
       0r  -> b_jn  (j=1..5)          [G6, 5]
       b_jn-> 1r    (j=1..5)          [G7, 5]
       0r  -> b_ju  (j=6,7,8)         [G8, 3]
       b_ju-> 1r    (j=6,7,8)         [G9, 3]
    Total: 1+1+3+3+1+5+5+3+3 = 25.

  Spec 3 (13 x 37; user said 39) -- G3 fully split
    Same rows.  Transitions:
       0n  -> 1r                              [G1, 1]
       0r  -> 0n                              [G2, 1]
       b_jn-> b_ir  (j=6..8, i=1..5)          [G3, 15 -- FULL split]
       0r  -> b_jn  (j=6..8)                  [G4, 3]
       0r  -> 1r                              [G5, 1]
       0r  -> b_jn  (j=1..5)                  [G6, 5]
       b_jn-> 1r    (j=1..5)                  [G7, 5]
       0r  -> b_ju  (j=6..8)                  [G8, 3]
       b_ju-> 1r    (j=6..8)                  [G9, 3]
    Total: 1+1+15+3+1+5+5+3+3 = 37 (user listed 39).

  Spec 4 (13 x 57; user said 59) -- G3 and G7 both fully split
    Same rows.  Transitions:
       (Spec 3 with G7: b_jn -> b_ir for j=1..5, i=1..5 (25 cols, FULL))
    Total: 1+1+15+3+1+5+25+3+3 = 57 (user listed 59).

  Spec 5 (11 x 53)
    Rows: 0n, b1n..b5n, b6n..b8n, 0p, 2p
    Transitions:
       Like Spec 4 but G8 pools to "0r -> 2u" (1 col) and
       G9 pools to "2u -> 1r" (1 col).
    Total: 1+1+15+3+1+5+25+1+1 = 53.

  Spec 6 (9 x 29)
    Rows: 0n, 1n, b6n..b8n, 0p, b6p, b7p, b8p
    Transitions:
       0n  -> 1r                              [G1, 1]
       0r  -> 0n                              [G2, 1]
       b_jn-> b_ir  (j=6..8, i=1..5)          [G3, 15]
       0r  -> b_jn  (j=6..8)                  [G4, 3]
       0r  -> 1r                              [G5, 1]
       0r  -> 1n                              [G6, 1 -- POOLED 1n source]
       1n  -> 1r                              [G7, 1 -- POOLED]
       0r  -> b_ju  (j=6..8)                  [G8, 3]
       b_ju-> 1r    (j=6..8)                  [G9, 3]
    Total: 1+1+15+3+1+1+1+3+3 = 29.

  Spec 7 (9 x 33)
    Rows: 0n, 1n, b6n..b8n, 0p, b6p, b7p, b8p
    Transitions: like Spec 6 but G7 destination-granular:
       1n -> b_ir  (i=1..5)                    [G7, 5 cols]
    Total: 1+1+15+3+1+1+5+3+3 = 33.

  Spec 8 (7 x 25; user said 6 rows)
    Rows: 0n, 1n, b6n..b8n, 0p, 2p
    Transitions:
       0n  -> 1r                              [G1, 1]
       0r  -> 0n                              [G2, 1]
       b_jn-> b_ir  (j=6..8, i=1..5)          [G3, 15]
       0r  -> b_jn  (j=6..8)                  [G4, 3]
       0r  -> 1r                              [G5, 1]
       0r  -> 1n                              [G6, 1]
       1n  -> 1r                              [G7, 1]
       0r  -> 2u                              [G8, 1]
       2u  -> 1r                              [G9, 1]
    Total: 1+1+15+3+1+1+1+1+1 = 25.  Rows: 7 (user listed 6).

  Spec 9 (5 x 13)
    Rows: 0n, 1n, 2n, 0p, 2p
    Transitions:
       0n  -> 1r                              [G1, 1]
       0r  -> 0n                              [G2, 1]
       2n  -> b_ir  (i=1..5)                  [G3, 5 -- destination granular]
       0r  -> 2n                              [G4, 1]
       0r  -> 1r                              [G5, 1]
       0r  -> 1n                              [G6, 1]
       1n  -> 1r                              [G7, 1]
       0r  -> 2u                              [G8, 1]
       2u  -> 1r                              [G9, 1]
    Total: 1+1+5+1+1+1+1+1+1 = 13.

==================================================================================
PIPELINE (same for every spec)
==================================================================================
- Data: prepare_jf_data_granular() from CLP_granular_final_group (9-bin
  classification: 0, 5 below-FPL bins, 3 above-FPL bins).
- First stage: cross-fit LASSO of each B-component on 28 base covariates,
  GroupKFold by person, K=5 folds, 5-fold internal CV for alpha.
- LP estimator: scipy.optimize.linprog (HiGHS) per observation.
- Bounds reported: 9 individual betas (or all if requested) + 3 KT-style
  composites (Take-Up Work, Take-Up Welfare, Exit 0r).
- 95% CIs: person-clustered Exp(1) multiplier bootstrap, 200 draws.
"""

import time
import warnings
from typing import Dict, List, Tuple, FrozenSet

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# Reuse data preparation, propensity score, bootstrap helper from baseline.
from CLP_granular_final_group import (
    PSCORE_VARS, JF_CONFIG, COV_VARS,
    JF_DTA_PATH, POLICY_RULES_PATH, TABLE4_MAT_PATH,
    K_FOLDS, RANDOM_SEED,
    fit_pscore_logit, load_table4_mat,
    prepare_jf_data_granular,
    multiplier_bootstrap_ci,
)

N_BOOTSTRAP = 200
np.random.seed(RANDOM_SEED)


# =============================================================================
# STATE LABEL DEFINITIONS
# =============================================================================
# Each label maps to a frozenset of (ebin, partic) cells it covers.  Aliases:
#   "0r" == "0p"                      (zero earnings on welfare = truthful)
#   "1p" == "1r"                      (below-FPL on welfare = truthful)
#   "2p" == "2u"                      (above-FPL on welfare = underreporter)
#   "b6u" == "b6p", etc.

def _frozencells(*pairs):
    return frozenset(pairs)


STATE_DEF: Dict[str, FrozenSet[Tuple[int, int]]] = {
    # Single observable cells -- non-welfare
    "0n":  _frozencells((0, 0)),
    "b1n": _frozencells((1, 0)), "b2n": _frozencells((2, 0)),
    "b3n": _frozencells((3, 0)), "b4n": _frozencells((4, 0)),
    "b5n": _frozencells((5, 0)),
    "b6n": _frozencells((6, 0)), "b7n": _frozencells((7, 0)),
    "b8n": _frozencells((8, 0)),
    # Single observable cells -- on welfare
    "0p":  _frozencells((0, 1)),
    "0r":  _frozencells((0, 1)),                   # alias for 0p
    "b1r": _frozencells((1, 1)), "b2r": _frozencells((2, 1)),
    "b3r": _frozencells((3, 1)), "b4r": _frozencells((4, 1)),
    "b5r": _frozencells((5, 1)),
    "b6u": _frozencells((6, 1)), "b7u": _frozencells((7, 1)),
    "b8u": _frozencells((8, 1)),
    "b6p": _frozencells((6, 1)),                   # alias for b6u
    "b7p": _frozencells((7, 1)),                   # alias for b7u
    "b8p": _frozencells((8, 1)),                   # alias for b8u
    # Pooled / coarse labels
    "1n":  _frozencells((1, 0), (2, 0), (3, 0), (4, 0), (5, 0)),
    "2n":  _frozencells((6, 0), (7, 0), (8, 0)),
    "1r":  _frozencells((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
    "1p":  _frozencells((1, 1), (2, 1), (3, 1), (4, 1), (5, 1)),
    "2u":  _frozencells((6, 1), (7, 1), (8, 1)),
    "2p":  _frozencells((6, 1), (7, 1), (8, 1)),
}


# =============================================================================
# CRITERIA HELPERS  (used to build composite q vectors)
# =============================================================================
def _is_not_working(s):
    return all(eb == 0 for (eb, _) in STATE_DEF[s])

def _is_working(s):
    return all(eb >= 1 for (eb, _) in STATE_DEF[s])

def _is_off_welfare(s):
    return all(pa == 0 for (_, pa) in STATE_DEF[s])

def _is_on_welfare(s):
    return all(pa == 1 for (_, pa) in STATE_DEF[s])

def _is_0r(s):
    return STATE_DEF[s] == STATE_DEF["0r"]


# =============================================================================
# THE 9 SPECIFICATIONS
# =============================================================================
def _spec1():
    return dict(
        name="Spec 1: 13 x 33 (= original CLP_granular_final_group)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", f"b{j}r") for j in range(1, 6)]                           # G1: 5
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{j}n", "1r") for j in range(6, 9)]                          # G3: 3
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", f"b{j}r") for j in range(1, 6)]                          # G5: 5
            + [("0r", f"b{j}n") for j in range(1, 6)]                          # G6: 5
            + [(f"b{j}n", "1r") for j in range(1, 6)]                          # G7: 5
            + [("0r", f"b{j}u") for j in range(6, 9)]                          # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                          # G9: 3
        ),
        expected_shape=(13, 33),
    )


def _spec2():
    return dict(
        name="Spec 2: 13 x 25 (1r pooled in G1 and G5)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{j}n", "1r") for j in range(6, 9)]                          # G3: 3
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", f"b{j}n") for j in range(1, 6)]                          # G6: 5
            + [(f"b{j}n", "1r") for j in range(1, 6)]                          # G7: 5
            + [("0r", f"b{j}u") for j in range(6, 9)]                          # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                          # G9: 3
        ),
        expected_shape=(13, 25),
    )


def _spec3():
    return dict(
        name="Spec 3: 13 x 37 (G3 fully split; user listed 39)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                      # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", f"b{j}n") for j in range(1, 6)]                          # G6: 5
            + [(f"b{j}n", "1r") for j in range(1, 6)]                          # G7: 5
            + [("0r", f"b{j}u") for j in range(6, 9)]                          # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                          # G9: 3
        ),
        expected_shape=(13, 37),
    )


def _spec4():
    return dict(
        name="Spec 4: 13 x 57 (G3 and G7 both fully split; user listed 59)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                      # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", f"b{j}n") for j in range(1, 6)]                          # G6: 5
            + [(f"b{j}n", f"b{i}r")
               for j in range(1, 6) for i in range(1, 6)]                      # G7: 25 (FULL)
            + [("0r", f"b{j}u") for j in range(6, 9)]                          # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                          # G9: 3
        ),
        expected_shape=(13, 57),
    )


def _spec5():
    return dict(
        name="Spec 5: 11 x 53 (above-FPL on-welfare pooled into 2p; G3, G7 full)",
        rows=["0n", "b1n", "b2n", "b3n", "b4n", "b5n",
              "b6n", "b7n", "b8n", "0p", "2p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                      # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", f"b{j}n") for j in range(1, 6)]                          # G6: 5
            + [(f"b{j}n", f"b{i}r")
               for j in range(1, 6) for i in range(1, 6)]                      # G7: 25
            + [("0r", "2u")]                                                    # G8: 1 (pooled)
            + [("2u", "1r")]                                                    # G9: 1 (pooled)
        ),
        expected_shape=(11, 53),
    )


def _spec6():
    return dict(
        name="Spec 6: 9 x 29 (1n pooled; G3 fully split; G7 pooled)",
        rows=["0n", "1n", "b6n", "b7n", "b8n",
              "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                      # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", "1n")]                                                    # G6: 1 (pooled)
            + [("1n", "1r")]                                                    # G7: 1 (pooled)
            + [("0r", f"b{j}u") for j in range(6, 9)]                          # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                          # G9: 3
        ),
        expected_shape=(9, 29),
    )


def _spec7():
    return dict(
        name="Spec 7: 9 x 33 (1n source pooled; 1n -> b_ir destination split)",
        rows=["0n", "1n", "b6n", "b7n", "b8n",
              "0p", "b6p", "b7p", "b8p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                      # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", "1n")]                                                    # G6: 1 (pooled)
            + [("1n", f"b{i}r") for i in range(1, 6)]                          # G7: 5 (1n source pooled)
            + [("0r", f"b{j}u") for j in range(6, 9)]                          # G8: 3
            + [(f"b{j}u", "1r") for j in range(6, 9)]                          # G9: 3
        ),
        expected_shape=(9, 33),
    )


def _spec8():
    return dict(
        name="Spec 8: 7 x 25 (1n and 2u/2p pooled; user listed 6 rows)",
        rows=["0n", "1n", "b6n", "b7n", "b8n", "0p", "2p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [(f"b{k}n", f"b{i}r")
               for k in range(6, 9) for i in range(1, 6)]                      # G3: 15
            + [("0r", f"b{j}n") for j in range(6, 9)]                          # G4: 3
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", "1n")]                                                    # G6: 1
            + [("1n", "1r")]                                                    # G7: 1
            + [("0r", "2u")]                                                    # G8: 1
            + [("2u", "1r")]                                                    # G9: 1
        ),
        expected_shape=(7, 25),
    )


def _spec9():
    return dict(
        name="Spec 9: 5 x 13 (most pooled; coarse 5-row + 2n -> b_ir destination split)",
        rows=["0n", "1n", "2n", "0p", "2p"],
        cols=(
            [("0n", "1r")]                                                     # G1: 1
            + [("0r", "0n")]                                                    # G2: 1
            + [("2n", f"b{i}r") for i in range(1, 6)]                          # G3: 5 (2n source pooled, dest split)
            + [("0r", "2n")]                                                    # G4: 1
            + [("0r", "1r")]                                                    # G5: 1
            + [("0r", "1n")]                                                    # G6: 1
            + [("1n", "1r")]                                                    # G7: 1
            + [("0r", "2u")]                                                    # G8: 1
            + [("2u", "1r")]                                                    # G9: 1
        ),
        expected_shape=(5, 13),
    )


SPECS = {
    "spec1": _spec1(), "spec2": _spec2(), "spec3": _spec3(),
    "spec4": _spec4(), "spec5": _spec5(), "spec6": _spec6(),
    "spec7": _spec7(), "spec8": _spec8(), "spec9": _spec9(),
}


# =============================================================================
# A-MATRIX BUILDER  (generic: works for any spec)
# =============================================================================
def build_A(spec) -> np.ndarray:
    """
    For each row r and each column (s_src, s_dst):
      - +1 if dst's covered cells overlap with r's AND src's don't (inflow)
      - -1 if src's covered cells overlap with r's AND dst's don't (outflow)
      -  0 otherwise (no involvement, OR pure stay transition)
    """
    rows, cols = spec["rows"], spec["cols"]
    A = np.zeros((len(rows), len(cols)), dtype=float)
    for r_idx, r_label in enumerate(rows):
        r_cells = STATE_DEF[r_label]
        for c_idx, (src, dst) in enumerate(cols):
            src_cells = STATE_DEF[src]
            dst_cells = STATE_DEF[dst]
            src_in = bool(src_cells & r_cells)
            dst_in = bool(dst_cells & r_cells)
            if dst_in and not src_in:
                A[r_idx, c_idx] = +1.0
            elif src_in and not dst_in:
                A[r_idx, c_idx] = -1.0
            # else 0 (neither, or stay -- both src and dst cover this row's cells)
    return A


def verify_A(A, spec):
    """Verify A's shape, rank, and entries; print conservation equations."""
    rows, cols = spec["rows"], spec["cols"]
    expected = spec["expected_shape"]
    assert A.shape == (len(rows), len(cols)), \
        f"Shape {A.shape} vs ({len(rows)}, {len(cols)})"
    assert set(np.unique(A)).issubset({-1.0, 0.0, 1.0})
    rank = np.linalg.matrix_rank(A)
    nnz = int((A != 0).sum())
    print(f"  {spec['name']}")
    print(f"    Shape: {A.shape},  expected: {expected},  rank: {rank},  "
          f"nnz: {nnz} (density {nnz/A.size:.2%})")
    if A.shape != expected:
        print(f"    [WARN] Shape doesn't match expected {expected}; "
              f"see docstring for noted user-stated discrepancies.")
    # zero-row / zero-col checks
    zr = np.where(np.abs(A).sum(axis=1) == 0)[0]
    zc = np.where(np.abs(A).sum(axis=0) == 0)[0]
    if len(zr) > 0:
        print(f"    [WARN] zero rows: {zr.tolist()}")
    if len(zc) > 0:
        print(f"    [WARN] zero cols: {zc.tolist()}")


def print_conservation_equations(A, spec):
    """Print the conservation-of-mass equation for each row."""
    rows, cols = spec["rows"], spec["cols"]
    col_names = [f"beta({s},{d})" for (s, d) in cols]
    print(f"\n  Conservation equations for {spec['name']}:")
    for r_idx, r_label in enumerate(rows):
        plus_idx  = [j for j in range(len(cols)) if A[r_idx, j] == +1]
        minus_idx = [j for j in range(len(cols)) if A[r_idx, j] == -1]
        rhs = " + ".join(col_names[j] for j in plus_idx) if plus_idx else "0"
        if minus_idx:
            rhs += " - " + " - ".join(col_names[j] for j in minus_idx)
        print(f"    Row {r_idx:2d} ({r_label:>5}): p^J - p^A = {rhs}")


# =============================================================================
# B-VECTOR  (one component per row of A)
# =============================================================================
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
# CROSS-FITTED LASSO FIRST STAGE  (GroupKFold by person)
# =============================================================================
def estimate_b0_lasso(B_obs, X_raw, person_id, K=K_FOLDS, seed=RANDOM_SEED):
    """
    For each B-component, cross-fit a LassoCV with internal 5-fold CV for alpha.
    Outer split: GroupKFold by person_id (matches CLP_granular_final_group's
    behaviour).  Within-fold StandardScaler.
    """
    n, k = B_obs.shape
    b_hat = np.zeros((n, k))
    if person_id is not None:
        gkf = GroupKFold(n_splits=K)
        folds = list(gkf.split(np.arange(n), groups=person_id))
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=K, shuffle=True, random_state=seed)
        folds = list(kf.split(np.arange(n)))

    print(f"    [LASSO] {n:,} obs x {X_raw.shape[1]} features, "
          f"{k} B-components, K={K} folds")
    for fold_i, (tr_idx, te_idx) in enumerate(folds, 1):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_raw[tr_idx])
        X_te = sc.transform(X_raw[te_idx])
        for j in range(k):
            y = B_obs[tr_idx, j]
            if y.std() < 1e-10:
                b_hat[te_idx, j] = y.mean()
                continue
            model = LassoCV(
                alphas=np.logspace(-4, 1, 30), cv=5, max_iter=4000,
                random_state=seed, n_jobs=1,
            )
            model.fit(X_tr, y)
            b_hat[te_idx, j] = model.predict(X_te)
        print(f"      Fold {fold_i}/{K} done", flush=True)
    return b_hat


# =============================================================================
# LP-BASED CLP ESTIMATOR  (per-i scipy.linprog HiGHS)
#
# IMPORTANT note on the box bound nu in [-5, 5]:
#   The dual feasible set {nu : A'nu >= q} is generally unbounded in this
#   problem (because some columns of A have only outflow entries, giving rise
#   to recession directions).  For finite primal LP the optimum is achieved
#   at a vertex with bounded components, but nothing in CLP theory pins the
#   magnitude of those components.  The [-5, 5] box is a *defensive* limit;
#   if the true LP optimum has |nu_i| > 5 for some i, the LP returns a
#   suboptimal solution and the CLP estimator becomes biased
#   (anti-conservative direction).  For the 9 specs implemented here this
#   has not been observed empirically -- but for higher-dimensional A
#   matrices, widen the bounds (or set to None for unbounded) and verify
#   no observation hits the cap.  See `clp_estimate_diagnostics` below to
#   probe for binding nu components and LP fallback usage.
# =============================================================================
# Module-level diagnostic counters (reset by reset_lp_diagnostics()).
LP_DIAG = {'fallback_count': 0, 'binding_nu_count': 0, 'total_solves': 0}


def reset_lp_diagnostics():
    LP_DIAG['fallback_count'] = 0
    LP_DIAG['binding_nu_count'] = 0
    LP_DIAG['total_solves'] = 0


def _solve_lp(c, A_ub_neg, b_ub_neg, bounds, fallback, nu_abs_cap=None):
    """
    Solve one LP and update LP_DIAG counters.  Falls back to `fallback`
    if the solver fails.  If `nu_abs_cap` is given and the optimum has any
    |nu_i| within 1e-6 of the cap, increments `binding_nu_count`.
    """
    LP_DIAG['total_solves'] += 1
    res = linprog(c=c, A_ub=A_ub_neg, b_ub=b_ub_neg, bounds=bounds,
                  method='highs', options={'disp': False})
    if res.status != 0:
        LP_DIAG['fallback_count'] += 1
        return fallback
    if nu_abs_cap is not None and np.any(np.abs(res.x) > nu_abs_cap - 1e-6):
        LP_DIAG['binding_nu_count'] += 1
    return res.x


def clp_estimate(q, b_hat, B_obs, A_mat, hub_row=None, nu_lo=-5.0, nu_hi=5.0):
    """
    sigma_hat(q) = (1/N) sum_i nu_hat_i' B_i,
       where nu_hat_i = argmin_{A^T nu >= q, lo<=nu<=hi} nu' b_hat[i].

    Returns (sigma_hat, contribs).  Updates LP_DIAG counters as a side effect.
    """
    n, k = b_hat.shape
    A_ub_neg = -A_mat.T
    b_ub_neg = -np.asarray(q, dtype=float)
    bounds = [(nu_lo, nu_hi)] * k

    fallback = np.full(k, -1.0)
    if hub_row is not None and 0 <= hub_row < k:
        fallback[hub_row] = -2.0     # the 0p hub gets a stronger negative

    nu_cap = max(abs(nu_lo), abs(nu_hi)) if (nu_lo is not None and nu_hi is not None) else None

    contribs = np.zeros(n)
    for i in range(n):
        nu_i = _solve_lp(b_hat[i], A_ub_neg, b_ub_neg, bounds, fallback,
                         nu_abs_cap=nu_cap)
        contribs[i] = float(nu_i @ B_obs[i])
    return contribs.mean(), contribs


def report_lp_diagnostics(label=""):
    """Print LP_DIAG counters; warn if fallback or nu-cap binding rates are high."""
    n = LP_DIAG['total_solves']
    if n == 0:
        return
    fb = LP_DIAG['fallback_count']
    bn = LP_DIAG['binding_nu_count']
    print(f"      [LP diagnostics{(' '+label) if label else ''}]: "
          f"{n:,} solves, fallback={fb} ({100*fb/n:.2f}%), "
          f"|nu|>=cap={bn} ({100*bn/n:.2f}%)")
    if fb > 0:
        print(f"      [WARN] LP fallback was triggered {fb} times -- bounds may be biased.")
    if bn > 0:
        print(f"      [WARN] {bn} solves had |nu| at the cap -- consider widening nu bounds "
              f"(currently +-5).  Bounds may be anti-conservatively biased.")


# =============================================================================
# COMPOSITE Q VECTORS  (KT-style: TUW, TUWelf, Exit)
# =============================================================================
def composite_q_vectors(spec, p):
    """
    Build q_beta vectors for the three KT composites, generically using
    cell-overlap criteria.

      (A) Take-Up Work    : src cells all (0,*); dst cells all (>=1,*)
                            denominator = p00_c + p01_c
      (B) Take-Up Welfare : src cells all (*,0); dst cells all (*,1)
                            denominator = p00_c + p10_c + p20_c
      (C) Exit 0r         : src cells == 0r; dst cells all (*,0)
                            denominator = p01_c
    """
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
        if _is_0r(src) and _is_off_welfare(dst):
            q_Exit[c_idx] = 1.0

    q_TUW    /= (p00_c + p01_c)
    q_TUWelf /= (p00_c + p10_c + p20_c)
    q_Exit   /= p01_c

    return {
        "Take-Up Work (not working -> working)":      q_TUW,
        "Take-Up Welfare (off -> on welfare)":         q_TUWelf,
        "Exit 0r (on-welfare zero earn -> off welfare)": q_Exit,
    }


def compute_composite_bounds(b_hat, B_obs, A_mat, person_id, spec, p,
                              hub_row=None, label=""):
    qs = composite_q_vectors(spec, p)
    results = {}
    print(f"\n    --- COMPOSITE BOUNDS ({label}) ---")
    print(f"    {'composite':<55}  {'LB_pi':>8}  {'UB_pi':>8}  "
          f"{'width':>8}    {'95% CI':>22}")
    print("    " + "-" * 110)
    for name, q_beta in qs.items():
        ub_hat, c_up = clp_estimate( q_beta, b_hat, B_obs, A_mat, hub_row=hub_row)
        ci_ub = list(multiplier_bootstrap_ci(c_up, n_bs=N_BOOTSTRAP, person_id=person_id))
        neg_lb, c_dn = clp_estimate(-q_beta, b_hat, B_obs, A_mat, hub_row=hub_row)
        lb_hat = -neg_lb
        ci_lb = [-x for x in
                 multiplier_bootstrap_ci(c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]
        lb_c = max(0.0, min(1.0, lb_hat))
        ub_c = max(0.0, min(1.0, ub_hat))
        ci_lo = max(0.0, min(1.0, ci_lb[0]))
        ci_hi = max(0.0, min(1.0, ci_ub[1]))
        results[name] = dict(
            lb=lb_hat, ub=ub_hat, width=ub_hat-lb_hat,
            ci_lb=tuple(ci_lb), ci_ub=tuple(ci_ub),
            lb_clipped=lb_c, ub_clipped=ub_c, ci_outer=(ci_lo, ci_hi),
        )
        print(f"    {name:<55}  {lb_c:8.4f}  {ub_c:8.4f}  "
              f"{ub_c-lb_c:8.4f}    [{ci_lo:.3f}, {ci_hi:.3f}]")
    return results


# =============================================================================
# TARGET PARAMETERS  (always reported per spec, in addition to composites)
# =============================================================================
# We always report bounds for these three parameter "families":
#   pi(1n, 1r), pi(2n, 1r), pi(0r, 2u)
# In specs where the source or destination is split into sub-bins, EACH
# matching beta column is reported individually.  Examples:
#   - Spec 1's pi(2n, 1r) -> 3 betas: beta(b6n,1r), beta(b7n,1r), beta(b8n,1r)
#   - Spec 4's pi(2n, 1r) -> 15 betas: beta(b_kn, b_ir) for k=6..8, i=1..5
#   - Spec 9's pi(2n, 1r) -> 5 betas: beta(2n, b_ir) for i=1..5
TARGET_PARAMETERS: List[Tuple[str, str, str]] = [
    ("pi(1n, 1r)", "1n", "1r"),
    ("pi(2n, 1r)", "2n", "1r"),
    ("pi(0r, 2u)", "0r", "2u"),
]


def find_target_cols(spec, target_src_label: str, target_dst_label: str):
    """
    Return list of column indices in `spec` whose (source, destination) labels
    are SUBSETS of the target labels' covered cells.  Subset-checking handles
    granular cells like "b6n" matching pooled "2n", and pooled "1n" matching
    pooled "1n", etc.
    """
    target_src_cells = STATE_DEF[target_src_label]
    target_dst_cells = STATE_DEF[target_dst_label]
    matching = []
    for c_idx, (src, dst) in enumerate(spec["cols"]):
        src_cells = STATE_DEF[src]
        dst_cells = STATE_DEF[dst]
        if src_cells <= target_src_cells and dst_cells <= target_dst_cells:
            matching.append(c_idx)
    return matching


def _bounds_for_col(j, spec, b_hat, B_obs, A_mat, person_id, hub_row=None):
    """Compute LB, UB, and bootstrap 95% CIs for a single beta column j."""
    n_cols = len(spec["cols"])
    q_up = np.zeros(n_cols); q_up[j] =  1.0
    q_dn = np.zeros(n_cols); q_dn[j] = -1.0
    ub_hat, c_up = clp_estimate(q_up, b_hat, B_obs, A_mat, hub_row=hub_row)
    ci_ub = list(multiplier_bootstrap_ci(
        c_up, n_bs=N_BOOTSTRAP, person_id=person_id))
    neg_lb, c_dn = clp_estimate(q_dn, b_hat, B_obs, A_mat, hub_row=hub_row)
    lb_hat = -neg_lb
    ci_lb = [-x for x in
             multiplier_bootstrap_ci(
                 c_dn, n_bs=N_BOOTSTRAP, person_id=person_id)[::-1]]
    return dict(
        lb=lb_hat, ub=ub_hat, width=ub_hat - lb_hat,
        ci_lb=tuple(ci_lb), ci_ub=tuple(ci_ub),
    )


def report_target_parameters(b_hat, B_obs, A_mat, person_id, spec, src_probs,
                              hub_row=None, label=""):
    """
    For each target parameter (pi(1n,1r), pi(2n,1r), pi(0r,2u)):
      - Find all matching beta columns in this spec.
      - Compute individual bounds for each.
      - Print and return as a nested dict.
    """
    target_results = {}
    print(f"\n    --- TARGET PARAMETER BOUNDS ({label}) ---")
    print(f"    For each parameter, all matching beta columns are listed,")
    print(f"    in BOTH beta-units (joint probabilities) AND pi-units")
    print(f"    (transition probabilities = beta / P^a(source)).")

    for tp_name, src_label, dst_label in TARGET_PARAMETERS:
        cols = find_target_cols(spec, src_label, dst_label)
        n = len(cols)
        suffix = f"{n} matching beta column{'s' if n != 1 else ''}"
        print(f"\n    {tp_name}: {suffix}")
        if n == 0:
            print(f"      (no columns in this spec match this parameter family)")
            target_results[tp_name] = {}
            continue

        # ---- First pass: compute everything ----
        param_results = {}
        rows_to_print = []
        for j in cols:
            r = _bounds_for_col(
                j, spec, b_hat, B_obs, A_mat, person_id, hub_row=hub_row,
            )
            sp = src_probs[j]
            (s_lab, d_lab) = spec["cols"][j]
            beta_name = f"beta({s_lab},{d_lab})"
            pi_name   = f"pi({s_lab},{d_lab})"

            # pi-unit point estimates and CIs (clipped to [0, 1])
            if sp > 1e-12:
                lb_pi    = max(0., min(1., r['lb'] / sp))
                ub_pi    = max(0., min(1., r['ub'] / sp))
                ci_lb_pi = (max(0., min(1., r['ci_lb'][0] / sp)),
                            max(0., min(1., r['ci_lb'][1] / sp)))
                ci_ub_pi = (max(0., min(1., r['ci_ub'][0] / sp)),
                            max(0., min(1., r['ci_ub'][1] / sp)))
            else:
                lb_pi = ub_pi = float('nan')
                ci_lb_pi = ci_ub_pi = (float('nan'), float('nan'))

            r['source_pa'] = sp
            r['lb_pi']     = lb_pi
            r['ub_pi']     = ub_pi
            r['ci_lb_pi']  = ci_lb_pi
            r['ci_ub_pi']  = ci_ub_pi
            r['pi_name']   = pi_name
            param_results[beta_name] = r
            rows_to_print.append((j, beta_name, pi_name, sp, r))

        # ---- BETA-units block ----
        print(f"      ---- bounds in beta-units (joint probability) ----")
        print(f"      {'col':>3}  {'beta':<26}  {'LB_beta':>9}  {'UB_beta':>9}  "
              f"{'width':>8}    {'95% CI(LB_beta)':>18}    {'95% CI(UB_beta)':>18}")
        for j, beta_name, pi_name, sp, r in rows_to_print:
            print(f"      {j:>3}  {beta_name:<26}  "
                  f"{r['lb']:+9.4f}  {r['ub']:+9.4f}  {r['width']:8.4f}    "
                  f"[{r['ci_lb'][0]:+.3f},{r['ci_lb'][1]:+.3f}]    "
                  f"[{r['ci_ub'][0]:+.3f},{r['ci_ub'][1]:+.3f}]")

        # ---- PI-units block ----
        print(f"      ---- bounds in pi-units  "
              f"(transition probability = beta / P^a(source); clipped to [0,1]) ----")
        print(f"      {'col':>3}  {'pi':<26}  {'P^a':>7}  {'LB_pi':>8}  {'UB_pi':>8}  "
              f"{'width':>8}    {'95% CI(LB_pi)':>16}    {'95% CI(UB_pi)':>16}")
        for j, beta_name, pi_name, sp, r in rows_to_print:
            print(f"      {j:>3}  {pi_name:<26}  {sp:7.4f}  "
                  f"{r['lb_pi']:8.4f}  {r['ub_pi']:8.4f}  "
                  f"{r['ub_pi'] - r['lb_pi']:8.4f}    "
                  f"[{r['ci_lb_pi'][0]:.3f},{r['ci_lb_pi'][1]:.3f}]    "
                  f"[{r['ci_ub_pi'][0]:.3f},{r['ci_ub_pi'][1]:.3f}]")

        target_results[tp_name] = param_results

    return target_results


# =============================================================================
# RUN ONE SPEC
# =============================================================================
def run_spec(spec_key, data, p, run_individual_betas=False):
    """
    Full CLP pipeline for one spec.
      data: dict from prepare_jf_data_granular -> (D, ebin, partic, ...).
      p:    dict from load_table4_mat (coarse marginals).
      run_individual_betas: if True, also bound every individual beta column.
    """
    spec = SPECS[spec_key]
    print(f"\n{'='*88}")
    print(f"  Running {spec_key}")
    print(f"  {spec['name']}")
    print(f"{'='*88}")

    D, ebin_arr, partic_arr, _state_row_old, df_incl, person_id, pscorewt = data

    # --- Build A and verify
    A_mat = build_A(spec)
    verify_A(A_mat, spec)

    # --- B-vector and source probs
    B_obs = compute_B(spec, D, ebin_arr, partic_arr, pscorewt)
    src_probs = compute_source_probs(spec, D, ebin_arr, partic_arr, pscorewt)
    print(f"    B_obs: {B_obs.shape}, sample mean B (= b0): "
          f"{B_obs.mean(axis=0).round(4)[:8]}{'...' if B_obs.shape[1] > 8 else ''}")

    # --- First-stage LASSO (GroupKFold by person)
    avail = [v for v in COV_VARS if v in df_incl.columns]
    X_raw = df_incl[avail].fillna(0).to_numpy(float)
    print(f"    Cross-fit LASSO with GroupKFold (K={K_FOLDS})")
    b_hat = estimate_b0_lasso(B_obs, X_raw, person_id, K=K_FOLDS, seed=RANDOM_SEED)
    print(f"    Max |E[B] - E[b_hat]|: "
          f"{np.max(np.abs(B_obs.mean(axis=0) - b_hat.mean(axis=0))):.2e}")

    # --- Identify hub row (0p) for LP fallback
    hub_row = None
    for r_idx, r_label in enumerate(spec["rows"]):
        if r_label in ("0p", "0r"):
            hub_row = r_idx
            break

    # --- Reset LP diagnostics for this spec
    reset_lp_diagnostics()

    # --- Target parameter bounds (ALWAYS reported first per user request)
    target_results = report_target_parameters(
        b_hat, B_obs, A_mat, person_id, spec, src_probs,
        hub_row=hub_row, label=spec_key,
    )

    # --- Composite bounds (KT Take-Up Work / Take-Up Welfare / Exit 0r)
    composite_results = compute_composite_bounds(
        b_hat, B_obs, A_mat, person_id, spec, p, hub_row=hub_row, label=spec_key,
    )

    # --- Report cumulative LP diagnostics for this spec
    report_lp_diagnostics(label=spec_key)

    # --- Individual beta bounds for ALL columns (optional, off by default)
    individual_results = {}
    if run_individual_betas:
        cols = spec["cols"]
        n_beta = len(cols)
        print(f"\n    Individual beta bounds for ALL {n_beta} cols:")
        print(f"    {'col':>3}  {'beta':<22}  {'LB':>9}  {'UB':>9}  "
              f"{'width':>8}  {'P^a':>7}  {'LB_pi':>8}  {'UB_pi':>8}")
        for j in range(n_beta):
            r = _bounds_for_col(
                j, spec, b_hat, B_obs, A_mat, person_id, hub_row=hub_row,
            )
            sp = src_probs[j]
            lb_pi = (max(0., min(1., r['lb'] / sp))
                     if sp > 1e-12 else float('nan'))
            ub_pi = (max(0., min(1., r['ub'] / sp))
                     if sp > 1e-12 else float('nan'))
            beta_name = f"beta({cols[j][0]},{cols[j][1]})"
            r['source_pa'] = sp
            r['lb_pi']     = lb_pi
            r['ub_pi']     = ub_pi
            individual_results[beta_name] = r
            print(f"    {j:>3}  {beta_name:<22}  {r['lb']:+9.4f}  {r['ub']:+9.4f}  "
                  f"{r['width']:8.4f}  {sp:7.4f}  {lb_pi:8.4f}  {ub_pi:8.4f}")

    return dict(
        spec_key=spec_key,
        spec=spec,
        A_shape=A_mat.shape,
        rank=np.linalg.matrix_rank(A_mat),
        target=target_results,
        composite=composite_results,
        individual=individual_results,
    )


# =============================================================================
# MAIN
# =============================================================================
SPECS_TO_RUN = ["spec1", "spec2", "spec3", "spec4", "spec5",
                "spec6", "spec7", "spec8", "spec9"]
# Target parameters (pi(1n,1r), pi(2n,1r), pi(0r,2u)) are ALWAYS reported per spec.
# This flag controls whether ALL OTHER beta columns are also bounded
# (slow for the larger specs).
RUN_INDIVIDUAL_BETAS = False


def main():
    t0 = time.time()
    print("=" * 88)
    print("CLP_granular_correct -- multi-specification CLP file")
    print("=" * 88)
    print(f"  Specs to run: {SPECS_TO_RUN}")
    print(f"  Individual beta bounds: {RUN_INDIVIDUAL_BETAS}")

    # Load data once (expensive: pscore logit)
    print(f"\n[STAGE 1] Load data via prepare_jf_data_granular()")
    data = prepare_jf_data_granular()
    p, _ = load_table4_mat()

    # Quick spec verification first (without running anything)
    print(f"\n[STAGE 2] Verify all specs build cleanly")
    for spec_key in SPECS_TO_RUN:
        spec = SPECS[spec_key]
        A = build_A(spec)
        verify_A(A, spec)

    # Run each spec
    print(f"\n[STAGE 3] Run all specs sequentially")
    all_results = {}
    for spec_key in SPECS_TO_RUN:
        all_results[spec_key] = run_spec(
            spec_key, data, p, run_individual_betas=RUN_INDIVIDUAL_BETAS,
        )

    # Cross-spec summary
    print(f"\n\n{'='*88}")
    print("SUMMARY: Composite bounds across specifications")
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
        tuw    = fmt("Take-Up Work (not working -> working)")
        tuwf   = fmt("Take-Up Welfare (off -> on welfare)")
        exit_  = fmt("Exit 0r (on-welfare zero earn -> off welfare)")
        print(f"{spec_key:<8}  {sh:<10}  {tuw:<28}  {tuwf:<28}  {exit_:<28}")

    # Cross-spec target-parameter summary
    # Each entry shows BOTH beta-units and pi-units bounds for a target parameter.
    print(f"\n\n{'='*88}")
    print("SUMMARY: Target parameters across specifications")
    print("Each entry: 'beta(s,d) = [LB_beta, UB_beta]   pi(s,d) = [LB_pi, UB_pi]'")
    print("=" * 88)
    for tp_name, _, _ in TARGET_PARAMETERS:
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
                pi_name = d.get('pi_name', beta_name.replace('beta(', 'pi('))
                lb_b = d['lb']
                ub_b = d['ub']
                lb_pi = d.get('lb_pi', float('nan'))
                ub_pi = d.get('ub_pi', float('nan'))
                print(f"      {beta_name:<24}=[{lb_b:+.4f},{ub_b:+.4f}]    "
                      f"{pi_name:<24}=[{lb_pi:.4f},{ub_pi:.4f}]")

    # Save BOTH composite and target-parameter results to CSV
    composite_path = ("/Users/gevorgkhandamiryan/Desktop/cursorclp/check/"
                      "granular_correct_multispec_composite.csv")
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

    target_path = ("/Users/gevorgkhandamiryan/Desktop/cursorclp/check/"
                   "granular_correct_multispec_target.csv")
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
                    pi=d.get('pi_name',
                             beta_name.replace('beta(', 'pi(')),
                    source_pa=d.get('source_pa', float('nan')),
                    # Beta-units bounds and CIs
                    lb_beta=d['lb'], ub_beta=d['ub'], width_beta=d['width'],
                    ci_lb_beta_lo=d['ci_lb'][0], ci_lb_beta_hi=d['ci_lb'][1],
                    ci_ub_beta_lo=d['ci_ub'][0], ci_ub_beta_hi=d['ci_ub'][1],
                    # Pi-units bounds and CIs (clipped to [0,1])
                    lb_pi=d.get('lb_pi', float('nan')),
                    ub_pi=d.get('ub_pi', float('nan')),
                    width_pi=(d.get('ub_pi', float('nan'))
                              - d.get('lb_pi', float('nan'))),
                    ci_lb_pi_lo=ci_lb_pi[0], ci_lb_pi_hi=ci_lb_pi[1],
                    ci_ub_pi_lo=ci_ub_pi[0], ci_ub_pi_hi=ci_ub_pi[1],
                ))
    pd.DataFrame(rows).to_csv(target_path, index=False)
    print(f"  Wrote target-parameter results to {target_path}")
    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
