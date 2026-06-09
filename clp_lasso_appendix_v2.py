"""
clp_lasso_appendix_v2.py
========================
BUGFIX-ONLY drop-in replacement for `clp_lasso_appendix.py`.

WHAT IS DIFFERENT FROM v1
-------------------------
1. `kt_composite_q_vectors` now uses the KT-faithful q vectors for the
   TUW (Take-Up Work) and Exit-0r composites.  v1 was missing cells:

       v1 q_tuw   = [1, 0, 0, 1, 0, 1, 0, 1, 0]        ← positions {0,3,5,7}
       v2 q_tuw   = [1, 0, 0, 1, 1, 1, 0, 1, 0]        ← positions {0,3,4,5,7}
                              ^----- β(0r, 1r) restored (KT Bounds.m line 533)

       v1 q_exit  = [0, 1, 0, 0, 0, 0, 0, 0, 0]        ← positions {1}
       v2 q_exit  = [0, 1, 0, 1, 0, 1, 0, 0, 0]        ← positions {1,3,5}
                           ^-----^-- β(0r,2n) and β(0r,1n) restored
                                     (KT Bounds.m line 622, "on welfare → off welfare")

   q_tuwelf was correct in v1 and is unchanged in v2.

2. Output filenames switched to `_v2` so v1 and v2 do not overwrite
   each other.  All other behaviour (data prep, first-stage estimator
   configurations, LP route via vertex enumeration, diagnostics block)
   is identical to v1.

Quick spot-checks (run the file and look at the printed composite table)
-----------------------------------------------------------------------
* TUW composite UB should now be POINT-IDENTIFIED for every config —
  because q_tuw = A^T (-1, 0, 0, -1, 0) lies in row(A) (verified
  numerically earlier), so T_q has a single vertex.  Both LB and UB
  collapse to ≈ 0.160, no more "ub=0.159805 for every estimator while
  lb varies".  The previous (buggy) q_tuw gave a 16-vertex T_{-q}
  which produced the misleading wide LB range.
* Exit 0r bounds become MEANINGFULLY DIFFERENT.  v1 was secretly
  bounding β(0r,0n)/p_{01,c} (one of three transitions), not the
  composite KT means.  v2 reports the correct three-transition sum.

References
----------
* `Bounds.m` lines 527-555 (TUW) and 619-660 (Exit 0r) in the KT
  replication package: AER_Code/Bounds.m.
* Earlier discussion in this session traced the same bug pattern in
  `clp_ols.py` and `clp_lasso_appendix.py`; this file is the patched
  drop-in for the latter.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold

warnings.filterwarnings("ignore")

# Reuse EVERYTHING from clp_lasso_appendix that does not need patching.
# We only redefine `kt_composite_q_vectors` (with the bug fix) and
# override the output paths.  The legacy module's main() reads
# `kt_composite_q_vectors` from its own namespace at call time, so we
# monkey-patch it onto the legacy module before invoking main().
from CLP_final import (
    build_A, enumerate_dual_vertices, clp_estimate, multiplier_bootstrap_ci,
    S_0N, S_1N, S_2N, S_0P, S_2P,
)
import clp_lasso_appendix as legacy


# =============================================================================
# Corrected composite q-vectors  (KT Bounds.m faithful)
# =============================================================================
def kt_composite_q_vectors_v2(p_kt):
    """Build the three KT composite q vectors in CLP β-ordering, with the
    TUW and Exit-0r positions matching KT Bounds.m.

    CLP β-ordering:
      [β(0n,1r), β(0r,0n), β(2n,1r), β(0r,2n), β(0r,1r),
       β(0r,1n), β(1n,1r), β(0r,2u), β(2u,1r)]
    """
    p00_c = p_kt['p00_c']; p01_c = p_kt['p01_c']
    p10_c = p_kt['p10_c']; p20_c = p_kt['p20_c']

    # TUW (Take-Up Work) -- non-zero at positions {0, 3, 4, 5, 7}
    #   = β(0n,1r) + β(0r,2n) + β(0r,1r) + β(0r,1n) + β(0r,2u)
    q_tuw = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0], dtype=float)
    q_tuw /= max(1e-12, p00_c + p01_c)

    # TUWelf (Take-Up Welfare) -- non-zero at {0, 2, 6} (unchanged from v1)
    #   = β(0n,1r) + β(2n,1r) + β(1n,1r)
    q_tuwelf = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0], dtype=float)
    q_tuwelf /= max(1e-12, p00_c + p10_c + p20_c)

    # Exit 0r (on welfare 0r → off welfare anywhere) -- {1, 3, 5}
    #   = β(0r,0n) + β(0r,2n) + β(0r,1n)
    q_exit = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=float)
    q_exit /= max(1e-12, p01_c)

    return {
        "TUW (Take-Up Work)":                  q_tuw,
        "TUWelf (Take-Up Welfare)":            q_tuwelf,
        "Exit 0r (welfare-zero -> off welfare)": q_exit,
    }


# =============================================================================
# Output paths (separate from v1 so files do not collide)
# =============================================================================
OUT_DIR        = os.path.dirname(os.path.abspath(__file__))
OUT_CSV_V2     = os.path.join(OUT_DIR, "clp_lasso_appendix_v2_results.csv")
OUT_DIAG_V2    = os.path.join(OUT_DIR, "clp_lasso_appendix_v2_diagnostics.csv")


# =============================================================================
# MAIN — monkey-patch the legacy module and run its main()
# =============================================================================
def main():
    print("=" * 96)
    print("  clp_lasso_appendix_v2.py  --  q-vector bugfix (TUW position 4,")
    print("                                Exit-0r positions 3 and 5 restored)")
    print("=" * 96)

    # Patch the legacy module:
    #   - kt_composite_q_vectors  : corrected q-vectors
    #   - output paths            : v2 filenames
    legacy.kt_composite_q_vectors = kt_composite_q_vectors_v2
    legacy.OUT_CSV                = OUT_CSV_V2
    legacy.OUT_DIAG_CSV           = OUT_DIAG_V2

    # Run the legacy pipeline with patched q-vectors.
    legacy.main()


if __name__ == "__main__":
    main()
