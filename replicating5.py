import numpy as np
from scipy.optimize import linprog

# ─────────────────────────────────────────────────────────────────────────────
# ▶  SET THIS PATH to your local copy of the KT replication data file
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "//Users/gevorgkhandamiryan/Desktop/cursorclp/AER_Code/DerivedData/Table4_mat_python.txt"


# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
print(f"Loading:  {DATA_PATH}")
raw   =  np.loadtxt(DATA_PATH, skiprows=1, delimiter="\t")
fullps = raw[0, :]       # row 0 = point estimates
bsdata = raw[1:, :]      # rows 1+ = bootstrap draws
N_bs   = bsdata.shape[0]

# ── Point-estimate state probabilities (using 0-based column indices) ──────
# Notation: p{earnings}{welfare}_{group}
#   earnings  0=zero, 1=below FPL, 2=above FPL
#   welfare   0=off(n), 1=on(p)
#   _t = JF (treatment),  _c = AFDC (control)
p10_c = fullps[3];   p10_t = fullps[2]
p20_c = fullps[5];   p20_t = fullps[4]
p01_c = fullps[7];   p01_t = fullps[6]
p11_c = fullps[9];   p11_t = fullps[8]
p21_c = fullps[11];  p21_t = fullps[10]

p00_c = 1 - p10_c - p20_c - p01_c - p11_c - p21_c
p00_t = 1 - p10_t - p20_t - p01_t - p11_t - p21_t

print("\n── Reweighted state probabilities (Table 4 check) ─────────────────")
for lbl, t, c in zip(["0n","1n","2n","0p","1p","2p"],
                     [p00_t,p10_t,p20_t,p01_t,p11_t,p21_t],
                     [p00_c,p10_c,p20_c,p01_c,p11_c,p21_c]):
    print(f"  {lbl}:  JF = {t:.4f}   AFDC = {c:.4f}")

# ── Bootstrap draws ────────────────────────────────────────────────────────
b10_c = bsdata[:,3];  b10_t = bsdata[:,2]
b20_c = bsdata[:,5];  b20_t = bsdata[:,4]
b01_c = bsdata[:,7];  b01_t = bsdata[:,6]
b11_c = bsdata[:,9];  b11_t = bsdata[:,8]
b21_c = bsdata[:,11]; b21_t = bsdata[:,10]

b00_c = 1 - b10_c - b20_c - b01_c - b11_c - b21_c
b00_t = 1 - b10_t - b20_t - b01_t - b11_t - b21_t


# ═════════════════════════════════════════════════════════════════════════════
# 2.  THE LINEAR SYSTEM   Aπ = b   (equation (11) in the paper)
# ═════════════════════════════════════════════════════════════════════════════
#
# π ordering (9 response probabilities):
#   index 0:  π_{0r→0n}   on welfare not working → off welfare not working
#   index 1:  π_{0n→1r}   off welfare not working → working range-1, on welfare
#   index 2:  π_{0r→2n}   on welfare not working → earning range-2, off welfare
#   index 3:  π_{2n→1r}   high earner off welfare → range-1 earner on welfare  (OPT-IN)
#   index 4:  π_{0r→1r}   on welfare not working → range-1 earner on welfare
#   index 5:  π_{0r→2u}   on welfare not working → underreporter in range-2
#   index 6:  π_{2u→1r}   underreporter range-2 → truthful range-1 reporter
#   index 7:  π_{1n→1r}   range-1 earner off welfare → range-1 earner on welfare
#   index 8:  π_{0r→1n}   on welfare not working → range-1 earner off welfare

# RHS vector  b  (observed JF−AFDC differences in state shares)
b_vec = np.array([
    p00_t - p00_c,                   # row 1: change in fraction 0n
    p10_t - p10_c,                   # row 2: change in fraction 1n
    p20_t - p20_c,                   # row 3: change in fraction 2n
    (p01_c - p01_t) / p01_c,         # row 4: fractional exit from welfare (0p)
    p21_t - p21_c,                   # row 5: change in fraction 2p
])

# Coefficient matrix  A  (5 × 9)
A_mat = np.array([
    [p01_c, -p00_c, 0,      0,      0,      0,      0,      0,      0    ],
    [0,      0,     0,      0,      0,      0,      0,     -p10_c,  p01_c],
    [0,      0,     p01_c, -p20_c,  0,      0,      0,      0,      0    ],
    [1,      0,     1,      0,      1,      1,      0,      0,      1    ],
    [0,      0,     0,      0,      0,      p01_c, -p21_c,  0,      0    ],
])

BOUNDS  = [(0, 1)] * 9        # 0 ≤ π_i ≤ 1
LP_OPTS = dict(method='highs', options={'presolve': True})

PI_NAMES = [
    "π_{0r→0n}", "π_{0n→1r}", "π_{0r→2n}", "π_{2n→1r}", "π_{0r→1r}",
    "π_{0r→2u}", "π_{2u→1r}", "π_{1n→1r}", "π_{0r→1n}",
]

# ═════════════════════════════════════════════════════════════════════════════
# 3.  LP POINT ESTIMATES
# ═════════════════════════════════════════════════════════════════════════════
def lp_bounds(A, b, bounds):
    n = A.shape[1]
    lb = np.full(n, np.nan);  ub = np.full(n, np.nan)
    for i in range(n):
        c = np.zeros(n); c[i] = 1
        r = linprog(c, A_eq=A, b_eq=b, bounds=bounds, **LP_OPTS)
        if r.status == 0: lb[i] = r.x[i]
        c[i] = -1
        r = linprog(c, A_eq=A, b_eq=b, bounds=bounds, **LP_OPTS)
        if r.status == 0: ub[i] = r.x[i]
    return lb, ub


LB_lp, UB_lp = lp_bounds(A_mat, b_vec, BOUNDS)

print("\n── TABLE 5 panel (a)  ─  LP point estimates ────────────────────────")
print(f"  {'Parameter':<14}  {'LP Lower':>10}  {'LP Upper':>10}")
for nm, lo, hi in zip(PI_NAMES, LB_lp, UB_lp):
    print(f"  {nm:<14}  {lo:10.4f}  {hi:10.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ANALYTICAL BOUNDS (closed-form, Online Appendix)
# ═════════════════════════════════════════════════════════════════════════════
def analytical_bounds(p00c, p10c, p20c, p01c, p11c, p21c,
                      p00t, p10t, p20t, p01t, p11t, p21t):
    ub = np.array([
        (p01c - p01t - (p21t - p21c)) / p01c,
        (p00c - p00t + p01c - p01t - (p21t - p21c)) / p00c,
        p20t / p01c,
        1.0,
        (p01c - p01t - (p21t - p21c)) / p01c,
        p21t / p01c,
        1.0,
        (p10c - p10t + p01c - p01t + p21c - p21t) / p10c,
        (p01c - p01t + p21c - p21t) / p01c,
    ])
    lb = np.array([
        0.0,
        (p00c - p00t) / p00c,
        0.0,
        (p20c - p20t) / p20c,
        0.0,
        (p21t - p21c) / p01c,
        0.0,
        (p10c - p10t) / p10c,
        0.0,
    ])
    return lb, ub

LB_an, UB_an = analytical_bounds(p00_c, p10_c, p20_c, p01_c, p11_c, p21_c,
                                  p00_t, p10_t, p20_t, p01_t, p11_t, p21_t)

# Verify LP == analytical
tol = 1e-4
match = (np.allclose(LB_lp, LB_an, atol=tol, equal_nan=True) and
         np.allclose(UB_lp, UB_an, atol=tol, equal_nan=True))
print("\n── Analytical vs LP check ─────────────────────────────────────────")
if match:
    print("  ✓ Analytical bounds agree with LP bounds (tol = 1e-4).")
else:
    print("  ✗ MISMATCH — check data consistency:")
    for i in range(9):
        if not (np.isclose(LB_lp[i], LB_an[i], atol=tol) and
                np.isclose(UB_lp[i], UB_an[i], atol=tol)):
            print(f"    {PI_NAMES[i]}: LP=[{LB_lp[i]:.4f},{UB_lp[i]:.4f}]  "
                  f"Analytical=[{LB_an[i]:.4f},{UB_an[i]:.4f}]")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  BOOTSTRAP BOUNDS
# ═════════════════════════════════════════════════════════════════════════════
def bs_analytical(b00c, b10c, b20c, b01c, b11c, b21c,
                  b00t, b10t, b20t, b01t, b11t, b21t):
    BUB = np.column_stack([
        (b01c - b01t - (b21t - b21c)) / b01c,
        (b00c - b00t + b01c - b01t - (b21t - b21c)) / b00c,
        b20t / b01c,
        np.ones(len(b01c)),
        (b01c - b01t - (b21t - b21c)) / b01c,
        b21t / b01c,
        np.ones(len(b01c)),
        (b10c - b10t + b01c - b01t + b21c - b21t) / b10c,
        (b01c - b01t + b21c - b21t) / b01c,
    ])
    BLB = np.column_stack([
        np.zeros(len(b01c)),
        (b00c - b00t) / b00c,
        np.zeros(len(b01c)),
        (b20c - b20t) / b20c,
        np.zeros(len(b01c)),
        (b21t - b21c) / b01c,
        np.zeros(len(b01c)),
        (b10c - b10t) / b10c,
        np.zeros(len(b01c)),
    ])
    return BLB, BUB

BLB, BUB = bs_analytical(b00_c, b10_c, b20_c, b01_c, b11_c, b21_c,
                          b00_t, b10_t, b20_t, b01_t, b11_t, b21_t)

# ── Naïve CI (Imbens–Manski 2002: extend by 1.65 × bootstrap std dev) ─────
L_naive = np.maximum(0, LB_an - 1.65 * BLB.std(axis=0))
U_naive = np.minimum(1, UB_an + 1.65 * BUB.std(axis=0))

print("\n── Naïve 90% CIs (Imbens–Manski) ──────────────────────────────────")
print(f"  {'Parameter':<14}  {'Lower':>10}  {'Upper':>10}")
for nm, lo, hi in zip(PI_NAMES, L_naive, U_naive):
    print(f"  {nm:<14}  {lo:10.4f}  {hi:10.4f}")



# ═════════════════════════════════════════════════════════════════════════════
# 6.  CONSERVATIVE CIs  (Chernozhukov–Lee–Rosen style)
# ═════════════════════════════════════════════════════════════════════════════
# For each bound: enumerate all possible binding expressions (candidate
# constraint vertices), compute bootstrap deviation T = BUBⱼ − PUBⱼ,
# take 95th percentile of max(−T) across candidate vertices, add to PE.

def conservative_upper(pe_binding, bs_cands, pe_cands):
    """
    pe_binding  : scalar    — point estimate of the (minimum) upper bound
    bs_cands    : (N, K)    — bootstrap draws for K candidate expressions
    pe_cands    : (K,)      — point estimates of K candidates
    """
    T    = bs_cands - pe_cands[np.newaxis, :]
    maxT = (-T).max(axis=1)
    r    = np.quantile(maxT, 0.95)
    return pe_binding + r


# ── π_0  (0r→0n) ────────────────────────────────────────────────────────
BUB0 = np.column_stack([
    (b01_c - b01_t + b21_c - b21_t) / b01_c,
    b00_t / b01_c,
    (b01_c - b01_t) / b01_c,
    (b01_c - b01_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b20_c - b20_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t + b20_c - b20_t) / b01_c,
])
PUB0 = np.array([
    (p01_c - p01_t + p21_c - p21_t) / p01_c,
    p00_t / p01_c,
    (p01_c - p01_t) / p01_c,
    (p01_c - p01_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p20_c - p20_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t + p20_c - p20_t) / p01_c,
])
UL0 = conservative_upper(UB_an[0], BUB0, PUB0)

# ── π_1  (0n→1r) ────────────────────────────────────────────────────────
BUB1 = np.column_stack([
    (b00_c - b00_t + b01_c - b01_t - (b21_t - b21_c)) / b00_c,
    (b00_c - b00_t + b01_c - b01_t) / b00_c,
    (b00_c - b00_t + b01_c - b01_t + b20_c - b20_t) / b00_c,
    (b00_c - b00_t + b01_c - b01_t + b21_c - b21_t + b20_c - b20_t) / b00_c,
    (b01_c - b01_t + b00_c - b00_t + b10_c - b10_t) / b00_c,
    (b01_c - b01_t + b00_c - b00_t + b10_c - b10_t + b20_c - b20_t) / b00_c,
    (b01_c - b01_t + b00_c - b00_t + b10_c - b10_t + b21_c - b21_t) / b00_c,
    (b01_c - b01_t + b00_c - b00_t + b10_c - b10_t + b20_c - b20_t + b21_c - b21_t) / b00_c,
])
PUB1 = np.array([
    (p00_c - p00_t + p01_c - p01_t - (p21_t - p21_c)) / p00_c,
    (p00_c - p00_t + p01_c - p01_t) / p00_c,
    (p00_c - p00_t + p01_c - p01_t + p20_c - p20_t) / p00_c,
    (p00_c - p00_t + p01_c - p01_t + p21_c - p21_t + p20_c - p20_t) / p00_c,
    (p01_c - p01_t + p00_c - p00_t + p10_c - p10_t) / p00_c,
    (p01_c - p01_t + p00_c - p00_t + p10_c - p10_t + p20_c - p20_t) / p00_c,
    (p01_c - p01_t + p00_c - p00_t + p10_c - p10_t + p21_c - p21_t) / p00_c,
    (p01_c - p01_t + p00_c - p00_t + p10_c - p10_t + p20_c - p20_t + p21_c - p21_t) / p00_c,
])
UL1 = conservative_upper(UB_an[1], BUB1, PUB1)

# ── π_2  (0r→2n) ────────────────────────────────────────────────────────
BUB2 = np.column_stack([
    b20_t / b01_c,
    (b01_c - b01_t) / b01_c,
    (b01_c - b01_t + b21_c - b21_t) / b01_c,
    2 * (b01_c - b01_t) / b01_c,
    (2*(b01_c - b01_t) + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t + b00_c - b00_t) / b01_c,
])
PUB2 = np.array([
    p20_t / p01_c,
    (p01_c - p01_t) / p01_c,
    (p01_c - p01_t + p21_c - p21_t) / p01_c,
    2*(p01_c - p01_t) / p01_c,
    (2*(p01_c - p01_t) + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t + p00_c - p00_t) / p01_c,
])
UL2 = conservative_upper(UB_an[2], BUB2, PUB2)

# ── π_4  (0r→1r) — has 15 candidate expressions ─────────────────────────
BUB4 = np.column_stack([
    (b01_c - b01_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b21_c - b21_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t + b20_c - b20_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b21_c - b21_t + b20_c - b20_t + b00_c - b00_t) / b01_c,
])
PUB4 = np.array([
    (p01_c - p01_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p21_c - p21_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t + p20_c - p20_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p21_c - p21_t + p20_c - p20_t + p00_c - p00_t) / p01_c,
])
UL4 = conservative_upper(UB_an[4], BUB4, PUB4)

# ── π_5  (0r→2u) ────────────────────────────────────────────────────────
BUB5 = np.column_stack([
    b21_t / b01_c,
    (b01_c - b01_t) / b01_c,
    (b01_c - b01_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b20_c - b20_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b10_c - b10_t + b20_c - b20_t + b00_c - b00_t) / b01_c,
])
PUB5 = np.array([
    p21_t / p01_c,
    (p01_c - p01_t) / p01_c,
    (p01_c - p01_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p20_c - p20_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p10_c - p10_t + p20_c - p20_t + p00_c - p00_t) / p01_c,
])
UL5 = conservative_upper(UB_an[5], BUB5, PUB5)

# ── π_7  (1n→1r) ────────────────────────────────────────────────────────
BUB7 = np.column_stack([
    (b10_c - b10_t + b01_c - b01_t + b21_c - b21_t) / b10_c,
    (b10_c - b10_t + b01_c - b01_t) / b10_c,
    (b10_c - b10_t + b01_c - b01_t + b00_c - b00_t) / b10_c,
    (b10_c - b10_t + b01_c - b01_t + b20_c - b20_t) / b10_c,
    (b10_c - b10_t + b01_c - b01_t + b00_c - b00_t + b21_c - b21_t) / b10_c,
    (b10_c - b10_t + b01_c - b01_t + b00_c - b00_t + b20_c - b20_t) / b10_c,
    (b10_c - b10_t + b01_c - b01_t + b21_c - b21_t + b20_c - b20_t) / b10_c,
    (b10_c - b10_t + b01_c - b01_t + b00_c - b00_t + b21_c - b21_t + b20_c - b20_t) / b10_c,
])
PUB7 = np.array([
    (p10_c - p10_t + p01_c - p01_t + p21_c - p21_t) / p10_c,
    (p10_c - p10_t + p01_c - p01_t) / p10_c,
    (p10_c - p10_t + p01_c - p01_t + p00_c - p00_t) / p10_c,
    (p10_c - p10_t + p01_c - p01_t + p20_c - p20_t) / p10_c,
    (p10_c - p10_t + p01_c - p01_t + p00_c - p00_t + p21_c - p21_t) / p10_c,
    (p10_c - p10_t + p01_c - p01_t + p00_c - p00_t + p20_c - p20_t) / p10_c,
    (p10_c - p10_t + p01_c - p01_t + p21_c - p21_t + p20_c - p20_t) / p10_c,
    (p10_c - p10_t + p01_c - p01_t + p00_c - p00_t + p21_c - p21_t + p20_c - p20_t) / p10_c,
])
UL7 = conservative_upper(UB_an[7], BUB7, PUB7)

# ── π_8  (0r→1n) ────────────────────────────────────────────────────────
BUB8 = np.column_stack([
    (b01_c - b01_t + b21_c - b21_t) / b01_c,
    b10_t / b01_c,
    (b01_c - b01_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t) / b01_c,
    (b01_c - b01_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t + b20_c - b20_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b20_c - b20_t + b21_c - b21_t) / b01_c,
    (b01_c - b01_t + b00_c - b00_t + b20_c - b20_t + b21_c - b21_t) / b01_c,
])
PUB8 = np.array([
    (p01_c - p01_t + p21_c - p21_t) / p01_c,
    p10_t / p01_c,
    (p01_c - p01_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t) / p01_c,
    (p01_c - p01_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t + p20_c - p20_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p20_c - p20_t + p21_c - p21_t) / p01_c,
    (p01_c - p01_t + p00_c - p00_t + p20_c - p20_t + p21_c - p21_t) / p01_c,
])
UL8 = conservative_upper(UB_an[8], BUB8, PUB8)

UL_consv = np.array([UL0, UL1, UL2, 1.0, UL4, UL5, 1.0, UL7, UL8])

print("\n── Conservative CI upper limits (lower limits same as Naïve) ───────")
print(f"  {'Parameter':<14}  {'Lower CI':>10}  {'Consv Upper':>12}")
for nm, lo, hi in zip(PI_NAMES, L_naive, np.minimum(1, UL_consv)):
    print(f"  {nm:<14}  {lo:10.4f}  {hi:12.4f}")



# ═════════════════════════════════════════════════════════════════════════════
# 7.  TABLE 5 panel (b) — RESTRICTED PREFERENCES (monetized utility)
# ═════════════════════════════════════════════════════════════════════════════
# Under monetized preferences π_{0r→1n}=0, so equation (2) uniquely identifies
#   π_{1n→1r} = −(p^j_{1n} − p^a_{1n}) / p^a_{1n}
print("\n── TABLE 5 panel (b)  ─  Restricted preferences ────────────────────")
pi1n1r = -(p10_t - p10_c) / p10_c
bs_1n1r = -(b10_t - b10_c) / b10_c
se_1n1r = bs_1n1r.std()
print(f"  π_{{1n→1r}} (point-identified) = {pi1n1r:.4f}")
print(f"  95% CI = [{pi1n1r - 1.96*se_1n1r:.4f},  {pi1n1r + 1.96*se_1n1r:.4f}]")



# ═════════════════════════════════════════════════════════════════════════════
# 8.  COMPOSITE MARGINS
# ═════════════════════════════════════════════════════════════════════════════
print("\n── Composite Margins ───────────────────────────────────────────────")

# ── (A) Not working → Working (0n/0p → 1r/1n/2n) — POINT IDENTIFIED ───────
f = np.array([0, p00_c, p01_c, 0, p01_c, p01_c, 0, 0, p01_c]) / (p01_c + p00_c)
r_lo = linprog( f, A_eq=A_mat, b_eq=b_vec, bounds=BOUNDS, **LP_OPTS)
r_hi = linprog(-f, A_eq=A_mat, b_eq=b_vec, bounds=BOUNDS, **LP_OPTS)
pi0plus = (p00_c + p01_c - (p00_t + p01_t)) / (p01_c + p00_c)
bs_0p   = (b00_c + b01_c - (b00_t + b01_t)) / (b01_c + b00_c)
se_0p   = bs_0p.std()
print(f"\n  (A) Not working → Working (0 → 1+):  POINT IDENTIFIED")
print(f"      Estimate  =  {pi0plus:.4f}")
print(f"      95% CI    = [{pi0plus - 1.96*se_0p:.4f}, {pi0plus + 1.96*se_0p:.4f}]")
print(f"      LP check  = [{r_lo.fun:.4f}, {-r_hi.fun:.4f}]")

# ── (B) Off welfare → On welfare (n → p) ─────────────────────────────────
f = np.array([0, p00_c, 0, p20_c, 0, 0, 0, p10_c, 0]) / (p00_c + p10_c + p20_c)
r_lo = linprog( f, A_eq=A_mat, b_eq=b_vec, bounds=BOUNDS, **LP_OPTS)
r_hi = linprog(-f, A_eq=A_mat, b_eq=b_vec, bounds=BOUNDS, **LP_OPTS)

plb_B = (p00_c - p00_t + p10_c - p10_t + p20_c - p20_t) / (p20_c + p10_c + p00_c)
blb_B = (b00_c - b00_t + b10_c - b10_t + b20_c - b20_t) / (b20_c + b10_c + b00_c)
L_B   = plb_B - 1.65 * blb_B.std()

pub_B   = (p00_c-p00_t + p10_c-p10_t + p20_c-p20_t + p01_c-p01_t + p21_c-p21_t) \
          / (p20_c + p10_c + p00_c)
bub_B1  = (b00_c-b00_t + b10_c-b10_t + b20_c-b20_t + b01_c-b01_t + b21_c-b21_t) \
          / (b20_c + b10_c + b00_c)
bub_B2  = (b00_c-b00_t + b10_c-b10_t + b20_c-b20_t + b01_c-b01_t) \
          / (b20_c + b10_c + b00_c)
pub_B2  = (p00_c-p00_t + p10_c-p10_t + p20_c-p20_t + p01_c-p01_t) \
          / (p20_c + p10_c + p00_c)

T_B     = np.column_stack([bub_B1, bub_B2]) - np.array([pub_B, pub_B2])
UL_B_c  = pub_B + np.quantile((-T_B).max(axis=1), 0.95)
UL_B_n  = pub_B + 1.65 * bub_B1.std()

print(f"\n  (B) Off welfare → On welfare (n → p):")
print(f"      Estimate          = {{{max(0,plb_B):.4f}, {min(1,pub_B):.4f}}}")
print(f"      Naïve CI          = [{max(0,L_B):.4f}, {min(1,UL_B_n):.4f}]")
print(f"      Conservative CI   = [{max(0,L_B):.4f}, {min(1,UL_B_c):.4f}]")

# ── (C) On welfare → Off welfare (p → n) ─────────────────────────────────
plb_C = (p00_t - p00_c + p20_t - p20_c + p10_t - p10_c) / (p21_c + p11_c + p01_c)
# Note: MATLAB line 625 has a typo (b20_t used instead of b10_t); corrected here
blb_C = (b00_t - b00_c + b20_t - b20_c + b10_t - b10_c) / (b21_c + b11_c + b01_c)
L_C   = plb_C - 1.65 * blb_C.std()

pub_C1 = (p01_c - p01_t + p21_c - p21_t) / (p01_c + p11_c + p21_c)
bub_C1 = (b01_c - b01_t + b21_c - b21_t) / (b01_c + b11_c + b21_c)
bub_C2 = (b00_t + b20_t + b10_t)          / (b01_c + b11_c + b21_c)
bub_C3 = (b01_c - b01_t)                   / (b01_c + b11_c + b21_c)
pub_C2 = (p00_t + p20_t + p10_t)           / (p01_c + p11_c + p21_c)
pub_C3 = (p01_c - p01_t)                   / (p01_c + p11_c + p21_c)

T_C    = np.column_stack([bub_C1, bub_C2, bub_C3]) - np.array([pub_C1, pub_C2, pub_C3])
UL_C_c = pub_C1 + np.quantile((-T_C).max(axis=1), 0.95)
UL_C_n = pub_C1 + 1.65 * bub_C1.std()

print(f"\n  (C) On welfare → Off welfare (p → n):")
print(f"      Estimate          = {{{max(0,plb_C):.4f}, {min(1,pub_C1):.4f}}}")
print(f"      Naïve CI          = [{max(0,L_C):.4f}, {min(1,UL_C_n):.4f}]")
print(f"      Conservative CI   = [{max(0,L_C):.4f}, {min(1,UL_C_c):.4f}]")

# ── (D) On welfare not working → Off welfare  (0r → n) ───────────────────
f_D  = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1], dtype=float)
r_lo = linprog( f_D, A_eq=A_mat, b_eq=b_vec, bounds=BOUNDS, **LP_OPTS)
r_hi = linprog(-f_D, A_eq=A_mat, b_eq=b_vec, bounds=BOUNDS, **LP_OPTS)

plb_D = p00_t - p00_c + p20_t - p20_c + p10_t - p10_c
blb_D = b00_t - b00_c + b20_t - b20_c + b10_t - b10_c
L_D   = plb_D - 1.65 * blb_D.std()

pub_D1 = (p01_c - p01_t + p21_c - p21_t) / p01_c
bub_D1 = (b01_c - b01_t + b21_c - b21_t) / b01_c
bub_D2 = (b00_t + b20_t + b10_t)          / b01_c
bub_D3 = (b01_c - b01_t)                   / b01_c
pub_D2 = (p00_t + p20_t + p10_t)           / p01_c
pub_D3 = (p01_c - p01_t)                   / p01_c

T_D    = np.column_stack([bub_D1, bub_D2, bub_D3]) - np.array([pub_D1, pub_D2, pub_D3])
UL_D_c = pub_D1 + np.quantile((-T_D).max(axis=1), 0.95)
UL_D_n = pub_D1 + 1.65 * bub_D1.std()

print(f"\n  (D) On welfare not working → Off welfare (0r → n):")
print(f"      LP check          = [{r_lo.fun:.4f}, {-r_hi.fun:.4f}]")
print(f"      Estimate          = {{{max(0,plb_D):.4f}, {min(1,pub_D1):.4f}}}")
print(f"      Naïve CI          = [{max(0,L_D):.4f}, {min(1,UL_D_n):.4f}]")
print(f"      Conservative CI   = [{max(0,L_D):.4f}, {min(1,UL_D_c):.4f}]")

print("\n" + "="*65)
print("Done — compare output with Table 5 of Kline & Tartari (2016, AER).")
print("="*65)

