#!/usr/bin/env python3
"""
Reproduce the 'two windows' hand-verifiable constant in CHO (Theorem 2.1)
by maximizing min(f1, f2) over 0 <= w1 <= w2 <= 2.

Source formulas:
- Inequality (1) on p.113:  b∞ ≤ τ + 1/τ − τ( w1(α1−1)^2 + (2−w2)(α2−1)^2 )
- Inequality (2) on p.114:  b∞ ≤ cτ + 1/(cτ) − (τ/c^2)*((w2−w1−2c)(c−(α2−α1))_+^2 + w1(c−α1)_+^2)

CHO parameter choice (p.114):
τ=1.07950, α1=0.72720, α2=1.31609, c=0.86838

We certify: max_{region} min(f1,f2) <= 1.99058 (up to tiny tolerance)
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from itertools import combinations
from typing import Iterable, Optional, Tuple

# High precision to avoid “float wins a proof” nonsense.
getcontext().prec = 80


D = Decimal

@dataclass(frozen=True)
class Params:
    tau: Decimal
    a1: Decimal
    a2: Decimal
    c: Decimal

P = Params(
    tau=D("1.07950"),
    a1=D("0.72720"),
    a2=D("1.31609"),
    c=D("0.86838"),
)

TARGET = D("1.99058")


def sq(x: Decimal) -> Decimal:
    return x * x


def pos(x: Decimal) -> Decimal:
    return x if x > 0 else D("0")


def f1(w1: Decimal, w2: Decimal, p: Params = P) -> Decimal:
    # b∞ ≤ τ + 1/τ − τ( w1(α1−1)^2 + (2−w2)(α2−1)^2 )   (CHO eq (1))
    tau = p.tau
    term = w1 * sq(p.a1 - D("1")) + (D("2") - w2) * sq(p.a2 - D("1"))
    return tau + (D("1") / tau) - tau * term


def f2(w1: Decimal, w2: Decimal, p: Params = P) -> Decimal:
    # b∞ ≤ cτ + 1/(cτ) − (τ/c^2)*((w2−w1−2c)(c−(α2−α1))_+^2 + w1(c−α1)_+^2)   (CHO eq (2))
    tau = p.tau
    c = p.c
    gap = p.a2 - p.a1
    A = sq(pos(c - gap))
    B = sq(pos(c - p.a1))
    term = (w2 - w1 - D("2") * c) * A + w1 * B
    return c * tau + (D("1") / (c * tau)) - (tau / sq(c)) * term


def feasible(w1: Decimal, w2: Decimal) -> bool:
    return (w1 >= 0) and (w2 >= 0) and (w1 <= w2) and (w2 <= 2)


# ---- LP solver by vertex enumeration in (w1, w2, z) ----
# Maximize z subject to:
#   z <= f1(w1,w2)
#   z <= f2(w1,w2)
#   0 <= w1
#   w1 <= w2
#   w2 <= 2
#   0 <= w2
#
# Since f1,f2 are affine in w1,w2 for these parameters (because _+ are positive),
# this is a small LP; vertices are intersections of 3 active constraints.

# Represent constraints as equalities for vertex generation:
#   w1=0
#   w2=0
#   w2=2
#   w1=w2
#   z=f1
#   z=f2
CONSTRAINTS = ("w1=0", "w2=0", "w2=2", "w1=w2", "z=f1", "z=f2")


def affine_coeffs_of_f(fun) -> Tuple[Decimal, Decimal, Decimal]:
    """
    Extract affine coefficients a + b*w1 + c*w2 by evaluating fun at 3 points.
    Works because f1,f2 are affine in w1,w2 (given our parameters).
    """
    a = fun(D("0"), D("0"))
    b = fun(D("1"), D("0")) - a
    c = fun(D("0"), D("1")) - a
    return a, b, c


A1, B1, C1 = affine_coeffs_of_f(f1)
A2, B2, C2 = affine_coeffs_of_f(f2)


def solve_vertex(active: Tuple[str, str, str]) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
    """
    Solve 3 equalities in variables (w1,w2,z). Return (w1,w2,z) if feasible.
    Uses exact Decimal arithmetic; linear system is tiny so we solve explicitly.
    """
    # Build linear system M x = rhs with x=(w1,w2,z)
    rows = []
    rhs = []

    for c in active:
        if c == "w1=0":
            rows.append((D("1"), D("0"), D("0"))); rhs.append(D("0"))
        elif c == "w2=0":
            rows.append((D("0"), D("1"), D("0"))); rhs.append(D("0"))
        elif c == "w2=2":
            rows.append((D("0"), D("1"), D("0"))); rhs.append(D("2"))
        elif c == "w1=w2":
            rows.append((D("1"), D("-1"), D("0"))); rhs.append(D("0"))
        elif c == "z=f1":
            # z = A1 + B1*w1 + C1*w2  ->  -B1*w1 -C1*w2 + 1*z = A1
            rows.append((-B1, -C1, D("1"))); rhs.append(A1)
        elif c == "z=f2":
            rows.append((-B2, -C2, D("1"))); rhs.append(A2)
        else:
            raise ValueError(c)

    # Solve 3x3 by Cramer's rule (stable enough in Decimal at high precision)
    (a11,a12,a13),(a21,a22,a23),(a31,a32,a33) = rows
    b1_, b2_, b3_ = rhs

    def det3(r1, r2, r3) -> Decimal:
        (x1,x2,x3),(y1,y2,y3),(z1,z2,z3) = r1,r2,r3
        return x1*(y2*z3 - y3*z2) - x2*(y1*z3 - y3*z1) + x3*(y1*z2 - y2*z1)

    Mdet = det3(rows[0], rows[1], rows[2])
    if Mdet == 0:
        return None

    # Replace columns
    r1x = (b1_, a12, a13); r2x = (b2_, a22, a23); r3x = (b3_, a32, a33)
    r1y = (a11, b1_, a13); r2y = (a21, b2_, a23); r3y = (a31, b3_, a33)
    r1z = (a11, a12, b1_); r2z = (a21, a22, b2_); r3z = (a31, a32, b3_)

    w1 = det3(r1x, r2x, r3x) / Mdet
    w2 = det3(r1y, r2y, r3y) / Mdet
    z  = det3(r1z, r2z, r3z) / Mdet

    if not feasible(w1, w2):
        return None

    # Check inequalities: z <= f1, z <= f2
    if z > f1(w1, w2) + D("1e-25"):
        return None
    if z > f2(w1, w2) + D("1e-25"):
        return None

    # Also check the region inequalities explicitly (tight tolerance)
    if w1 < -D("1e-25") or w2 < -D("1e-25") or w2 > D("2") + D("1e-25") or w1 - w2 > D("1e-25"):
        return None

    return (w1, w2, z)


def main() -> None:
    best = None

    for active in combinations(CONSTRAINTS, 3):
        v = solve_vertex(active)
        if v is None:
            continue
        if best is None or v[2] > best[2]:
            best = v

    assert best is not None, "No feasible vertices found (bug)."
    w1, w2, z = best
    val1 = f1(w1, w2)
    val2 = f2(w1, w2)

    print("Params:", P)
    print("Affine f1 = A1 + B1*w1 + C1*w2 =", (A1, B1, C1))
    print("Affine f2 = A2 + B2*w1 + C2*w2 =", (A2, B2, C2))
    print()
    print("Max over region of min(f1,f2) occurs at:")
    print("  w1 =", w1)
    print("  w2 =", w2)
    print("  z  =", z)
    print("  f1 =", val1)
    print("  f2 =", val2)
    print()
    print("TARGET (CHO Thm 2.1):", TARGET)
    print("z - TARGET =", z - TARGET)

    # Certification: allow microscopic tolerance for Decimal algebra + extraction
    tol = D("1e-18")
    assert z <= TARGET + tol, f"FAILED: got z={z} > {TARGET}"

    print("\nCERTIFIED: max min(f1,f2) <= 1.99058 (within tolerance).")


if __name__ == "__main__":
    main()
