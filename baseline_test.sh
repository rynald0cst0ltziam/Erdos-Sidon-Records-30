python3 verify.py | tee baseline.out
grep -q "Theorem 3.3 gives b_inf <= 1.9636454840813407" baseline.out
