
import numpy as np
import scipy.optimize

def check_duals():
    # Simple LP: max x subject to x <= 1
    c = np.array([-1.0]) # linprog minimizes
    A = np.array([[1.0]])
    b = np.array([1.0])
    
    res = scipy.optimize.linprog(c, A_ub=A, b_ub=b, method='highs')
    print("Status:", res.status)
    print("Primal x:", res.x)
    print("Dual (ineqlin marginals):", res.ineqlin.marginals if hasattr(res, 'ineqlin') else "N/A")
    print("Dual (upper bounds marginals):", res.upper.marginals if hasattr(res, 'upper') else "N/A")

if __name__ == "__main__":
    check_duals()
