# Computing the event-level epsilon for our probabilistic length protection strategy
# Requirements: Install sympy and numpy in the environment

import sympy as sp
import numpy as np

def bisection_method_modified(f, y, l, u, tol):
    """
    Find the value x such that f(x) = y using the bisection method.
    Extra constraint: f(x) <= y
    
    Parameters:
    f - Function to be evaluated. It should accept a single numeric argument.
    y - The target value for which to find f(x) = y.
    l - Lower limit of the initial interval.
    u - Upper limit of the initial interval.
    tol - Tolerance for the convergence criterion.
    
    Returns:
    x - Approximation to the solution f(x) = y.
    """
    fl = f(l) - y
    fu = f(u) - y

    # Check if the target value y is within the range of f in [l, u]
    if fl * fu > 0:
        raise ValueError(
            f"The target value y={y} is not within the range of f in the interval [l={l}, u={u}].\n"
            f"f(l)={f(l)}, f(u)={f(u)}"
        )
    
    while (u - l) / 2 > tol:
        m = (l + u) / 2
        fm = f(m) - y
        if fm == 0:  # Exact solution found
            return m
        elif fl * fm < 0:
            u = m
            fu = fm
        else:
            l = m
            fl = fm
    
    # Return the midpoint as the best approximation
    return (l + u) / 2

# Define symbolic variables
eps, n, eps_l, Delta = sp.symbols('eps n eps_l Delta')

# Define the formula tight
formula =2*sp.ln (
    sp.exp(-2 * n * eps_l / Delta) * (
        1 + (
            1 - sp.exp(-eps_l / Delta)
        ) * (
            sp.exp(eps / 2) * sp.exp(2 * eps_l / Delta) * (
                1 - sp.exp(eps * n / 2) * sp.exp(2 * n * eps_l / Delta)
            ) / (
                1 - sp.exp(eps / 2) * sp.exp(2 * eps_l / Delta)
            )
        )
    )
)



#Define the non-tight formula
# Define symbolic variables
eps, n_sym, eps_l_sym, s = sp.symbols('eps n eps_l Delta')
# Define the symbolic function
formula2 =sp.ln((sp.exp(eps*n_sym + eps + 2*eps_l_sym*n_sym/s + 2*eps_l_sym/s) - 
        sp.exp(eps*n_sym + eps + 2*eps_l_sym*n_sym/s) + sp.exp(eps) - 1) *sp.exp(-2*eps_l_sym*n_sym/s) / (sp.exp(eps + 2*eps_l_sym/s) - 1)
)
# Function to compute f(eps) for given parameters
def calcular_f(n_value, eps_l_value, s_value):
    """
    Substitute specific values into the formula and create a numerical function.
    """
    esp_con_valores = formula.subs({n: n_value, eps_l: eps_l_value, Delta: s_value})
    esp_simplificado = sp.simplify(esp_con_valores)
    esp_func = sp.lambdify(eps, esp_simplificado, modules=["numpy"])
    return esp_func
    
# Function to compute f(eps) for given parameters
def calcular_f2(n_value, eps_l_value, s_value):
    """
    Substitute specific values into the formula and create a numerical function.
    """
    esp_con_valores = formula2.subs({n: n_value, eps_l: eps_l_value, Delta: s_value})
    esp_simplificado = sp.simplify(esp_con_valores)
    esp_func = sp.lambdify(eps, esp_simplificado, modules=["numpy"])
    return esp_func

# Example usage
if __name__ == "__main__":
    # Parameters
    n_value = 10
    eps_l_value = 0.7
    s_value = 1
    # Create the numerical function
    funcion_f = calcular_f(n_value, eps_l_value, s_value)
    funcion_f2 = calcular_f(n_value, eps_l_value, s_value)

    # Target value and bisection parameters
    y = 20
    l = y / n_value  # Lower bound
    u = l+1  # Upper bound (adjust if necessary)
    tolerance = 1e-15

    try:
        # Perform the bisection method
        inverse_value = bisection_method_modified(funcion_f, y, l, u, tolerance)
        inverse_value2 = bisection_method_modified(funcion_f2, y, l, u, tolerance)
        print("The approximate value of the inverse is:", inverse_value2)
        print("The final privacy is:", funcion_f2(inverse_value2))
        
        print("The tight value of the inverse is:", inverse_value)
        print("The final privacy is:", funcion_f(inverse_value))
    except ValueError as e:
        print("Error during bisection method:", e)

