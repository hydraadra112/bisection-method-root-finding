from typing import Callable, Tuple, List, Dict
from numbers import Real

def find_root(
    interval: Tuple[Real, Real],
    tolerance: Real,
    f: Callable[[Real], Real],
    print_output: bool = False,
    get_logs: bool = False
    ):
    """ Finds the root of given function by repeatedly halving the given interval 
    Args:
        interval (tuple(int or float, int or float)): The interval to be used find the root
        tolerance (int or float): The specified maximum allowable error hat determines when to stop the iterative process of finding a root
        f (Callable): The function to be used to find its root with given interval
        print_output (bool): Prints the output of interval and midpoint for every iteration
        get_logs (bool): Decide if to get calculations for every iteration  
    """
    a, b = interval

    # Validate inputs
    if not (isinstance(a, Real) and isinstance(b, Real)):
        raise ValueError("Interval endpoints must be real numbers.")
    if not isinstance(tolerance, Real):
        raise ValueError("Tolerance must be a real number.")

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError(
            "Function must have opposite signs at interval endpoints.\n"
            f"Your interval ({a}, {b}): f(a)={fa}, f(b)={fb}"
        )

    logs: List[Dict[str, Real]] = [] if get_logs else None

    iteration = 1
    try:
        while abs(b - a) > tolerance:
            midpoint = (a + b) / 2
            fm = f(midpoint)

            if get_logs:
                logs.append({
                    "iteration": iteration,
                    "a": a,
                    "b": b,
                    "midpoint": midpoint,
                    "f(a)": fa,
                    "f(b)": fb,
                    "f(midpoint)": fm
                })

            if print_output:
                print(
                    f"Iteration {iteration}: "
                    f"[{a:.9f}, {b:.9f}], midpoint = {midpoint:.9f}"
                )

            # Decide which interval to keep
            if fa * fm < 0:
                b = midpoint
                fb = fm
            else:
                a = midpoint
                fa = fm

            iteration += 1

        root = (a + b) / 2

        return (root, logs) if get_logs else root

    except Exception as e:
        raise RuntimeError(f"Root finding failed: {e}")
