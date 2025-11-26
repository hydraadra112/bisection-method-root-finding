from typing import Callable, Tuple

def find_root(interval: Tuple[int|float, int|float],
              tolerance: int|float,
              f: Callable,
              print_output: bool = False,
              ) -> float:

    """ Finds the root of given function by repeatedly halving the given interval 
    Args:
        interval (tuple(int or float, int or float)): The interval to be used find the root
        tolerance (int or float): The specified maximum allowable error hat determines when to stop the iterative process of finding a root
        f (Callable): The function to be used to find its root with given interval
        print_output (bool): Prints the output of interval and midpoint for every iteration
    
    Returns:
        int or float: The root of given function within given interval
    """
    (a, b) = interval

    # Check if given data is integer or float
    if not isinstance(a, int|float) or not isinstance(b, int|float):
        raise ValueError("Interval is not in integer or float data type.")
    
    if not isinstance(tolerance, int|float):
        raise ValueError("Tolerance is not in integer or float data type.")
    
    if f(a) * f(b) > 0:
        raise ValueError("The function must have opposite signs at the interval endpoints.\n" \
        f"Your intervals ({a}, {b}): f(a)={f(a)}, f(b)={f(b)}\n" \
        "Try another interval.")
    
    iteration = 1
    
    try:
        while abs(b - a) > tolerance:
            midpoint = (a + b) / 2

            if print_output:
                print(f"Iteration {iteration}: Interval = [{a:.9f}, {b:.9f}], Midpoint = {midpoint:.9f}")
            
            if f(midpoint) * f(a) < 0:
                b = midpoint
            else:
                a = midpoint

            iteration += 1

        return (a + b) / 2
    except Exception as e:
        raise RuntimeError(f"Root finding failed: {e}")

# For future improvements, you can add `find_root` here.