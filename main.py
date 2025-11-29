from bisection_method import find_root

def f(x):
    return x**3 - 4*x - 9

def run_bisection(a, b, tolerance, f, print_output=False):
    root, logs = find_root(
        interval=(a, b),
        tolerance=tolerance,
        f=f,
        print_output=print_output,
        get_logs=True
    )
    return root, logs