from bisection_method import find_root

def f(x):
    return x**3 - 4*x - 9

root, logs = find_root(
        interval=(2, 3),
        tolerance=0.0001,
        f=f,
        print_output=True,
        get_logs=True
    )

print(root, logs)