from bisection_method import find_root

def f(x):
    return x**3 - 4*x - 9

if __name__ == '__main__':
    root = find_root((2, 3), tolerance=0.0001, f=f, print_output=True)
    print(root)