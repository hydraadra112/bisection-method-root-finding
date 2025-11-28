from bisection_method import find_root

def f(x):
    return x**3 - 4*x - 9

if __name__ == '__main__':
    root, logs = find_root(interval=(2.0, 3.0), # Interval to be used, packed as (a, b)
                tolerance=0.0001, # Tolerance
                f=f, # Function to use
                print_output=True, # Optional to see output per iteration
                get_logs=True
                )   
    
    print(root)
    print(logs, type(logs))