# Bisection Method - Root Finding

A bisection method root finding solver, our final project for **CCS 239 - Optimization Theory and Applications** course.

Prepared by:

- Artacho, Cristopher Ian
- Carado, John Manuel
- Tacuel, Allan Andrews

from BSCS 4-A, Batch Ryzen (2022 - 2026).

How to use the bisection method root finding solver:

```python
# Import the custom made module
from bisection_method import find_root

# Define your function here
def f(x):
    return x**3 - 4*x - 9

# Find root
root, logs = find_root(interval=(2, 3), # Interval to be used, packed as (a, b)
                tolerance=0.0001, # Tolerance
                f=f,              # Function to use
                print_output=True # Optional to see output per iteration
                get_logs=True     # Optional to get logs of calculation
                )

# Print out root from console
print(root, logs)
```
