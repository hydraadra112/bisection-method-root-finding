# **Bisection Method - Root Finding**

## Details

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

## Setup

To properly set up this application, follow the steps below:

1. **Install `uv` package manager**

```bash
# For MacOS or Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# For Windows
# Open Powershell as Administrator
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

You can still use the application without `uv`, as long as you use `pip` to manage your dependencies.

Although we strongly recommend using `uv` (for reproducibility) for anyone with a background in programming, especially if you seek to contribute.

2. **Clone/Download this Repository** (This assumes you have Git installed)

```bash
# Run this command to clone the repository
git clone https://github.com/hydraadra112/bisection-method-root-finding.git
```

3. **Navigate to the project directory**. You could use the `cd` command in your terminal, or simply open up the terminal where the project is stored.

```bash
# After cloning
cd bisection-method-root-finding
```

4. **Install the Dependencies**

```bash
# If you use uv
uv sync

# If you use pip
pip install numpy==2.3.5 pandas==2.3.3 streamlit==1.51.0 sympy==1.14.0
```

4. **Run the Streamlit application**

```bash
streamlit run streamlit_app.py
```

And it should open up the Streamlit application in your local files.

**Enjoy root finding!** 😃
