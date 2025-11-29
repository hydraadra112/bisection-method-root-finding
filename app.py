import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from main import run_bisection


# FOR EQUATION INPUT
def create_user_function(expr):
    x = sp.symbols('x')
    parsed = sp.sympify(expr)

    def f(val):
        return float(parsed.subs(x, val))

    return f


# STREAMLIT UI
st.title("🔢 Bisection Method Calculator")
st.write("Enter the function and interval parameters below.")


# OTHER USER INPUTS
equation = st.text_input("Equation f(x):", "x**3 - 4*x - 9")
a = st.number_input("a (lower bound)", value=2.0)
b = st.number_input("b (upper bound)", value=3.0)
start_x = st.number_input("x (starting value)", value=0.0)  # Required by instructions
tolerance = st.number_input("Tolerance ℰ", value=0.0001, format="%.6f")



if st.button("Compute Root"):
    try:
        f = create_user_function(equation)

        root, logs = run_bisection(a, b, tolerance, f)

        st.success(f"Root found: {root}")

        # DEBUG SECTION
        # st.subheader("🔍 DEBUG INFORMATION")
        # st.write("Raw logs returned by find_root():")
        # st.write(logs)
        # st.write("Type of logs:", type(logs))
        # if isinstance(logs, list) and len(logs) > 0:
        #     st.write("First log entry:", logs[0])
        # else:
        #     st.write("Logs is not a list or is empty.")
        # st.markdown("---")

        # BUILD TABLES
        if not isinstance(logs, list) or len(logs) == 0:
            st.warning("No valid iteration logs were returned.")
            st.stop()

        processed = []

        for log in logs:
            iteration = log.get("iteration", None)
            a_i = log.get("a", None)
            b_i = log.get("b", None)
            c_i = log.get("midpoint", None)
            f_c_i = log.get("f(midpoint)", None)
            f_a_i = log.get("f(a)", None)
            f_b_i = log.get("f(b)", None)
            # SKIP INVALID LOGS
            if a_i is None or b_i is None or c_i is None or f_c_i is None:
                continue

            diff = abs(b_i - a_i)
            within_tol = diff < tolerance

            processed.append({
                "Iteration": iteration,
                "a": a_i,
                "b": b_i,
                "c": c_i,
                "f(a)": f_a_i,
                "f(b)": f_b_i,
                "f(c)": f_c_i,
                "|b - a|": diff,
                "within ℰ": within_tol
            })


        if processed and processed[-1]["Iteration"] is not None:
            final_iteration = processed[-1]["Iteration"] + 1
        else:
            final_iteration = len(processed) + 1

        processed.append({
            "Iteration": final_iteration,
            "a": a_i,
            "b": b_i,
            "c": root,
            "f(a)": f_a_i,
            "f(b)": f_b_i,
            "f(c)": f(root),
            "|b - a|": abs(b_i - a_i),
            "within ℰ": True
        })

        # CONVERT TO TABLE AND DISPLAY
        st.subheader("📊 Iteration Logs")
        df = pd.DataFrame(processed)
        st.dataframe(df)

        st.subheader("📈 Function Graph with Root")

        # Generate x-values slightly beyond [a, b] for better visualization
        x_vals = np.linspace(a - 1, b + 1, 400)
        y_vals = [f(xv) for xv in x_vals]

        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, label=f"f(x) = {equation}", color='blue')
        plt.axhline(0, color='black', linewidth=0.7)  # x-axis

        # Plot all midpoints (c values) from iterations
        c_vals = [entry["c"] for entry in processed]
        plt.scatter(c_vals, [f(c) for c in c_vals], color='orange', label="Midpoints (iterations)", zorder=5)

        # Highlight the root
        plt.scatter(root, f(root), color='red', s=100, zorder=10, label=f"Root ≈ {root:.6f}")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Function Plot with Bisection Method Progress")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)


    except Exception as e:
        st.error(f"Error: {str(e)}")