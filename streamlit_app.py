import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable
from numbers import Real
import altair as alt
from bisection_method import find_root

# --- PARSER FUNCTION ---
def equation_parser(equation_str: str) -> Tuple[Callable, str]:
    """
    Parses a string expression into a callable function using SymPy.
    Args:
        equation_str (str): The equation as string to parse as callable function
    
    Returns:
        tuple: A tuple containing:
            - f_dynamic (Callble): A callable function
            - error_msg (str): Error message for parsing
    """
    try:
        x = sp.symbols('x')
        expr = sp.sympify(equation_str)
        
        f = sp.lambdify(x, expr, modules=['numpy', 'math'])
        return f, None
    except Exception as e:
        return None, f"Error parsing function: {e}"

# HELPER FUNCTIONS
def get_ba_and_within(df: pd.DataFrame, tolerance: Real) -> pd.DataFrame:
    """ Calculates |b-a| & checks if it is within tolerance """
    additional_data: List[Dict[str, Real]] = []
    for i in range(len(df)):
        df_current_index = df.iloc[i]
        val = abs(df_current_index['b'] - df_current_index['a'])  # Ensure positive
        within = val < tolerance
        additional_data.append({
            '|b-a|': val,
            'within_tolerance': within
        })
    return pd.DataFrame(additional_data)

def plot_function(f: Callable, a: Real, b: Real, root: Real) -> None:
    
    # Expand the range slightly for the visual
    margin = abs(b - a) * 0.5 if a != b else 1.0
    x_min = min(a, b) - margin
    x_max = max(a, b) + margin
    
    # Generate 100 points for a smooth line
    x_range = np.linspace(x_min, x_max, 100)
    
    try:
        y_range = f(x_range)
    except Exception:
        # Fallback for functions that might not vectorize perfectly or have domain errors
        y_range = [f(val) for val in x_range]

    source = pd.DataFrame({
        'x': x_range,
        'f(x)': y_range
    })

    # B. Create Data for the Root Marker
    root_data = pd.DataFrame({
        'x': [root],
        'f(x)': [f(root)] 
    })

    # C. Build Altair Layers
    # 1. The Curve
    line_chart = alt.Chart(source).mark_line().encode(
        x=alt.X('x', title='x'),
        y=alt.Y('f(x)', title='f(x)')
    )

    # 2. The Zero Line (Axis)
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')

    # 3. The Root Marker (Red Dot)
    point_chart = alt.Chart(root_data).mark_circle(color='red', size=100).encode(
        x='x',
        y='f(x)',
        tooltip=[alt.Tooltip('x', format='.5f'), alt.Tooltip('f(x)', format='.5f')]
    )

    # D. Combine and Display
    st.altair_chart((line_chart + zero_line + point_chart).interactive(), width='stretch')


def main():
    # STREAMLIT UI
    st.title("🔢 Bisection Method Root Finder")
    st.write("Enter the function and interval parameters below.")
    
    # Create two tabs
    tab1, tab2 = st.tabs(["Root Finder", "Instructions"])
    
    with tab1:
        st.info("💡 You can use standard Python syntax (e.g., `x**3`) or SymPy syntax.")
        st.caption("Trig functions supported: `sin(x)`, `cos(x)`, `exp(x)`.")
        
        function_input = st.text_input("Enter function f(x):", value="x**3 - 4*x - 9")
        
        f_dynamic, error_msg = equation_parser(function_input)
        
        if error_msg:
            st.error(error_msg)
        
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Interval (a)", value=2.0)
            b = st.number_input("Interval (b)", value=3.0)
        with col2:
            tolerance = st.number_input("Tolerance ℰ", value=0.0001, format="%.5f", step=0.0001)
            # Only enable button if function parsed correctly
            compute_root = st.button("Compute Root", disabled=(f_dynamic is None))

        if compute_root and f_dynamic:
            try:
                # Pass the dynamic function 'f_dynamic' instead of a hardcoded one
                root, logs = find_root((a,b), tolerance, f_dynamic, False, True)
                
                df = pd.DataFrame(logs)
                ba_within = get_ba_and_within(df.copy(), tolerance)
                df = pd.concat([df, ba_within], axis=1)

                df.drop(labels=['iteration'], axis=1, inplace=True)    
                df.index += 1

                st.success(f"Root found: {root}")

                # Create a final row explicitly with the root
                final_row = {
                    'a': root,
                    'b': root,
                    'midpoint': root,
                    'f(a)': f_dynamic(root),
                    'f(b)': f_dynamic(root),
                    'f(midpoint)': f_dynamic(root),
                    '|b-a|': 0.0,
                    'within_tolerance': True
                }
                df_final = pd.concat([df, pd.DataFrame([final_row])], ignore_index=True)

                col3, col4 = st.columns(2)
                with col3:
                    st.dataframe(df_final)
                with col4:
                    # Pass f_dynamic to the plotter
                    plot_function(f_dynamic, a, b, root)
            except Exception as e:
                st.error(f"An error occurred during computation: {e}")

    with tab2:
        st.header("How the Bisection Method Works")
        st.markdown("""
        The **Bisection Method** is a numerical technique to find a root of a continuous function \(f(x)\) on an interval \([a, b]\) where \(f(a)\) and \(f(b)\) have opposite signs.
        
        **Steps:**
        1. Compute the midpoint: \(midpoint = (a + b)/2\)
        2. Evaluate \(f(a)\), \(f(midpoint)\), \(f(b)\)
        3. Determine which subinterval contains the root:
            - If \(f(a)*f(midpoint) < 0\), root is in [a, midpoint]
            - Else, root is in [midpoint, b]
        4. Repeat until \(|b - a| < tolerance\)

        **How to Use This App:**
        - **Input Function**: Type your equation using `x` as the variable (e.g., `x^2 - 4`).
        - **Interval**: Enter `[a, b]` where the root lies.
        - **Tolerance**: Set the precision.
        - **Compute**: Click to calculate.
        """)

    st.divider()
    st.caption("Prepared by: ")
    st.caption("Artacho, Cristopher Ian")
    st.caption("Carado, John Manuel")
    st.caption("Tacuel, Allan Andrews")

if __name__ == "__main__":
    main()