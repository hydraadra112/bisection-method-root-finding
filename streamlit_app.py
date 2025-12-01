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
    
    # 4. Marker for initial 'a' (start of interval)
    # Using a dashed green line to mark the lower bound 'a'
    a_marker = alt.Chart(pd.DataFrame({'a': [a]})).mark_rule(
        color='green', 
        strokeDash=[5, 5], 
        size=1.5
    ).encode(
        x='a',
        tooltip=[alt.Tooltip('a', title='a (Interval Start)', format='.5f')]
    )

    # 5. Marker for initial 'b' (end of interval)
    # Using a dashed blue line to mark the upper bound 'b'
    b_marker = alt.Chart(pd.DataFrame({'b': [b]})).mark_rule(
        color='blue', 
        strokeDash=[5, 5], 
        size=1.5
    ).encode(
        x='b',
        tooltip=[alt.Tooltip('b', title='b (Interval End)', format='.5f')]
    )

    # D. Combine and Display - Add the new markers
    st.altair_chart((line_chart + zero_line + point_chart + a_marker + b_marker).interactive(), width='stretch')


def main():
    # STREAMLIT UI
    st.title("🔢 Bisection Method Root Finder")
    
    # Create two tabs
    tab1, tab2 = st.tabs(["Root Finder", "About"])
    
    with tab1:
        st.write("Enter the function and interval parameters below.")
        
        function_input = st.text_input("Enter function f(x):", value="x**3 - 4*x - 9")
        st.caption("We recommend using Python syntax when defining your equations. Trig functions are supported: `sin(x)`, `cos(x)`, `exp(x)`.")
        #st.caption("We recommend using Python syntax when defining your equations.")
        
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
        st.header("What's this about?")
        st.write("A Streamlit application for root finding using the bisection method. " \
        "This is one of the core concepts that was taught to us by our instructor, **Mr. John Alexis Gemino**, " \
        "for our course **CCS 239 - Optimization Theory & Applications**.", unsafe_allow_html=True)

        ref = r'https://flexiple.com/python/bisection-method-python'

        st.write("This application will serve as the output of our final project for the CCS 239 course, where we are " \
        "tasked to choose a single method that was taught in this course, and implement a GUI for usability. " \
        "Our group chose the root finding using the [bisection method](%s). We do not have a particular reason in mind when we chose this method, " \
        "except that we can intuitively say that it is easy to implement without the use of LLM's (or AI chatbots)." % ref)

        st.write("As for the task execution, workload has been delegated throughout the trio.")
        st.write("**Artacho, Cristopher Ian** was in charge of implementing the basic functionalities and UI of the streamlit application.")
        st.write("**Carado, John Manuel** was in charge of implementing the root finding solver, and assisted in implementing the streamlit application.")
        st.write("**Tacuel, Allan Andrews** was in charge of the documentation to be submitted.")

        st.write("The solver, `bisection_method.py` is designed to be a standalone Python module, which means you can copy it onto your device, " \
        "and use it for personal reasons. You can implement an even more complex UI if you want to, the solver can still be easily integrated. " \
        "Below is a sample code for the bisection root finder solver in Python. If you seek additional details, simply download the solver, and see the docstring documentation.")
        code = '''
from bisection_method import find_root
# Sample function
def f(x):
    return x**3 - 4*x - 9

root, logs = find_root(
            interval=(2, 3),
            tolerance=0.0001,
            f=f,
            print_output=True,
            get_logs=True)'''
        st.code(code, language='python')

        st.write("If you wish to have a customized Python package of root finders or optimizations, I (Manuel) would be very willing to help you out " \
        "for the next batch of Optimization course takers. I am doing this for the sake of open source, to contribute back to the world.")

    st.divider()
    st.caption("Prepared by: ")
    st.caption("Artacho, Cristopher Ian")
    st.caption("Carado, John Manuel")
    st.caption("Tacuel, Allan Andrews")

if __name__ == "__main__":
    main()