import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from numbers import Real
import altair as alt
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
from bisection_method import find_root

# Sample function
def f(x):
    return x**3 - 4*x - 9

# HELPER FUNCTIONS
def get_ba_and_within(df: pd.DataFrame, tolerance: Real) -> pd.DataFrame:
    """ Calculates and saves the |b-a| & checks tolerance if < |b-a| """
    # Get |b-a| & Within bool
    additional_data: List[Dict[str, Real]] = []
    for i in range(len(df)):
        df_current_index = df.iloc[i]
        val = (df_current_index['b'] - df_current_index['a'])
        # within = val < tolerance
        additional_data.append({
            '|b-a|': val,
            # 'within': within
        })
    return pd.DataFrame(additional_data)

def plot_function(a: Real, b: Real, root: Real) -> None:

    x_min = min(a, b) - 1.0
    x_max = max(a, b) + 1.0
    
    # Generate 100 points for a smooth line
    x_range = np.linspace(x_min, x_max, 100)
    y_range = f(x_range)
    
    source = pd.DataFrame({
        'x': x_range,
        'f(x)': y_range
    })

    # B. Create Data for the Root Marker
    root_data = pd.DataFrame({
        'x': [root],
        'f(x)': [f(root)] # Should be very close to 0
    })

    # C. Build Altair Layers
    # 1. The Curve
    line_chart = alt.Chart(source).mark_line().encode(
        x=alt.X('x', title='x'),
        y=alt.Y('f(x)', title='f(x)')
    )

    # 2. The Zero Line (Axis) - Helps visually locate the root
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3,3]).encode(y='y')

    # 3. The Root Marker (Red Dot)
    point_chart = alt.Chart(root_data).mark_circle(color='red', size=100).encode(
        x='x',
        y='f(x)',
        tooltip=[alt.Tooltip('x', format='.5f'), alt.Tooltip('f(x)', format='.5f')]
    )

    # D. Combine and Display
    # st.line_chart cannot layer these, so we use st.altair_chart
    st.altair_chart((line_chart + zero_line + point_chart).interactive(), use_container_width=True)


def main():
    # STREAMLIT UI
    st.title("🔢 Bisection Method Root Finder")
    st.write("Enter the function and interval parameters below.")

    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Interval (a)", value=2.0)
        b = st.number_input("Interval (b)", value=3.0)
    with col2:
        # start_x = st.number_input("x (starting value)", value=0.0)  # Required by instructions
        tolerance = st.number_input("Tolerance ℰ", value=0.0001, format="%.5f", step=0.0001)
        compute_root = st.button("Compute Root")

    if compute_root:
        root, logs = find_root((a,b), tolerance, f, False, True)
        
        df = pd.DataFrame(logs)
        ba_within = get_ba_and_within(df.copy(), tolerance)
        df = pd.concat([df, ba_within], axis=1)

        df.drop(labels=['iteration'], axis=1, inplace=True)    
        df.index += 1
        st.success(f"Root found: {root}")

        col3, col4 = st.columns(2)
        with col3:
            st.dataframe(df)
        with col4:
            plot_function(a,b,root)

    st.divider()
    st.caption("Prepared by: ")
    st.caption("Artacho, Cristopher Ian")
    st.caption("Carado, John Manuel")
    st.caption("Tacuel, Allan Andrews")
        
        

if __name__ == "__main__":
    main()