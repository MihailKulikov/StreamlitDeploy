import streamlit as st
import sympy as sp
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from lib import get_meler_nodes, get_meler_coeffs


@st.cache(ttl=3600, max_entries=10)
def calculate_expected_integral(f_str, begin, end):
    f: sp.Expr = sp.sympify(f_str)
    return float(sp.integrate((f / sp.sympify("sqrt(1 - x ** 2)")), ("x", begin, end)))


@st.cache(ttl=3600, max_entries=10)
def calculate_actual_integral(f, nodes, coefficients):
    return sum(
        map(
            lambda value: value[0] * value[1],
            zip(coefficients, (f.evalf(subs={"x": node}) for node in nodes)),
        )
    )


wort_title = "Вычисление интегралов при помощи КФ Мелера"
begin = -1
end = 1
st.set_page_config(page_title=wort_title, page_icon=":eyeglasses:")

st.title(wort_title)

st.sidebar.title("Параметры задачи")
f_str = st.sidebar.text_input("f(x) = ", "cos(x) * (1 + x ** 2)")
f: sp.Expr = sp.sympify(f_str)

st.sidebar.text("Область интегрирования:")
st.sidebar.latex("A = -1, B = 1")
max_nodes_count = st.sidebar.number_input(
    "Максимальное количество узлов",
    1,
    value=8,
    step=1,
)
selected_node_counts = sorted(
    st.sidebar.multiselect(
        "Количество узлов", range(1, max_nodes_count + 1), default=[3, 5, 8]
    )
)
signs_after_comma = st.sidebar.slider("Количество знаков после запятой", 0, 15, 12, 1)
showing_expected_integral = st.sidebar.checkbox(
    "Показывать точное значение интеграла", True
)
showing_nodes_on_axis = st.sidebar.checkbox("Показывать узлы на оси", True)
showing_nodes_table = st.sidebar.checkbox("Показывать таблицу с узлами", True)

expected = calculate_expected_integral(f_str, begin, end)
errors = {}

for node_count in selected_node_counts:
    nodes = get_meler_nodes(node_count)
    coefficients = get_meler_coeffs(node_count)

    st.subheader(f"Количество узлов: {node_count}")
    if showing_nodes_table:
        st.table(
            pd.DataFrame({"Узлы": nodes, "Коэффициенты": coefficients}).style.format(
                "{:." + str(signs_after_comma) + "}"
            )
        )

    actual = calculate_actual_integral(f, nodes, coefficients)

    if showing_nodes_on_axis:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=nodes,
                y=[0 for _ in nodes],
                marker={"size": 7},
                line={"color": "red", "width": 3},
                name="Узлы",
                mode="markers",
                hovertemplate="%{x}",
            )
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(
            showgrid=False,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=2,
            showticklabels=False,
        )
        fig.update_layout(height=70, plot_bgcolor="white", margin=dict(b=0, t=60))
        st.write(fig)

    st.markdown(
        rf"""Приближенное значение интеграла $I_{{actual}} = {actual:.{signs_after_comma}}$"""
    )
    absolute_error = float(abs(actual - expected))
    errors[node_count] = absolute_error
    if showing_expected_integral:
        st.markdown(
            rf"""Точное значиение интеграла (подсчитано с помощью **sympy**) $I_{{expected}} = {expected:.{signs_after_comma}}$"""
        )
        st.markdown(
            rf"""$|I_{{expected}} - I_{{actual}}| = {absolute_error:.{signs_after_comma}}$"""
        )

st.subheader("Сравнение абсолютных погрешностей")
fig = plt.figure()
sns.barplot(
    data=pd.DataFrame(
        {
            "Absolute error": [
                errors[node_count] for node_count in selected_node_counts
            ],
            "Node count": selected_node_counts,
        }
    ),
    x="Absolute error",
    y="Node count",
    orient="h",
).set_xscale("log")
st.pyplot(fig)
