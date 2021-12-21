import sympy as sp
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

from collections import namedtuple

Section = namedtuple("Section", ["begin", "end"])


def get_sections_with_root(f, domain: Section, partition_count):
    h = (domain.end - domain.begin) / partition_count
    points = [domain.begin + i * h for i in range(partition_count + 1)]

    section_with_roots = []

    for begin, end in zip(points[:-1], points[1:]):
        if f(end) * f(begin) <= 0:
            section_with_roots.append(Section(begin, end))

    return section_with_roots


def apply_secant_method(f, section: Section, eps: float):
    x_0 = section.begin
    x_1 = section.end

    def new_solution(x_1, x_0):
        return x_1 - f(x_1) / (f(x_1) - f(x_0)) * (x_1 - x_0)

    x_2 = new_solution(x_1, x_0)
    while abs(x_2 - x_1) >= eps:
        x_0 = x_1
        x_1 = x_2
        x_2 = new_solution(x_1, x_0)

    return x_2


def get_roots(f, domain: Section, partition_count=100000, eps=1e-12):
    return list(
        set(
            apply_secant_method(f, section, eps)
            for section in get_sections_with_root(f, domain, partition_count)
        )
    )


def get_legendre_polynomials(max_degree):
    class LegendrePolynom:
        def __init__(self, prev_poly, prev_prev_poly, degree):
            self.prev_poly = prev_poly
            self.prev_prev_poly = prev_prev_poly
            self.deegre = degree

        def __call__(self, x):
            return (2 * self.deegre - 1) / self.deegre * self.prev_poly(x) * x - (
                self.deegre - 1
            ) / self.deegre * self.prev_prev_poly(x)

    polynomials = [lambda _: 1, lambda x: x]
    if max_degree <= 1:
        return polynomials[: max_degree + 1]

    for degree in range(2, max_degree + 1):
        polynomials.append(LegendrePolynom(polynomials[-1], polynomials[-2], degree))

    return polynomials


@st.cache(ttl=3600, max_entries=10)
def find_nodes_coefficients_gauss(max_degree, eps=1e-12):
    polynomials = get_legendre_polynomials(max_degree)
    node_count = len(polynomials) - 1

    polynom = polynomials[node_count]
    roots = get_roots(polynom, Section(-1, 1), eps=eps)

    assert len(roots) == node_count

    coefficients = [
        (
            2
            * (1 - roots[k - 1] ** 2)
            / ((node_count ** 2) * (polynomials[node_count - 1](roots[k - 1]) ** 2))
        )
        for k in range(1, node_count + 1)
    ]

    return roots, coefficients


def linear_offset(nodes, coefficients, domain):
    q = (domain.end - domain.begin) / 2
    return list(map(lambda node: domain.begin + q * (node + 1), nodes)), list(
        map(lambda coef: q * coef, coefficients)
    )


def calculate_gauss_integral(f, nodes, coefficients):
    return sum(
        float(f.evalf(subs={"x": node})) * coef
        for node, coef in zip(nodes, coefficients)
    )


@st.cache(ttl=3600, max_entries=10)
def calculate_expected_integral(f_str, p_str, begin, end):
    f: sp.Expr = sp.sympify(f_str)
    p: sp.Expr = sp.sympify(p_str)
    return float(sp.integrate(f * p, ("x", begin, end)))


def show_nodes_on_axis(nodes):
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


def calculate_gauss_integral_with_partition(
    f, nodes, coefficients, partition_count, begin, end
):
    h = (end - begin) / partition_count
    actual_with_partition = 0
    for i in range(1, partition_count + 1):
        new_section = Section(begin + (i - 1) * h, begin + i * h)
        partition_nodes, partition_coefficients = linear_offset(
            nodes, coefficients, new_section
        )
        actual_with_partition += calculate_gauss_integral(
            f, partition_nodes, partition_coefficients
        )
    return actual_with_partition


@st.cache(ttl=3600, max_entries=10)
def calculate_moments(p_str, begin, end, nodes_count):
    p: sp.Expr = sp.sympify(p_str)
    return [
        float(sp.integrate(p * (sp.sympify("x") ** i), ("x", begin, end)))
        for i in range(2 * int(nodes_count))
    ]


@st.cache(ttl=3600, max_entries=10)
def show_error_minimizing(
    partition_count, f_str, p_str, nodes, coefficients, expected, begin, end
):
    errors = []
    f = sp.sympify(f_str)
    p = sp.sympify(p_str)
    for i in range(1, partition_count + 1):
        errors.append(
            float(
                abs(
                    calculate_gauss_integral_with_partition(
                        f * p, nodes, coefficients, i, begin, end
                    )
                    - expected
                )
            )
        )

    return px.line(
        pd.DataFrame(
            {
                "Absolute error": errors,
                "Partition count": range(1, partition_count + 1),
            }
        ),
        x="Partition count",
        y="Absolute error",
    )
