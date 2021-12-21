import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from lib import *


wort_title = "Применение составной КФ Гаусса"
st.set_page_config(page_title=wort_title, page_icon=":eyeglasses:")
st.title(wort_title)
st.subheader("Сравнение составной с обычной")

st.sidebar.title("Параметры задачи")

f_str = st.sidebar.text_input("f(x) = ", "sin(x)")
f: sp.Expr = sp.sympify(f_str)
p_str = st.sidebar.text_input("p(x) = ", "sqrt(1-x)")
p: sp.Expr = sp.sympify(p_str)

begin = st.sidebar.number_input("A", value=0.0, step=0.01)
end = st.sidebar.number_input("B", value=1.0, step=0.01)

if begin >= end:
    st.sidebar.error("A должен быть меньше или равен B. Использую отрезок [0; 1].")
    begin, end = 0, 1

gauss_node_count = st.sidebar.number_input(
    "Количество узлов КФ Гаусса", value=3, step=1, min_value=1
)
nast_node_count = st.sidebar.number_input(
    "Количество узлов КФ НАСТ с весом p(x) на отрезке [A; B]",
    value=3,
    step=1,
    min_value=1,
)
partition_count = st.sidebar.number_input("Количество разбиений", value=10, step=1)
signs_after_comma = st.sidebar.slider("Количество знаков после запятой", 0, 15, 12, 1)
showing_expected_integral = st.sidebar.checkbox(
    "Показывать точное значение интеграла", True
)
showing_nodes_on_axis = st.sidebar.checkbox("Показывать узлы на оси", True)
showing_nodes_table = st.sidebar.checkbox("Показывать таблицу с узлами", True)
showing_error_minimizing = st.sidebar.checkbox("Показать уменьшение ошибки с увеличением числа узлов", True)
showing_error_comparison = st.sidebar.checkbox("Показать сравнение абсолютных ошибок", True)

nodes, coefficients = find_nodes_coefficients_gauss(gauss_node_count)
offset_nodes, offset_coefficients = linear_offset(
    nodes, coefficients, Section(begin, end)
)
if showing_nodes_table:
    st.table(
        pd.DataFrame({"Узлы": nodes, "Коэффициенты": coefficients}).style.format(
            "{:." + str(signs_after_comma) + "}"
        )
    )

if showing_nodes_on_axis:
    show_nodes_on_axis(nodes)

if showing_nodes_table:
    st.table(
        pd.DataFrame(
            {
                "Сдвинутые узлы": offset_nodes,
                "Сдвинутые коэффициенты": offset_coefficients,
            }
        ).style.format("{:." + str(signs_after_comma) + "}")
    )

if showing_nodes_on_axis:
    show_nodes_on_axis(offset_nodes)

expected = float(calculate_expected_integral(f_str, p_str, begin, end))
actual_without_partition = calculate_gauss_integral(
    f * p, offset_nodes, offset_coefficients
)
h = (end - begin) / partition_count
st.markdown(rf"h = {h:.{signs_after_comma}}")

actual_with_partition = calculate_gauss_integral_with_partition(
    f * p, nodes, coefficients, partition_count, begin, end
)

st.markdown(
    rf"Приближенное значение интеграла (КФ Гаусса) $I_{{Gauss}} = {actual_without_partition:.{signs_after_comma}}$"
)
st.markdown(
    rf"Приблеженное значение интеграла (составная КФ Гаусса) $I_{{CG}} = {actual_with_partition:.{signs_after_comma}}$"
)

if showing_expected_integral:
    st.markdown(
        rf"Точное значение интеграла (подсчитано с помощью **sympy**) $I_{{expected}} = {expected:.{signs_after_comma}}$"
    )
    gauss_error = abs(actual_without_partition - expected)
    st.markdown(
        rf"$|I_{{expected}} - I_{{Gauss}}| = {gauss_error:.{signs_after_comma}}$"
    )
    cg_error = abs(actual_with_partition - expected)
    st.markdown(rf"$|I_{{expected}} - I_{{CG}}| = {cg_error:.{signs_after_comma}}$")

st.subheader("Сходимость при увеличении количества разбиений")
if show_error_minimizing:
    st.plotly_chart(show_error_minimizing(partition_count, f_str, p_str, nodes, coefficients, expected, begin, end))

st.subheader("КФ типа Гаусса с весом p(x) на [A; B]")

moments = calculate_moments(p_str, begin, end, nast_node_count)

st.table(
    pd.DataFrame({"Моменты": moments}).style.format(
        "{:." + str(signs_after_comma) + "}"
    )
)
left_part = np.array(
    [
        [moments[j] for j in np.arange(nast_node_count - 1 + i, i - 1, -1)]
        for i in range(nast_node_count)
    ]
)

right_part = np.array([-moments[nast_node_count + i] for i in range(nast_node_count)])
polynom_coeffs = [x for x in reversed(np.linalg.solve(left_part, right_part))]
polynom_coeffs.append(1)
polynomial: sp.Expr = sum(
    sp.Symbol("x") ** power * coef for power, coef in enumerate(polynom_coeffs)
)
st.markdown(rf"Ортогональный полином для веса p(x) на [A;B]")
st.markdown(rf"$\omega(x) = {sp.latex(polynomial)}$")

roots = list(map(float, sp.roots(polynomial).keys()))

left_part = np.array(
    [[roots[j] ** i for j in range(nast_node_count)] for i in range(nast_node_count)]
)

nast_coefficients = np.linalg.solve(left_part, moments[:nast_node_count])

if showing_nodes_table:
    st.table(
        pd.DataFrame({"Узлы": roots, "Коэффициенты": nast_coefficients}).style.format(
            "{:." + str(signs_after_comma) + "}"
        )
    )
if showing_nodes_on_axis:
    show_nodes_on_axis(roots)

nast_integral = sum(
    map(
        lambda coef_node: coef_node[0] * float(f.subs({"x": coef_node[1]})),
        zip(nast_coefficients, roots),
    )
)
st.markdown(
    rf"Приближенное значение интеграла (КФ с весом p(x) на [A; B]) $I_{{p}} = {nast_integral:.{signs_after_comma}}$"
)

if showing_expected_integral:
    st.markdown(
        rf"Точное значение интеграла (подсчитано с помощью **sympy**) $I_{{expected}} = {expected:.{signs_after_comma}}$"
    )
    nast_error = abs(nast_integral - expected)
    st.markdown(rf"$|I_{{expected}} - I_{{p}}| = {nast_error:.{signs_after_comma}}$")

if showing_error_comparison:
    st.subheader("Сравнение абсолютных погрешностей (авторства Егора Порсева)")
    fig = plt.figure()
    sns.barplot(
        data=pd.DataFrame(
            {
                "Абсолютная погрешность": [gauss_error, cg_error, nast_error],
                "Тип КФ": [
                    "КФ Гаусса",
                    "Составная КФ Гаусса",
                    "НАСТ КФ с весом p(x) на [A;B]",
                ],
            }
        ),
        x="Абсолютная погрешность",
        y="Тип КФ",
        orient="h",
    ).set_xscale("log")
    st.pyplot(fig)
