import math
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTabWidget, QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class MethodResult:
    def __init__(self, root, iterations, converged, message):
        self.root = root
        self.iterations = iterations
        self.converged = converged
        self.message = message


# Функция f(x) = √(x² + 2) + 2sin(x) - 3 = 0
def f(x):
    return math.sqrt(x ** 2 + 2) + 2 * math.sin(x) - 3


# Производная f'(x) = x/√(x² + 2) + 2cos(x)
def df(x):
    return x / math.sqrt(x ** 2 + 2) + 2 * math.cos(x)


# Метод простой итерации с lambda: φ(x) = x - λf(x). λ - коэффициент релаксации
def phi(x, lambda_val):
    return x - lambda_val * f(x)


def dphi(x, lambda_val):
    return 1 - lambda_val * df(x)


# Вторая производная f''(x) = 2/(x² + 2)^(3/2) - 2sin(x)
def ddf(x):
    return 2 / math.pow(x ** 2 + 2, 1.5) - 2 * math.sin(x)


# Метод простой итерации
def simple_iteration_interval(a, b, eps, max_iter):
    fa, fb = f(a), f(b)

    if fa * fb >= 0:
        return MethodResult(0, 0, False, f"❌ f(a)*f(b)={fa * fb:.6f} >= 0 — функция не меняет знак на [a,b]")

    # Определяем знак лямбды по положению корня
    # Если f(a) < 0 и f(b) > 0, корень между a и b, двигаемся от a к b -> lambda = +0.1
    # Если f(a) > 0 и f(b) < 0, корень между a и b, двигаемся от b к a -> lambda = -0.1
    if fa < 0 and fb > 0:
        lambda_val = 0.1  # двигаемся вправо от a к b
        x = a
    else:
        lambda_val = -0.1  # двигаемся влево от b к a
        x = b

    for i in range(max_iter):
        x_new = phi(x, lambda_val)

        if math.isnan(x_new):
            return MethodResult(0, i, False, f"φ(x) неопределена при x={x:.6f}")
        if x_new < a or x_new > b:
            return MethodResult(0, i, False, f"Итерация вышла за границы [{a:.6f}, {b:.6f}]")

        if abs(x_new - x) < eps:
            return MethodResult(x_new, i + 1, True, f"Метод сошёлся")

        x = x_new

    return MethodResult(x, max_iter, False, f"Не сошёлся за {max_iter} итераций, λ={lambda_val}")


# Метод Ньютона
def newton_interval(a, b, eps, max_iter):
    fa, fb = f(a), f(b)

    if fa * fb >= 0:
        return MethodResult(0, 0, False, f"❌ f(a)*f(b)={fa * fb:.6f} >= 0 — функция не меняет знак на [a,b]")

    dfa = df(a)
    d2fa = ddf(a)
    cond_a = abs(fa * d2fa) < dfa * dfa

    dfb = df(b)
    d2fb = ddf(b)
    cond_b = abs(fb * d2fb) < dfb * dfb

    if cond_a:
        x = a
    elif cond_b:
        x = b
    else:
        return MethodResult(0, 0, False, "❌ Условие сходимости не выполняется ни в точке a, ни в точке b")

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-10:
            return MethodResult(x, i, False, "f'(x)≈0 — деление невозможно")
        x_new = x - fx / dfx
        if abs(x_new - x) < eps:
            return MethodResult(x_new, i + 1, True, "Метод сошёлся")
        x = x_new
    return MethodResult(x, max_iter, False, "Не сошёлся за max_iter")


# Метод секущих
def secant(x0, x1, eps, max_iter):
    fx0 = f(x0)
    dfx0 = df(x0)
    d2fx0 = ddf(x0)
    cond0 = abs(fx0 * d2fx0) < dfx0 * dfx0

    fx1 = f(x1)
    dfx1 = df(x1)
    d2fx1 = ddf(x1)
    cond1 = abs(fx1 * d2fx1) < dfx1 * dfx1

    if not cond0 and not cond1:
        return MethodResult(0, 0, False, "❌ Условие сходимости не выполняется ни в точке x₀, ни в точке x₁")

    for i in range(max_iter):
        if abs(fx1 - fx0) < 1e-10:
            return MethodResult(x1, i, False, "f(x₁) ≈ f(x₀)")
        x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(x_new - x1) < eps:
            return MethodResult(x_new, i + 1, True, "Метод сошёлся")
        x0, x1 = x1, x_new
        fx0, fx1 = fx1, f(x1)
    return MethodResult(x1, max_iter, False, "Не сошелся")


# Метод хорд
def chord(a, b, eps, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        return MethodResult(0, 0, False, f"f(a)*f(b) = {fa * fb:.6f} >= 0 — нет гарантии наличия корня на [a,b]")

    dfa = df(a)
    d2fa = ddf(a)
    cond_a = abs(fa * d2fa) < dfa * dfa

    dfb = df(b)
    d2fb = ddf(b)
    cond_b = abs(fb * d2fb) < dfb * dfb

    if cond_a:
        x = a
    elif cond_b:
        x = b
    else:
        return MethodResult(0, 0, False, "❌ Условие сходимости не выполняется ни в точке a, ни в точке b")

    for i in range(max_iter):
        fx = f(x)
        x_new = x - fx * (x - b) / (fx - fb)
        if abs(x_new - x) < eps:
            return MethodResult(x_new, i + 1, True, "Метод сошёлся")
        x = x_new
    return MethodResult(x, max_iter, False, "Не сошелся за max_iter")


# Метод дихотомии
def bisection(a, b, eps, max_iter):
    fa, fb = f(a), f(b)

    if fa * fb >= 0:
        return MethodResult(0, 0, False, f"f(a)*f(b) = {fa * fb:.6f} >= 0 — нет гарантии корня на [a,b]")

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        if (b - a) < eps:
            return MethodResult(c, i + 1, True, "Метод сошёлся")
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return MethodResult((a + b) / 2, max_iter, False, "Не сошелся за max_iter")


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.plot_function()

    def plot_function(self):
        ax = self.fig.add_subplot(111)

        x_min, x_max = -5, 5
        y_min, y_max = -4, 4

        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=1)
        ax.axvline(x=0, color='black', linewidth=1)

        ax.arrow(x_max, 0, 0.3, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
        ax.arrow(0, y_max, 0, 0.3, head_width=0.1, head_length=0.2, fc='black', ec='black')

        ax.text(x_max + 0.2, -0.3, 'x', fontsize=12)
        #ax.text(0.1, y_max + 0.2, 'f(x)', fontsize=12)

        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = [f(x) for x in x_vals]
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = (x^2 + 2)^0.5 + 2sin(x) - 3')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        #ax.set_title('f(x) = (x² + 2) + 2sin(x) - 3')


class MethodWidget(QWidget):
    def __init__(self, method_name, default_a, default_b, solve_func, method_num):
        super().__init__()
        self.method_name = method_name
        self.solve_func = solve_func
        self.method_num = method_num
        self.init_ui(default_a, default_b)

    def init_ui(self, default_a, default_b):
        layout = QVBoxLayout()

        title = QLabel(f"<b>{self.method_name}</b>")
        layout.addWidget(title)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("a:"))
        self.a_entry = QLineEdit()
        self.a_entry.setText(default_a)
        input_layout.addWidget(self.a_entry)

        input_layout.addWidget(QLabel("b:"))
        self.b_entry = QLineEdit()
        self.b_entry.setText(default_b)
        input_layout.addWidget(self.b_entry)

        layout.addLayout(input_layout)

        self.solve_btn = QPushButton("Найти решение")
        self.solve_btn.clicked.connect(self.solve)
        layout.addWidget(self.solve_btn)

        self.result_label = QTextEdit()
        self.result_label.setMaximumHeight(100)
        self.result_label.setReadOnly(True)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def solve(self):
        try:
            eps = float(main_window.eps_entry.text())
            a = float(self.a_entry.text())
            b = float(self.b_entry.text())

            res = self.solve_func(a, b, eps, 1000)

            if hasattr(res, 'root') and (res.root < a or res.root > b):
                self.result_label.setText(f"Корень вне заданных границ: x = {res.root:.6f}")
            else:
                self.result_label.setText(f"Корень: x = {res.root:.6f}\nИтераций: {res.iterations}\n{res.message}")

        except ValueError as e:
            self.result_label.setText(f"Ошибка: Некорректный ввод данных\n{str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Численные методы решения уравнений: √(x² + 2) + 2sin(x) - 3 = 0")
        self.setGeometry(100, 100, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Точность
        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("Точность ε:"))
        self.eps_entry = QLineEdit()
        self.eps_entry.setText("0.0001")
        eps_layout.addWidget(self.eps_entry)
        eps_layout.addStretch()

        main_layout.addLayout(eps_layout)

        # Кнопка очистки
        self.clear_btn = QPushButton("Очистить результаты")
        self.clear_btn.clicked.connect(self.clear_results)
        main_layout.addWidget(self.clear_btn)

        # Вкладки
        self.tabs = QTabWidget()

        # Вкладка с графиком
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        self.plot_canvas = PlotCanvas(self, width=8, height=6)
        plot_layout.addWidget(self.plot_canvas)
        # info_label = QLabel("Найдите пересечения с осью X для определения начальных приближений")
        # plot_layout.addWidget(info_label)

        # Вкладка с методами
        methods_tab = QWidget()
        methods_layout = QVBoxLayout(methods_tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Создаем методы с начальными приближениями для уравнения √(x² + 2) + 2sin(x) - 3 = 0
        # Уравнение имеет корни примерно в интервалах: [-2, -1], [1, 2]
        self.method_widgets = []

        method1 = MethodWidget("Метод простой итерации", "-4.0", "-2.0",
                               lambda a, b, eps, max_iter: simple_iteration_interval(a, b, eps, max_iter), 1)
        self.method_widgets.append(method1)

        method2 = MethodWidget("Метод Ньютона", "-4.0", "-2.0",
                               lambda a, b, eps, max_iter: newton_interval(a, b, eps, max_iter), 2)
        self.method_widgets.append(method2)

        method3 = MethodWidget("Метод секущих", "-4.0", "-2.0",
                               lambda a, b, eps, max_iter: secant(a, b, eps, max_iter), 3)
        self.method_widgets.append(method3)

        method4 = MethodWidget("Метод хорд", "-4.0", "-2.0",
                               lambda a, b, eps, max_iter: chord(a, b, eps, max_iter), 4)
        self.method_widgets.append(method4)

        method5 = MethodWidget("Метод дихотомии", "-4.0", "-2.0",
                               lambda a, b, eps, max_iter: bisection(a, b, eps, max_iter), 5)
        self.method_widgets.append(method5)

        for widget in self.method_widgets:
            scroll_layout.addWidget(widget)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        methods_layout.addWidget(scroll)

        self.tabs.addTab(plot_tab, "График")
        self.tabs.addTab(methods_tab, "Методы")

        main_layout.addWidget(self.tabs)

    def clear_results(self):
        for widget in self.method_widgets:
            widget.result_label.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())