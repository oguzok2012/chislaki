import math
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QTabWidget, QTextEdit, QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


# F1: 2x² - 2xy + 3y² + 5x - 4y - 8 = 0
# F2: 3x² + xy - 2y² + 4x - 2y + 1 = 0

def F1(x, y):
    return 2 * x * x - 2 * x * y + 3 * y * y + 5 * x - 4 * y - 8


def F2(x, y):
    return 3 * x * x + x * y - 2 * y * y + 4 * x - 2 * y + 1


# Частные производные для матрицы Якоби
def dF1_dx(x, y):
    return 4 * x - 2 * y + 5


def dF1_dy(x, y):
    return -2 * x + 6 * y - 4


def dF2_dx(x, y):
    return 6 * x + y + 4


def dF2_dy(x, y):
    return x - 4 * y - 2


class Matrix2x2:
    def __init__(self, a11, a12, a21, a22):
        self.a11 = a11
        self.a12 = a12
        self.a21 = a21
        self.a22 = a22

    def det(self):
        return self.a11 * self.a22 - self.a12 * self.a21

    def inverse(self):
        d = self.det()
        if abs(d) < 1e-10:
            return None
        return Matrix2x2(
            self.a22 / d, -self.a12 / d,
            -self.a21 / d, self.a11 / d
        )


class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def jacobian(x, y):
    return Matrix2x2(
        dF1_dx(x, y), dF1_dy(x, y),
        dF2_dx(x, y), dF2_dy(x, y)
    )


def mul_mat_vec(m, v):
    return Vector2(
        m.a11 * v.x + m.a12 * v.y,
        m.a21 * v.x + m.a22 * v.y
    )


def mul_mat_mat(a, b):
    return Matrix2x2(
        a.a11 * b.a11 + a.a12 * b.a21, a.a11 * b.a12 + a.a12 * b.a22,
        a.a21 * b.a11 + a.a22 * b.a21, a.a21 * b.a12 + a.a22 * b.a22
    )


def vec_norm(v):
    return math.sqrt(v.x * v.x + v.y * v.y)


class SystemResult:
    def __init__(self, x, y, iterations, converged, message):
        self.x = x
        self.y = y
        self.iterations = iterations
        self.converged = converged
        self.message = message


# Модифицированный метод Ньютона (Якобиан вычисляется однажды в x0)
def newton_system(x0, y0, eps, max_iter):
    x, y = x0, y0

    # Вычисляем матрицу Якоби только один раз в начальной точке
    J = jacobian(x0, y0)
    Jinv = J.inverse()
    if Jinv is None:
        return SystemResult(x0, y0, 0, False, "❌ Матрица Якоби вырожденная")

    for i in range(max_iter):
        f1, f2 = F1(x, y), F2(x, y)

        # Вычисляем dx = -J^-1 * F
        delta = mul_mat_vec(Jinv, Vector2(-f1, -f2))
        error = vec_norm(delta)

        x += delta.x
        y += delta.y

        if error < eps:
            return SystemResult(x, y, i + 1, True, "✅ Метод сходится - условие |Δx| < ε выполнено")

    return SystemResult(x, y, max_iter, False, "⚠️ Достигнуто максимальное количество итераций")


lambda_val = 0.05


# Итерационные функции φ
def phi1(x, y):
    return x - lambda_val * F1(x, y)


def phi2(x, y):
    return y - lambda_val * F2(x, y)


# Частные производные φ
def dphi1_dx(x, y):
    return 1 - lambda_val * dF1_dx(x, y)


def dphi1_dy(x, y):
    return -lambda_val * dF1_dy(x, y)


def dphi2_dx(x, y):
    return -lambda_val * dF2_dx(x, y)


def dphi2_dy(x, y):
    return 1 - lambda_val * dF2_dy(x, y)


# Метод простой итерации
def simple_iteration_system(x0, y0, eps, max_iter):
    x, y = x0, y0
    k = 0
    converged = False

    # Шаг 1: Оценка условия сходимости в окрестности начальной точки
    sup_norm = 0.0
    samples = 5
    radius = 0.2

    for i in range(samples):
        for j in range(samples):
            xi = x0 + (-radius + (2 * radius) * i / (samples - 1))
            yi = y0 + (-radius + (2 * radius) * j / (samples - 1))

            # Вычисляем матрицу Якоби в точке (xi, yi)
            J = Matrix2x2(
                dphi1_dx(xi, yi), dphi1_dy(xi, yi),
                dphi2_dx(xi, yi), dphi2_dy(xi, yi)
            )

            # Вычисляем норму матрицы (супремум по строкам)
            row1_norm = abs(J.a11) + abs(J.a12)
            row2_norm = abs(J.a21) + abs(J.a22)
            max_row_norm = max(row1_norm, row2_norm)

            if max_row_norm > sup_norm:
                sup_norm = max_row_norm

    # Формируем сообщение о сходимости
    if sup_norm < 1.0:
        msg = f"✅ sup||Φ'|| ~= {sup_norm:.3f} < 1 — условие сходимости выполнено"
    else:
        msg = f"⚠️ sup||Φ'|| ~= {sup_norm:.3f} >= 1 — сходимость не гарантирована"

    # Шаг 2: Итерационный процесс
    for k in range(max_iter):
        # Вычисляем следующее приближение
        x_new = phi1(x, y)
        y_new = phi2(x, y)

        # Проверяем сходимость по разнице между итерациями
        error_x = abs(x_new - x)
        error_y = abs(y_new - y)
        max_error = max(error_x, error_y)

        if max_error <= eps:
            converged = True
            k += 1
            msg += f"\n✅ Решение найдено за {k} итераций"
            break

        # Переход к следующей итерации
        x, y = x_new, y_new
        k += 1

    if not converged:
        msg += f"\n⚠️ Не сошёлся за {max_iter} итераций"

    return SystemResult(x, y, k, converged, msg)


# Метод Зейделя
def seidel_system(x0, y0, eps, max_iter):
    x, y = x0, y0
    k = 0
    converged = False

    J0 = jacobian(x0, y0)
    B = J0.inverse()
    if B is None:
        return SystemResult(x0, y0, 0, False, "❌ J(x0) вырожденная")

    omega = 0.05

    # Проверка условия сходимости
    sup_norm = 0.0
    samples = 5
    radius = 0.2

    for i in range(samples):
        for j in range(samples):
            xi = x0 + (-radius + (2 * radius) * i / (samples - 1))
            yi = y0 + (-radius + (2 * radius) * j / (samples - 1))

            Jx = jacobian(xi, yi)
            BJ = mul_mat_mat(B, Jx)
            J = Matrix2x2(
                1 - omega * BJ.a11, -omega * BJ.a12,
                -omega * BJ.a21, 1 - omega * BJ.a22
            )

            row1_norm = abs(J.a11) + abs(J.a12)
            row2_norm = abs(J.a21) + abs(J.a22)
            max_row_norm = max(row1_norm, row2_norm)

            if max_row_norm > sup_norm:
                sup_norm = max_row_norm

    if sup_norm < 1.0:
        msg = f"✅ sup||Φ'|| ~= {sup_norm:.3f} < 1 — условие сходимости выполнено"
    else:
        msg = f"⚠️ sup||Φ'|| ~= {sup_norm:.3f} >= 1 — сходимость не гарантирована"

    # Итерационный процесс
    for k in range(max_iter):
        # Обновляем x
        Fv = Vector2(F1(x, y), F2(x, y))
        B_F = mul_mat_vec(B, Fv)
        x_new = x - omega * B_F.x

        # Обновляем y с новым x
        Fv = Vector2(F1(x_new, y), F2(x_new, y))
        B_F = mul_mat_vec(B, Fv)
        y_new = y - omega * B_F.y

        # Проверяем сходимость по разнице между итерациями
        error_x = abs(x_new - x)
        error_y = abs(y_new - y)
        max_error = max(error_x, error_y)

        if max_error <= eps:
            converged = True
            k += 1
            msg += f"\n✅ Решение найдено за {k} итераций"
            break

        # Переход к следующей итерации
        x, y = x_new, y_new
        k += 1

    if not converged:
        msg += f"\n⚠️ Не сошёлся за {max_iter} итераций"

    return SystemResult(x, y, k, converged, msg)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.plot_system()

    def plot_system(self):
        ax = self.fig.add_subplot(111)

        x_min, x_max = -5, 5
        y_min, y_max = -5, 5

        # Создаем сетку для построения линий уровня
        x = np.linspace(x_min, x_max, 1000)

        # Рисуем F1 = 0 (синий)
        for x_val in x:
            # Решаем квадратное уравнение 3y² + (-2x - 4)y + (2x² + 5x - 8) = 0
            a = 3.0
            b = -2 * x_val - 4
            c = 2 * x_val * x_val + 5 * x_val - 8
            disc = b * b - 4 * a * c

            if disc >= 0:
                y1 = (-b + math.sqrt(disc)) / (2 * a)
                y2 = (-b - math.sqrt(disc)) / (2 * a)

                if y_min <= y1 <= y_max:
                    ax.plot(x_val, y1, 'b.', markersize=1, alpha=0.7)
                if y_min <= y2 <= y_max:
                    ax.plot(x_val, y2, 'b.', markersize=1, alpha=0.7)

        # Рисуем F2 = 0 (красный)
        for x_val in x:
            # Решаем квадратное уравнение -2y² + (x - 2)y + (3x² + 4x + 1) = 0
            a = -2.0
            b = x_val - 2
            c = 3 * x_val * x_val + 4 * x_val + 1
            disc = b * b - 4 * a * c

            if disc >= 0:
                y1 = (-b + math.sqrt(disc)) / (2 * a)
                y2 = (-b - math.sqrt(disc)) / (2 * a)

                if y_min <= y1 <= y_max:
                    ax.plot(x_val, y1, 'r.', markersize=1, alpha=0.7)
                if y_min <= y2 <= y_max:
                    ax.plot(x_val, y2, 'r.', markersize=1, alpha=0.7)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Система уравнений: F₁=0 (синий), F₂=0 (красный)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)

        # Добавляем легенду
        ax.plot([], [], 'b.', label='F₁ = 0: 2x² - 2xy + 3y² + 5x - 4y - 8 = 0')
        ax.plot([], [], 'r.', label='F₂ = 0: 3x² + xy - 2y² + 4x - 2y + 1 = 0')
        ax.legend(loc='upper right', fontsize=8)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система нелинейных уравнений")
        self.setGeometry(100, 100, 900, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.setup_plot_tab()
        self.setup_methods_tab()

    def setup_plot_tab(self):
        plot_tab = QWidget()
        layout = QVBoxLayout(plot_tab)

        info_label = QLabel("Графики: F₁=0 (синий), F₂=0 (красный)\n"
                            "F₁: 2x² - 2xy + 3y² + 5x - 4y - 8 = 0\n"
                            "F₂: 3x² + xy - 2y² + 4x - 2y + 1 = 0\n"
                            "Точки пересечения — решения системы")
        layout.addWidget(info_label)

        self.plot_canvas = PlotCanvas(self, width=7, height=5)
        layout.addWidget(self.plot_canvas)

        self.tabs.addTab(plot_tab, "График")

    def setup_methods_tab(self):
        methods_tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        # Общие настройки
        common_layout = QHBoxLayout()
        common_layout.addWidget(QLabel("Точность ε:"))
        self.eps_entry = QLineEdit()
        self.eps_entry.setText("0.0001")
        self.eps_entry.setMaximumWidth(100)
        common_layout.addWidget(self.eps_entry)

        self.clear_btn = QPushButton("Очистить результаты")
        self.clear_btn.clicked.connect(self.clear_results)
        common_layout.addWidget(self.clear_btn)
        common_layout.addStretch()

        layout.addLayout(common_layout)
        layout.addWidget(QLabel(""))  # Отступ

        # Методы в сетке
        methods_grid = QGridLayout()

        # Метод Ньютона
        self.setup_newton_method(methods_grid, 0)

        # Метод простой итерации
        self.setup_simple_iteration_method(methods_grid, 1)

        # Метод Зейделя
        self.setup_seidel_method(methods_grid, 2)

        layout.addLayout(methods_grid)
        layout.addStretch()

        scroll.setWidget(scroll_widget)
        methods_tab_layout = QVBoxLayout(methods_tab)
        methods_tab_layout.addWidget(scroll)

        self.tabs.addTab(methods_tab, "Методы")

    def setup_newton_method(self, grid, col):
        group = QVBoxLayout()
        group.addWidget(QLabel("Метод Ньютона"))

        x0_layout = QHBoxLayout()
        x0_layout.addWidget(QLabel("x₀:"))
        self.x0_newton = QLineEdit()
        self.x0_newton.setText("1.5")
        self.x0_newton.setMaximumWidth(100)
        x0_layout.addWidget(self.x0_newton)
        x0_layout.addStretch()
        group.addLayout(x0_layout)

        y0_layout = QHBoxLayout()
        y0_layout.addWidget(QLabel("y₀:"))
        self.y0_newton = QLineEdit()
        self.y0_newton.setText("2.0")
        self.y0_newton.setMaximumWidth(100)
        y0_layout.addWidget(self.y0_newton)
        y0_layout.addStretch()
        group.addLayout(y0_layout)

        solve_btn = QPushButton("Найти решение")
        self.result_newton = QTextEdit()
        self.result_newton.setMaximumHeight(120)
        self.result_newton.setReadOnly(True)

        solve_btn.clicked.connect(self.solve_newton)
        group.addWidget(solve_btn)
        group.addWidget(self.result_newton)

        widget = QWidget()
        widget.setLayout(group)
        grid.addWidget(widget, 0, col)

    def setup_simple_iteration_method(self, grid, col):
        group = QVBoxLayout()
        group.addWidget(QLabel("Метод простой итерации"))

        x0_layout = QHBoxLayout()
        x0_layout.addWidget(QLabel("x₀:"))
        self.x0_simple = QLineEdit()
        self.x0_simple.setText("0.0")
        self.x0_simple.setMaximumWidth(100)
        x0_layout.addWidget(self.x0_simple)
        x0_layout.addStretch()
        group.addLayout(x0_layout)

        y0_layout = QHBoxLayout()
        y0_layout.addWidget(QLabel("y₀:"))
        self.y0_simple = QLineEdit()
        self.y0_simple.setText("-1.0")
        self.y0_simple.setMaximumWidth(100)
        y0_layout.addWidget(self.y0_simple)
        y0_layout.addStretch()
        group.addLayout(y0_layout)

        solve_btn = QPushButton("Найти решение")
        self.result_simple = QTextEdit()
        self.result_simple.setMaximumHeight(120)
        self.result_simple.setReadOnly(True)

        solve_btn.clicked.connect(self.solve_simple)
        group.addWidget(solve_btn)
        group.addWidget(self.result_simple)

        widget = QWidget()
        widget.setLayout(group)
        grid.addWidget(widget, 0, col)

    def setup_seidel_method(self, grid, col):
        group = QVBoxLayout()
        group.addWidget(QLabel("Метод Зейделя"))

        x0_layout = QHBoxLayout()
        x0_layout.addWidget(QLabel("x₀:"))
        self.x0_seidel = QLineEdit()
        self.x0_seidel.setText("1.5")
        self.x0_seidel.setMaximumWidth(100)
        x0_layout.addWidget(self.x0_seidel)
        x0_layout.addStretch()
        group.addLayout(x0_layout)

        y0_layout = QHBoxLayout()
        y0_layout.addWidget(QLabel("y₀:"))
        self.y0_seidel = QLineEdit()
        self.y0_seidel.setText("2.0")
        self.y0_seidel.setMaximumWidth(100)
        y0_layout.addWidget(self.y0_seidel)
        y0_layout.addStretch()
        group.addLayout(y0_layout)

        solve_btn = QPushButton("Найти решение")
        self.result_seidel = QTextEdit()
        self.result_seidel.setMaximumHeight(120)
        self.result_seidel.setReadOnly(True)

        solve_btn.clicked.connect(self.solve_seidel)
        group.addWidget(solve_btn)
        group.addWidget(self.result_seidel)

        widget = QWidget()
        widget.setLayout(group)
        grid.addWidget(widget, 0, col)

    def solve_newton(self):
        try:
            eps = float(self.eps_entry.text())
            x0 = float(self.x0_newton.text())
            y0 = float(self.y0_newton.text())
            res = newton_system(x0, y0, eps, 1000)

            if not res.converged:
                text = f"❌ Не сошёлся\nx = {res.x:.6f}, y = {res.y:.6f}\nИтераций: {res.iterations}\n{res.message}"
            else:
                text = f"x = {res.x:.6f}, y = {res.y:.6f}\nИтераций: {res.iterations}\n{res.message}"

            self.result_newton.setText(text)
        except ValueError:
            self.result_newton.setText("❌ Ошибка ввода данных")

    def solve_simple(self):
        try:
            eps = float(self.eps_entry.text())
            x0 = float(self.x0_simple.text())
            y0 = float(self.y0_simple.text())
            res = simple_iteration_system(x0, y0, eps, 1000)

            if not res.converged:
                text = f"❌ Не сошёлся\nx = {res.x:.6f}, y = {res.y:.6f}\nИтераций: {res.iterations}\n{res.message}"
            else:
                text = f"x = {res.x:.6f}, y = {res.y:.6f}\nИтераций: {res.iterations}\n{res.message}"

            self.result_simple.setText(text)
        except ValueError:
            self.result_simple.setText("❌ Ошибка ввода данных")

    def solve_seidel(self):
        try:
            eps = float(self.eps_entry.text())
            x0 = float(self.x0_seidel.text())
            y0 = float(self.y0_seidel.text())
            res = seidel_system(x0, y0, eps, 1000)

            if not res.converged:
                text = f"❌ Не сошёлся\nx = {res.x:.6f}, y = {res.y:.6f}\nИтераций: {res.iterations}\n{res.message}"
            else:
                text = f"x = {res.x:.6f}, y = {res.y:.6f}\nИтераций: {res.iterations}\n{res.message}"

            self.result_seidel.setText(text)
        except ValueError:
            self.result_seidel.setText("❌ Ошибка ввода данных")

    def clear_results(self):
        self.result_newton.clear()
        self.result_simple.clear()
        self.result_seidel.clear()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()