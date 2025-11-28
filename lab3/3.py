import sys


# Исходные данные
XI = [-6.0, -5.44, -4.67, -4.04, -3.2, -2.29, -1.66, -0.89, -0.05, 0.51, 1.0]
YI = [-1.451, 2.523, 3.536, 4.914, 4.978, 6.168, 3.769, 2.246, 1.915, -0.021, -2.945]
X_STAR = -2.863


class SplineCoeffs:
    """Коэффициенты кубического сплайна на одном отрезке"""

    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


def solve_tridiagonal(a, b, c, d):
    """
    Решение трехдиагональной системы уравнений методом прогонки (Томаса)

    Параметры:
        a - нижняя диагональ (a[0] не используется)
        b - главная диагональ
        c - верхняя диагональ (c[n-1] не используется)
        d - правая часть

    Возвращает:
        x - решение системы
    """
    n = len(d)
    if n == 0:
        return []

    cp = [0.0] * n
    dp = [0.0] * n
    x = [0.0] * n

    # Прямой ход
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

    # Обратный ход
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def build_natural_cubic_spline(X, Y):
    """
    Построение естественного кубического сплайна

    Параметры:
        X - массив узлов интерполяции
        Y - массив значений функции в узлах

    Возвращает:
        splines - список коэффициентов SplineCoeffs для каждого отрезка
    """
    n = len(X) - 1
    if n < 1:
        raise ValueError("Недостаточно точек для построения сплайна")

    # Вычисление длин отрезков
    h = [X[i + 1] - X[i] for i in range(n)]

    # Проверка корректности данных
    for i in range(n):
        if h[i] <= 0:
            raise ValueError("Узлы должны быть упорядочены по возрастанию")
        if h[i] == 0:
            raise ValueError("Узлы не должны совпадать")

    # Вычисление разностей для правой части системы
    alpha = [0.0] * n
    for i in range(1, n):
        alpha[i] = 3 * ((Y[i + 1] - Y[i]) / h[i] - (Y[i] - Y[i - 1]) / h[i - 1])

    # Для естественного сплайна m0 = mn = 0
    # Система решается только для m1...m_{n-1}
    size = n - 1

    if size == 0:
        # Особый случай: только один отрезок
        splines = []
        a_i = Y[0]
        b_i = (Y[1] - Y[0]) / h[0]
        splines.append(SplineCoeffs(a_i, b_i, 0.0, 0.0))
        return splines

    # Подготовка матрицы для метода прогонки
    a_vec = [0.0] * size  # нижняя диагональ
    b_vec = [0.0] * size  # главная диагональ
    c_vec = [0.0] * size  # верхняя диагональ
    d_vec = [0.0] * size  # правая часть

    # Первое уравнение
    b_vec[0] = 2 * (h[0] + h[1])
    c_vec[0] = h[1]
    d_vec[0] = alpha[1]

    # Внутренние уравнения
    for i in range(1, size - 1):
        a_vec[i] = h[i]
        b_vec[i] = 2 * (h[i] + h[i + 1])
        c_vec[i] = h[i + 1]
        d_vec[i] = alpha[i + 1]

    # Последнее уравнение (если size > 1)
    if size > 1:
        a_vec[size - 1] = h[n - 2]
        b_vec[size - 1] = 2 * (h[n - 2] + h[n - 1])
        d_vec[size - 1] = alpha[n - 1]

    # Решение системы для вторых производных в узлах
    m_inner = solve_tridiagonal(a_vec, b_vec, c_vec, d_vec)

    # Формирование полного вектора вторых производных
    # m0 = 0, m1...m_{n-1} из решения, mn = 0
    m_full = [0.0] * (n + 1)
    for i in range(len(m_inner)):
        m_full[i + 1] = m_inner[i]
    m_full[n] = 0.0

    # Вычисление коэффициентов сплайна для каждого отрезка
    splines = []
    for i in range(n):
        a_i = Y[i]  # значение в левом конце
        b_i = (Y[i + 1] - Y[i]) / h[i] - h[i] * (2 * m_full[i] + m_full[i + 1]) / 6
        c_i = m_full[i] / 2
        d_i = (m_full[i + 1] - m_full[i]) / (6 * h[i])

        splines.append(SplineCoeffs(a_i, b_i, c_i, d_i))

    return splines


def evaluate_spline(x_nodes, coeffs, x_val):
    """
    Вычисление значения сплайна в заданной точке

    Параметры:
        x_nodes - массив узлов интерполяции
        coeffs - список коэффициентов сплайна
        x_val - точка, в которой вычисляется значение

    Возвращает:
        y_val - значение сплайна в точке x_val
    """
    # Поиск отрезка, содержащего x_val
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            dx = x_val - x_nodes[i]
            return (coeffs[i].a + coeffs[i].b * dx +
                    coeffs[i].c * dx * dx + coeffs[i].d * dx * dx * dx)

    # Если точка вне диапазона, используем экстраполяцию первым/последним отрезком
    if x_val < x_nodes[0]:
        dx = x_val - x_nodes[0]
        return (coeffs[0].a + coeffs[0].b * dx +
                coeffs[0].c * dx * dx + coeffs[0].d * dx * dx * dx)
    else:
        dx = x_val - x_nodes[-2]
        return (coeffs[-1].a + coeffs[-1].b * dx +
                coeffs[-1].c * dx * dx + coeffs[-1].d * dx * dx * dx)


def find_interval(x_nodes, x_val):
    """
    Поиск индекса отрезка, содержащего заданную точку

    Параметры:
        x_nodes - массив узлов
        x_val - искомая точка

    Возвращает:
        index - индекс отрезка [x_i, x_{i+1}], содержащего x_val
    """
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x_val <= x_nodes[i + 1]:
            return i
    return 0


def calculate_spline_value(x_nodes, coeffs, x_val):
    """
    Вычисление значения сплайна с подробной информацией о вычислениях

    Параметры:
        x_nodes - массив узлов
        coeffs - коэффициенты сплайна
        x_val - точка вычисления

    Возвращает:
        tuple: (y_value, interval_index, dx, calculation_steps)
    """
    interval_idx = find_interval(x_nodes, x_val)
    dx = x_val - x_nodes[interval_idx]

    coeff = coeffs[interval_idx]
    calculation_steps = [
        f"a_{interval_idx} = {coeff.a:.6f}",
        f"b_{interval_idx}·Δx = {coeff.b:.6f}·{dx:.6f} = {coeff.b * dx:.6f}",
        f"c_{interval_idx}·Δx² = {coeff.c:.6f}·{dx * dx:.6f} = {coeff.c * dx * dx:.6f}",
        f"d_{interval_idx}·Δx³ = {coeff.d:.6f}·{dx * dx * dx:.6f} = {coeff.d * dx * dx * dx:.6f}"
    ]

    y_val = evaluate_spline(x_nodes, coeffs, x_val)

    return y_val, interval_idx, dx, calculation_steps


def verify_spline_at_nodes(x_nodes, y_nodes, coeffs):
    """
    Проверка точности сплайна в узловых точках

    Параметры:
        x_nodes - узлы интерполяции
        y_nodes - значения в узлах
        coeffs - коэффициенты сплайна

    Возвращает:
        list - список кортежей (x, s_value, y_true, error) для каждого узла
    """
    verification = []
    for i in range(len(x_nodes)):
        s_val = evaluate_spline(x_nodes, coeffs, x_nodes[i])
        error = abs(s_val - y_nodes[i])
        verification.append((x_nodes[i], s_val, y_nodes[i], error))

    return verification




# ==============================
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ==============================

import matplotlib

matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QTabWidget, QTextEdit, QLabel,
                             QScrollArea, QGridLayout)
from PyQt6.QtCore import Qt


class PlotWidget(QWidget):
    def __init__(self, xi, yi, coeffs, x_star):
        super().__init__()
        self.xi = xi
        self.yi = yi
        self.coeffs = coeffs
        self.x_star = x_star
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Создаем matplotlib figure
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)

        # Заголовки
        title_label = QLabel("График кубического сплайна")
        description_label = QLabel("Синяя линия — кубический сплайн\n"
                                   "Черные точки — данные таблицы, Красный крест — точка интерполяции x*")

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(self.canvas)

        self.setLayout(layout)
        self.plot_spline()

    def plot_spline(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Построение сплайна
        x_min, x_max = min(self.xi), max(self.xi)
        x_plot = []
        y_plot = []

        steps = 100
        for i in range(len(self.xi) - 1):
            for j in range(steps):
                x_val = self.xi[i] + j * (self.xi[i + 1] - self.xi[i]) / steps
                y_val = evaluate_spline(self.xi, self.coeffs, x_val)
                x_plot.append(x_val)
                y_plot.append(y_val)

        ax.plot(x_plot, y_plot, 'b-', linewidth=2, label='Кубический сплайн')

        # Исходные точки
        ax.plot(self.xi, self.yi, 'ko', markersize=6, label='Исходные данные')

        # Точка интерполяции
        y_star = evaluate_spline(self.xi, self.coeffs, self.x_star)
        ax.plot(self.x_star, y_star, 'rx', markersize=12, markeredgewidth=3,
                label=f'x* = {self.x_star}')

        # Настройки графика
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Естественный кубический сплайн дефекта 1')
        ax.legend()

        # Добавляем немного места по краям
        margin = 0.1 * (x_max - x_min)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(min(self.yi) - 0.5, max(self.yi) + 0.5)

        self.canvas.draw()


class ResultsWidget(QWidget):
    def __init__(self, xi, yi, coeffs, x_star):
        super().__init__()
        self.xi = xi
        self.yi = yi
        self.coeffs = coeffs
        self.x_star = x_star
        self.init_ui()

    def init_ui(self):
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        # Вычисление результатов с использованием чистых численных методов
        y_val, interval_idx, dx, calc_steps = calculate_spline_value(
            self.xi, self.coeffs, self.x_star
        )
        verification = verify_spline_at_nodes(self.xi, self.yi, self.coeffs)

        # Основные результаты
        result_text = f"""Естественный кубический сплайн дефекта 1

Точка интерполяции: x* = {self.x_star:.3f}
Значение сплайна: S({self.x_star:.3f}) = {y_val:.6f}

Точка x* находится в интервале [x{interval_idx}, x{interval_idx + 1}] = [{self.xi[interval_idx]:.3f}, {self.xi[interval_idx + 1]:.3f}]
"""

        result_label = QLabel(result_text)
        result_label.setWordWrap(True)
        layout.addWidget(result_label)
        layout.addWidget(QLabel("─" * 80))


        # Коэффициенты для интервала с x*
        coeff = self.coeffs[interval_idx]
        interval_text = f"""Коэффициенты сплайна на отрезке, содержащем x* = {self.x_star:.3f}:

Отрезок [x{interval_idx}, x{interval_idx + 1}] = [{self.xi[interval_idx]:.3f}, {self.xi[interval_idx + 1]:.3f}]:

S{interval_idx}(x) = {coeff.a:.6f} + {coeff.b:.6f}(x-{self.xi[interval_idx]:.3f}) + {coeff.c:.6f}(x-{self.xi[interval_idx]:.3f})² + {coeff.d:.6f}(x-{self.xi[interval_idx]:.3f})³

a{interval_idx} = {coeff.a:.6f}
b{interval_idx} = {coeff.b:.6f}
c{interval_idx} = {coeff.c:.6f}
d{interval_idx} = {coeff.d:.6f}

Вычисление S{interval_idx}({self.x_star:.3f}):
Δx = {self.x_star:.3f} - {self.xi[interval_idx]:.3f} = {dx:.6f}
"""

        # Добавляем шаги вычисления
        for step in calc_steps:
            interval_text += f"{step}\n"

        interval_text += f"         = {y_val:.6f}"

        interval_label = QLabel(interval_text)
        interval_label.setWordWrap(True)
        layout.addWidget(interval_label)
        layout.addWidget(QLabel("─" * 80))

        # Проверка в узловых точках
        check_text = "Проверка в узловых точках:\n\n"
        for x, s_val, y_true, error in verification:
            check_text += f"x = {x:.3f}: S(x) = {s_val:.6f}, y = {y_true:.3f}, |S-y| = {error:.3e}\n"

        check_label = QLabel(check_text)
        check_label.setWordWrap(True)
        layout.addWidget(check_label)
        layout.addWidget(QLabel("─" * 80))

        # Информация о сплайне
        info_text = """Естественный кубический сплайн дефекта 1:

Свойства:
• Сплайн проходит через все узловые точки
• Непрерывны функция и её первые две производные
• На концах отрезка вторая производная равна нулю (естественные граничные условия)
• Дефект 1 означает, что третья производная может иметь разрывы в узлах

Формула сплайна на отрезке [xi, xi+1]:
Si(x) = ai + bi(x-xi) + ci(x-xi)² + di(x-xi)³

где ai, bi, ci, di — коэффициенты сплайна на i-м отрезке"""

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Результаты интерполяции"))
        main_layout.addWidget(QLabel("─" * 80))
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        result_label.setStyleSheet("font-size: 16px;")
        interval_label.setStyleSheet("font-size: 16px;")
        check_label.setStyleSheet("font-size: 16px;")
        info_label.setStyleSheet("font-size: 16px;")


class CoefficientsWidget(QWidget):
    def __init__(self, xi, yi, coeffs):
        super().__init__()
        self.xi = xi
        self.yi = yi
        self.coeffs = coeffs
        self.init_ui()

    def init_ui(self):
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)

        coeffs_text = "Коэффициенты сплайна на всех отрезках:\n\n"
        for i in range(len(self.coeffs)):
            coeff = self.coeffs[i]
            coeffs_text += f"Отрезок [x{i}, x{i + 1}] = [{self.xi[i]:.3f}, {self.xi[i + 1]:.3f}]:\n"
            coeffs_text += f"  S{i}(x) = {coeff.a:.6f} + {coeff.b:.6f}(x-{self.xi[i]:.3f}) + {coeff.c:.6f}(x-{self.xi[i]:.3f})² + {coeff.d:.6f}(x-{self.xi[i]:.3f})³\n"
            coeffs_text += f"  a{i} = {coeff.a:.6f}\n"
            coeffs_text += f"  b{i} = {coeff.b:.6f}\n"
            coeffs_text += f"  c{i} = {coeff.c:.6f}\n"
            coeffs_text += f"  d{i} = {coeff.d:.6f}\n\n"

        text_edit = QTextEdit()
        text_edit.setPlainText(coeffs_text)
        text_edit.setReadOnly(True)

        layout.addWidget(text_edit)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)

        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel("Коэффициенты сплайна"))
        main_layout.addWidget(QLabel("─" * 80))
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)


class DataTableWidget(QWidget):
    def __init__(self, xi, yi):
        super().__init__()
        self.xi = xi
        self.yi = yi
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Создаем таблицу данных
        grid_layout = QGridLayout()

        # Заголовки
        grid_layout.addWidget(QLabel("i:"), 0, 0)
        grid_layout.addWidget(QLabel("xi:"), 1, 0)
        grid_layout.addWidget(QLabel("yi:"), 2, 0)

        # Данные
        for i in range(len(self.xi)):
            grid_layout.addWidget(QLabel(f"{i:8d}"), 0, i + 1)
            grid_layout.addWidget(QLabel(f"{self.xi[i]:8.3f}"), 1, i + 1)
            grid_layout.addWidget(QLabel(f"{self.yi[i]:8.3f}"), 2, i + 1)

        table_widget = QWidget()
        table_widget.setLayout(grid_layout)

        layout.addWidget(QLabel("Исходные данные"))
        layout.addWidget(QLabel("─" * 80))
        layout.addWidget(table_widget)

        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.xi = XI
        self.yi = YI
        self.x_star = X_STAR

        # Использование чистых численных методов
        self.coeffs = build_natural_cubic_spline(self.xi, self.yi)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Естественный кубический сплайн")
        self.setGeometry(100, 100, 1200, 800)

        # Создаем вкладки
        tab_widget = QTabWidget()

        # Вкладка с графиком
        plot_widget = PlotWidget(self.xi, self.yi, self.coeffs, self.x_star)
        tab_widget.addTab(plot_widget, "График")

        # Вкладка с результатами
        results_widget = ResultsWidget(self.xi, self.yi, self.coeffs, self.x_star)
        tab_widget.addTab(results_widget, "Результаты")

        # Вкладка с коэффициентами
        coeffs_widget = CoefficientsWidget(self.xi, self.yi, self.coeffs)
        tab_widget.addTab(coeffs_widget, "Коэффициенты")


        self.setCentralWidget(tab_widget)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()