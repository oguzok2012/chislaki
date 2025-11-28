import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QLabel, QScrollArea, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================
# МАТЕМАТИЧЕСКИЕ ФУНКЦИИ
# ==============================

xi = [1.24, 1.45, 1.66, 1.87, 2.08, 2.29, 2.50, 2.71, 2.92, 3.13, 3.34]
yi = [3.9237, 2.5215, -0.1023, -1.6948, -1.9692, -1.7318, -0.9247, 0.1532, 2.5417, 6.9841, 8.9956]
x_star = 3.413


def solve_slau(A, b):
    """Решение СЛАУ методом Гаусса"""
    n = len(b)
    augmented = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Прямой ход
    for i in range(n):
        # Поиск главного элемента
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

        # Исключение
        for k in range(i + 1, n):
            factor = augmented[k][i] / augmented[i][i]
            for j in range(i, n + 1):
                augmented[k][j] -= factor * augmented[i][j]

    # Обратный ход
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][n]
        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]
        x[i] /= augmented[i][i]

    return x


def least_squares(X, Y, degree):
    """
    Метод наименьших квадратов
    [ a₀·Σx⁰    + a₁·Σx¹    + a₂·Σx²    + ... + aₘ·Σxᵐ      = Σ(y·x⁰) ]
    [ a₀·Σx¹    + a₁·Σx²    + a₂·Σx³    + ... + aₘ·Σxᵐ⁺¹    = Σ(y·x¹) ]
    [ a₀·Σx²    + a₁·Σx³    + a₂·Σx⁴    + ... + aₘ·Σxᵐ⁺²    = Σ(y·x²) ]
    ...
    [ a₀·Σxᵐ    + a₁·Σxᵐ⁺¹  + a₂·Σxᵐ⁺²  + ... + aₘ·Σx²ᵐ      = Σ(y·xᵐ) ]

    Функция ошибки:      Φ = Σ[a₀ + a₁·xᵢ + a₂·xᵢ²  ... - yᵢ]²
                         ∂Φ/∂aₖ = Σ[2 · (a₀ + a₁·xᵢ + ... + aₘ·xᵢᵐ - yᵢ) · xᵢᵏ] = 0
    это и дает -         a₀·Σxᵢᵏ + a₁·Σxᵢᵏ⁺¹ + ... + aₘ·Σxᵢᵏ⁺ᵐ = Σ(yᵢ·xᵢᵏ)
    """
    n = degree + 1
    matrix = [[0.0] * n for _ in range(n)]
    right_side = [0.0] * n

    # Заполнение матрицы
    for i in range(n):
        for j in range(n):
            power = i + j
            total = 0.0
            for x_val in X:
                total += x_val ** power
            matrix[i][j] = total

    # Заполнение правой части
    for i in range(n):
        total = 0.0
        for idx, x_val in enumerate(X):
            total += Y[idx] * (x_val ** i)
        right_side[i] = total

    return solve_slau(matrix, right_side)


def evaluate_polynomial(coeffs, x):
    """Вычисление значения полинома в точке x"""
    result = 0.0
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result


def calculate_squared_error(X, Y, coeffs):
    """Вычисление суммы квадратов ошибок"""
    total = 0.0
    for i, x_val in enumerate(X):
        predicted = evaluate_polynomial(coeffs, x_val)
        error = predicted - Y[i]
        total += error * error
    return total


def polynomial_to_string(coeffs):
    """Преобразование коэффициентов в строку"""
    if not coeffs:
        return "0"

    result = f"{coeffs[0]:.6f}"
    for i in range(1, len(coeffs)):
        sign = "+"
        val = coeffs[i]
        if val < 0:
            sign = "-"
            val = -val

        if i == 1:
            result += f" {sign} {val:.6f}·x"
        else:
            result += f" {sign} {val:.6f}·x^{i}"

    return result


def calculate_all_results():
    """Вычисление всех результатов"""
    coeffs1 = least_squares(xi, yi, 1)
    coeffs2 = least_squares(xi, yi, 2)
    coeffs3 = least_squares(xi, yi, 3)

    error1 = calculate_squared_error(xi, yi, coeffs1)
    error2 = calculate_squared_error(xi, yi, coeffs2)
    error3 = calculate_squared_error(xi, yi, coeffs3)

    value1 = evaluate_polynomial(coeffs1, x_star)
    value2 = evaluate_polynomial(coeffs2, x_star)
    value3 = evaluate_polynomial(coeffs3, x_star)

    min_error = min(error1, error2, error3)
    best_degree = 1
    if error2 == min_error:
        best_degree = 2
    elif error3 == min_error:
        best_degree = 3

    # Текстовые результаты
    poly1_text = (f"Многочлен 1-й степени\n"
                  f"F₁(x) = {polynomial_to_string(coeffs1)}\n"
                  f"Сумма квадратов ошибок: Φ₁ = {error1:.6f}\n"
                  f"F₁({x_star:.3f}) = {value1:.6f}")

    poly2_text = (f"Многочлен 2-й степени\n"
                  f"F₂(x) = {polynomial_to_string(coeffs2)}\n"
                  f"Сумма квадратов ошибок: Φ₂ = {error2:.6f}\n"
                  f"F₂({x_star:.3f}) = {value2:.6f}")

    poly3_text = (f"Многочлен 3-й степени\n"
                  f"F₃(x) = {polynomial_to_string(coeffs3)}\n"
                  f"Сумма квадратов ошибок: Φ₃ = {error3:.6f}\n"
                  f"F₃({x_star:.3f}) = {value3:.6f}")

    compare_text = (f"Сравнение результатов\n\n"
                    f"Степень 1:  Φ = {error1:.8f},  F(x*) = {value1:.10f}\n"
                    f"Степень 2:  Φ = {error2:.8f},  F(x*) = {value2:.10f}\n"
                    f"Степень 3:  Φ = {error3:.8f},  F(x*) = {value3:.10f}\n\n"
                    f"Наименьшая сумма квадратов ошибок у многочлена {best_degree}-й степени: Φ = {min_error:.6f}")

    # Коэффициенты
    coeffs1_text = "Многочлен 1-й степени: F₁(x) = a₀ + a₁·x\n"
    for i, c in enumerate(coeffs1):
        coeffs1_text += f"  a{i} = {c:.10f}\n"

    coeffs2_text = "Многочлен 2-й степени: F₂(x) = a₀ + a₁·x + a₂·x²\n"
    for i, c in enumerate(coeffs2):
        coeffs2_text += f"  a{i} = {c:.10f}\n"

    coeffs3_text = "Многочлен 3-й степени: F₃(x) = a₀ + a₁·x + a₂·x² + a₃·x³\n"
    for i, c in enumerate(coeffs3):
        coeffs3_text += f"  a{i} = {c:.10f}\n"

    # Проверка в узловых точках
    check_text = "\n"
    for i in range(len(xi)):
        f1 = evaluate_polynomial(coeffs1, xi[i])
        f2 = evaluate_polynomial(coeffs2, xi[i])
        f3 = evaluate_polynomial(coeffs3, xi[i])
        check_text += f"x = {xi[i]:6.2f}:  y = {yi[i]:8.4f},  F₁(x) = {f1:8.4f},  F₂(x) = {f2:8.4f},  F₃(x) = {f3:8.4f}\n"

    return {
        'coeffs1': coeffs1,
        'coeffs2': coeffs2,
        'coeffs3': coeffs3,
        'errors': (error1, error2, error3),
        'values': (value1, value2, value3),
        'best_degree': best_degree,
        'min_error': min_error,
        'texts': {
            'poly1': poly1_text,
            'poly2': poly2_text,
            'poly3': poly3_text,
            'compare': compare_text,
            'coeffs1': coeffs1_text,
            'coeffs2': coeffs2_text,
            'coeffs3': coeffs3_text,
            'check': check_text
        }
    }


# ==============================
# ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ==============================

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()

    def plot_polynomials(self, coeffs1, coeffs2, coeffs3):
        """Построение графиков полиномов"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Настройки графика для новых данных
        x_min, x_max = 1.0, 3.5
        y_min, y_max = -3.0, 10.0

        # Построение полиномов
        x_vals = [x_min + i * (x_max - x_min) / 200 for i in range(201)]

        y1_vals = [evaluate_polynomial(coeffs1, x) for x in x_vals]
        y2_vals = [evaluate_polynomial(coeffs2, x) for x in x_vals]
        y3_vals = [evaluate_polynomial(coeffs3, x) for x in x_vals]

        ax.plot(x_vals, y1_vals, 'r-', linewidth=2, label='1-я степень')
        ax.plot(x_vals, y2_vals, 'g-', linewidth=2, label='2-я степень')
        ax.plot(x_vals, y3_vals, 'b-', linewidth=2, label='3-я степень')

        # Исходные точки
        ax.scatter(xi, yi, color='black', s=50, zorder=5, label='Исходные данные')

        # Точка x*
        y_star = evaluate_polynomial(coeffs1, x_star)
        ax.plot([x_star], [y_star], 'rx', markersize=10, markeredgewidth=2, label=f'x* = {x_star}')

        # Настройка внешнего вида
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=1)
        ax.axvline(x=0, color='k', linewidth=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Аппроксимация методом наименьших квадратов')
        ax.legend()

        self.draw()


class ScrollLabel(QScrollArea):
    def __init__(self, text=""):
        super().__init__()
        self.setWidgetResizable(True)
        content = QWidget()
        self.setWidget(content)
        layout = QVBoxLayout(content)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setFont(QFont("Arial", 12))
        layout.addWidget(self.label)


class LeastSquaresApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.results = calculate_all_results()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Метод наименьших квадратов (МНК)")
        self.setGeometry(100, 100, 1200, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Создаем вкладки
        tabs = QTabWidget()

        # Вкладка с графиком
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)

        plot_canvas = PlotCanvas(self, width=10, height=8)
        plot_canvas.plot_polynomials(
            self.results['coeffs1'],
            self.results['coeffs2'],
            self.results['coeffs3']
        )

        plot_info = QLabel(
            "Красная линия — многочлен 1-й степени\n"
            "Зеленая линия — многочлен 2-й степени\n"
            "Синяя линия — многочлен 3-й степени\n"
            "Черные точки — исходные данные\n"
            "Красный крест — точка x*"
        )
        plot_info.setFont(QFont("Arial", 15))

        plot_layout.addWidget(plot_canvas)
        plot_layout.addWidget(plot_info)

        # Вкладка с результатами
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        header_text = (f"Метод наименьших квадратов (МНК)\n"
                       f"Точка вычисления: x* = {x_star:.3f}\n\n")

        results_text = (f"{header_text}\n"
                        f"{self.results['texts']['poly1']}\n\n"
                        f"{self.results['texts']['poly2']}\n\n"
                        f"{self.results['texts']['poly3']}\n\n"
                        f"{self.results['texts']['compare']}\n\n"
                        f"Метод наименьших квадратов (МНК):\n\n"
                        f"Метод позволяет найти приближающий многочлен заданной степени,\n"
                        f"минимизирующий сумму квадратов отклонений от исходных данных.\n\n"
                        f"Для многочлена степени n:\n"
                        f"F(x) = a₀ + a₁·x + a₂·x² + ... + aₙ·xⁿ\n\n"
                        f"Коэффициенты находятся из решения нормальной системы МНК.\n\n"
                        f"Сумма квадратов ошибок:\n"
                        f"Φ = Σ(F(xᵢ) - yᵢ)²\n\n"
                        f"Чем меньше Φ, тем лучше приближение.")

        results_label = ScrollLabel(results_text)
        results_label.label.setFont(QFont("Courier New", 15))
        results_layout.addWidget(results_label)

        # Вкладка с коэффициентами
        coeffs_tab = QWidget()
        coeffs_layout = QVBoxLayout(coeffs_tab)

        coeffs_text = (f"{self.results['texts']['coeffs1']}\n"
                       f"{self.results['texts']['coeffs2']}\n"
                       f"{self.results['texts']['coeffs3']}")

        coeffs_label = ScrollLabel(coeffs_text)
        coeffs_label.label.setFont(QFont("Courier New", 15))
        coeffs_layout.addWidget(coeffs_label)

        # Вкладка с проверкой
        check_tab = QWidget()
        check_layout = QVBoxLayout(check_tab)

        check_text = self.results['texts']['check']
        check_label = ScrollLabel(check_text)
        check_label.label.setFont(QFont("Courier New", 15))
        check_layout.addWidget(check_label)

        # Вкладка с таблицей данных
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)

        data_text = "Исходные данные:\n\n"
        data_text += "i:  " + "".join(f"{i:8d}" for i in range(len(xi))) + "\n"
        data_text += "xi: " + "".join(f"{x:8.2f}" for x in xi) + "\n"
        data_text += "yi: " + "".join(f"{y:8.4f}" for y in yi) + "\n"

        data_label = ScrollLabel(data_text)
        data_label.label.setFont(QFont("Courier New", 15))
        data_layout.addWidget(data_label)

        # Добавляем вкладки
        tabs.addTab(plot_tab, "График")
        tabs.addTab(results_tab, "Результаты")
        tabs.addTab(coeffs_tab, "Коэффициенты")
        tabs.addTab(check_tab, "Проверка")
        tabs.addTab(data_tab, "Таблица данных")

        layout.addWidget(tabs)


def main():
    app = QApplication(sys.argv)
    window = LeastSquaresApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()