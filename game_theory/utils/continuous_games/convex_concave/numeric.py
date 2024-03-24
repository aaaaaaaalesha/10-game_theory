"""Численный метод решения выпукло-вогнутых игр с непрерывным ядром."""
import logging
from collections import deque

import numpy as np
import sympy
from sympy.core import function

from game_theory.utils.continuous_games.exceptions import ContinuousGameException
from game_theory.utils.continuous_games.types import SizeType, ValueType
from game_theory.utils.matrix_games.brown_robinson.brown_robinson import BrownRobinson
from game_theory.utils.matrix_games.game_matrix import GameMatrix

_logger = logging.getLogger(__name__)


class NumericMethod:
    """
    Численный метод решения выпукло-вогнутых игр с непрерывным ядром.
    """

    def __init__(self, kernel_func: function.Lambda, accuracy: float = 0.1, deltas_count: SizeType = 3):
        # Функция ядра непрерывной выпукло-вогнутой игры.
        self.kernel_func: function.Lambda = kernel_func
        # Точность численного метода.
        self.accuracy: float = accuracy
        # Разности между соседними вычисленными значениями цены игры на итерациях.
        max_len: SizeType = deltas_count
        self.__deltas: deque[float] = deque(maxlen=max_len)
        self.__estimates: deque[tuple[float, float, ValueType]] = deque(maxlen=max_len)

        # Проверяем, что игра является выпукло-вогнутой.
        x, y = sympy.symbols(("x", "y"))
        kernel_xx = sympy.diff(self.kernel_func(x, y), x, 2)
        kernel_yy = sympy.diff(self.kernel_func(x, y), y, 2)
        if not (kernel_xx < 0 < kernel_yy):
            err_msg = (
                "Игра не является выпукло-вогнутой, "
                "т.к. для функции ядра одновременно не выполняется оба условия: \n"
                f"H_xx = {kernel_xx:.2f} < 0 и H_yy = {kernel_yy:.2f} > 0"
            )
            raise ContinuousGameException(err_msg)

    def solve(self) -> tuple[float, float, ValueType]:
        n = 2
        prev_game_price_estimate, game_price_estimate = None, None
        while len(self.__deltas) == 0 or sum(self.__deltas) > self.accuracy:
            _logger.info(f"N = {n} (шаг: {1 / n:.3f})")
            grid_game_matrix = GameMatrix(self._generate_grid_approximation_matrix(n))
            _logger.info(grid_game_matrix)

            # Проверка седла.
            i, lgp_value = grid_game_matrix.lowest_game_price
            j, hgp_value = grid_game_matrix.highest_game_price
            if lgp_value == hgp_value:
                game_price_estimate = lgp_value
                x_estimate, y_estimate = i / n, j / n
                _logger.info("Седловая точка найдена:")
            else:
                br_method = BrownRobinson(grid_game_matrix, accuracy=self.accuracy)
                br_method.solve()
                game_price_estimate = br_method.game_price_estimation
                x_mixed_strategies, y_mixed_strategies = br_method.mixed_strategies
                x_estimate, y_estimate = np.argmax(x_mixed_strategies) / n, np.argmax(y_mixed_strategies) / n
                _logger.info("Седловой точки нет. Решение методом Брауна-Робинсон:")

            _logger.info(f"x = {x_estimate:.3f}; y = {y_estimate:.3f}; H = {game_price_estimate:.3f}\n")
            if prev_game_price_estimate is not None:
                # Добавляем разность между оценками цен игры на текущей и предыдущей итерации в deque.
                self.__push_estimates(
                    delta=np.abs(game_price_estimate - prev_game_price_estimate),
                    x_est=x_estimate,
                    y_est=y_estimate,
                    game_price_est=game_price_estimate,
                )

            prev_game_price_estimate = game_price_estimate
            n += 1

        x_estimate, y_estimate, game_price_estimate = (
            np.mean([est_tuple[i] for est_tuple in self.__estimates]) for i in range(3)
        )
        _logger.info(
            "Таким образом численно найдено решение задачи:\n"
            f"x ≈ {x_estimate:.3f}; y ≈ {y_estimate:.3f}; H ≈ {game_price_estimate:.3f}"
        )

        return x_estimate, y_estimate, game_price_estimate

    def _generate_grid_approximation_matrix(self, iteration_number: int) -> np.ndarray:
        """
        Генерирует квадратную матрицу аппроксимации функции ядра (выигрышей) на сетке.
        Матрица имеет размерность `iteration_number` + 1
        Элемент матрицы a_ij = H(i / iteration_number; j / iteration_number), где H - функция ядра от двух переменных.
        :param int iteration_number: Размерность сетки.
        :return: Матрица сетки.
        """
        return np.array(
            [
                [
                    float(self.kernel_func(i / iteration_number, j / iteration_number))
                    for j in range(iteration_number + 1)
                ]
                for i in range(iteration_number + 1)
            ]
        )

    def __push_estimates(self, delta: float, x_est: float, y_est: float, game_price_est: ValueType) -> None:
        """Добавляем оценки результатов и разностей прироста цены игры в deque-контейнеры."""
        if len(self.__deltas) == self.__deltas.maxlen:
            self.__deltas.popleft()
            self.__estimates.popleft()

        self.__deltas.append(delta)
        self.__estimates.append((x_est, y_est, game_price_est))
