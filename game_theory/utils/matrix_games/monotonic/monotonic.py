"""Монотонный итеративный алгоритм решения матричной игры (n x m)-игры с нулевой суммой."""
import logging
import random

import numpy as np
from numpy import ndarray

from game_theory.utils.matrix_games.game_matrix import GameMatrix
from game_theory.utils.matrix_games.types import IndexType, ValueType

_logger = logging.getLogger(__name__)


class Monotonic:
    """Класс инкапсулирует решение матричной игры монотонным итеративным методом."""

    def __init__(self, game_matrix: GameMatrix):
        self.game: GameMatrix = game_matrix

        self.iteration_number: int = 0
        self.strategy_index: IndexType
        self.strategy_x: np.ndarray[float]
        self.scores_c: np.ndarray[ValueType]
        self.price_v: ValueType
        self.indicator_mask_j: np.ndarray[bool]

    def solve(self):
        # Решение за игрока A.
        _logger.info("Решение игры относительно игрока A")
        price_a, strategy_a = self._base_solve(np.float16(self.game.matrix))
        iterations_a_count = self.iteration_number
        # Решения за игрока B.
        _logger.info("Решение игры относительно игрока B")
        price_b, strategy_b = self._base_solve(np.float16(self.game.matrix.T))
        iterations_b_count = self.iteration_number

        _logger.info(f"Итераций игроков сделано {(iterations_a_count, iterations_b_count)}")
        return (price_a, strategy_a), (price_b, strategy_b)

    def _base_solve(self, matrix: np.ndarray[ValueType]):
        m, n = matrix.shape
        self.iteration_number = 0
        _logger.info("Итерация 0:")
        # Выбираем произвольную (x^0) чистую стратегию (выставляя 1 только в одну позицию).
        strategy_index: IndexType = random.randint(0, m - 1)
        self.strategy_x = np.array([0] * n)
        self.strategy_x[strategy_index] = 1
        # Выбираем вектор (c^0), соответствующий выбранной стратегии.
        self.scores_c: np.ndarray = matrix[strategy_index].copy()
        # Текущая цена игры.
        self.price_v = np.min(self.scores_c)
        # Вектор-индикатор, который показывает принадлежность к множеству.
        self.indicator_mask_j: np.ndarray[bool] = np.isclose(self.scores_c, self.price_v)
        self.__log_calculated_parameters()

        alpha_values = np.array((np.inf, np.inf))
        # Выполняем итерации без заданной точности, то есть пока α_N не станет 0.
        while not np.allclose(alpha_values, [0, 1]):
            optimal_strategy_x_, optimal_scores_c_, alpha_values = self.perform_iteration(matrix)

            alpha, _ = alpha_values
            if not np.isclose(alpha, 0):
                self.strategy_x = (1 - alpha) * self.strategy_x + alpha * optimal_strategy_x_
                self.scores_c = (1 - alpha) * self.scores_c + alpha * optimal_scores_c_
                self.price_v = np.min(self.scores_c)
                self.indicator_mask_j = np.isclose(self.scores_c, self.price_v)
                self.__log_calculated_parameters()
                # Если в J_i попали все столбцы, завершаем алгоритм.
                if np.all(self.indicator_mask_j):
                    _logger.info(
                        f"Так как в J^{self.iteration_number} попали все номера столбцов, " f"останавливаем алгоритм"
                    )
                    return self.price_v, self.strategy_x.copy()

        _logger.info(f"Получили α_{self.iteration_number} = {alpha:.0f}, поэтому останавливаем алгоритм")

        return self.price_v, self.strategy_x.copy()

    def perform_iteration(self, matrix: np.ndarray[ValueType], accuracy=3) -> tuple[ndarray, ndarray, ndarray]:
        self.iteration_number += 1
        i = self.iteration_number
        _logger.info(f"\nИтерация {self.iteration_number}:")
        # Выбираем только столбцы, удовлетворяющие нашему индикатору.
        sub_game_matrix_a = GameMatrix(matrix[:, self.indicator_mask_j].copy())
        _logger.info(f"Рассмотрим подыгру Г^{i}: " f"\n{np.round(sub_game_matrix_a.matrix, accuracy)}")
        # Решаем подыгру и находим оптимальную стратегию x_.
        _, optimal_strategy_x_ = sub_game_matrix_a.solve()
        optimal_scores_c_: np.ndarray = self.__reduce_sum(matrix, np.array(optimal_strategy_x_))
        _logger.info(
            f"Оптимальная стратегия игрока: "
            f"\n\t‾x_{i} = {np.round(optimal_strategy_x_, accuracy)}"
            f"\n\t‾c_{i} = {np.round(optimal_scores_c_, accuracy)}"
        )
        # Находим оптимальную стратегию игрока в подыгре из двух строк.
        sub_game_gamma = GameMatrix(np.stack((self.scores_c, optimal_scores_c_)))
        _logger.info(
            f"Находим оптимальную стратегию игрока в подыгре из двух строк: "
            f"\n{np.round(sub_game_gamma.matrix, accuracy)}"
        )
        sub_game_gamma = sub_game_gamma.reduce_dimension(method="nbr_drop")
        _logger.info(
            f"Матрица после уменьшения размерности: "
            f"\n{np.round(sub_game_gamma.reduce_dimension().matrix, accuracy)}"
        )
        _, alpha_values = sub_game_gamma.solve()
        alpha_values = (alpha_values[1] if len(alpha_values) == 2 else 0, alpha_values[0])
        _logger.info(
            f"В результате получена оптимальная стратегия: "
            f"(α_{i}, 1 - α_{i}) = "
            f"{np.round(alpha_values, accuracy)}"
        )

        return np.array(optimal_strategy_x_), optimal_scores_c_, np.array(alpha_values)

    @staticmethod
    def __reduce_sum(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.sum((rhs * lhs.T).T, 0)

    def __log_calculated_parameters(self, accuracy=3):
        j_indexes = [i + 1 for i in list(*np.where(self.indicator_mask_j))]
        _logger.info(
            "\n".join(
                [
                    f"\tx^{self.iteration_number} = {np.round(self.strategy_x, accuracy)}",
                    f"\tc^{self.iteration_number} = {np.round(self.scores_c, accuracy)}",
                    f"\tv^{self.iteration_number} = {round(self.price_v, accuracy)}",
                    f"\tJ^{self.iteration_number} = {j_indexes}",
                ]
            )
        )
