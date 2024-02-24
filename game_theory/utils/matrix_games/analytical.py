"""Реализация аналитического (матричного) метода решения матричной игры."""
import logging

import numpy as np

from .game_matrix import GameMatrix

_logger = logging.getLogger(__name__)


class AnalyticalSolver:
    """Класс решения матричной игры аналитическим (матричным) методом."""

    def __init__(self, game_matrix: GameMatrix):
        self.game_matrix = game_matrix

    def player_a_solve(self) -> np.ndarray:
        """
        Решение матричной СЛАУ для нахождения цены игры и смешанных стратегий игрока A.
        :returns: Решение матричной СЛАУ вида (x_1, ..., x_n, game_price_value).
        """
        return self.__solve_linear_system(self.game_matrix.matrix.T)

    def player_b_solve(self):
        """
        Решение матричной СЛАУ для нахождения цены игры и смешанных стратегий игрока B.
        :returns: Решение матричной СЛАУ вида (y_1, ..., y_m, game_price_value).
        """
        return self.__solve_linear_system(self.game_matrix.matrix)

    @staticmethod
    def __solve_linear_system(base_matrix: np.ndarray) -> np.ndarray:
        """
        Для базовой матрицы формирует СЛАУ для нахождения цены игры и смешанных стратегий игрока.
        :returns: Решение матричной СЛАУ вида (v_1, ..., v_n, game_price_value).
        """
        # Добавление строки матрицы A.
        a_matrix = np.vstack((base_matrix, np.ones(base_matrix.shape[1])))
        # Добавление столбца матрицы A.
        a_matrix = np.hstack((a_matrix, np.array([-1] * (a_matrix.shape[0] - 1) + [0]).reshape(-1, 1)))

        # Формирование столбца b.
        b_column: np.ndarray = np.array([0] * (a_matrix.shape[0] - 1) + [1])
        # Решение матричной СЛАУ.
        return np.linalg.solve(a_matrix, b_column)
