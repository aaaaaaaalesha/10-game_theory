"""Прямоугольная матрица игры двух лиц."""
import copy
import logging

import numpy as np
from prettytable import PrettyTable

from .exceptions import MatrixGameException
from .types import IndexType, ValueType

_logger = logging.getLogger(__name__)


class GameMatrix:
    """Реализация матрицы игры для двух игроков: A и B"""

    def __init__(
        self,
        matrix: np.array,
        player_a_strategy_labels: list[str] | None = None,
        player_b_strategy_labels: list[str] | None = None,
    ):
        self.game_matrix: np.array = matrix
        self.player_a_strategy_labels = player_a_strategy_labels
        self.player_b_strategy_labels = player_b_strategy_labels
        if player_a_strategy_labels is None:
            self.player_a_strategy_labels: list[str] = [f"a{i + 1}" for i in range(matrix.shape[0])]
        if player_b_strategy_labels is None:
            self.player_b_strategy_labels: list[str] = [f"b{i + 1}" for i in range(matrix.shape[1])]

    def __str__(self):
        strategy_table = PrettyTable(
            title="Таблица стратегий (игрока А)",
            field_names=("Стратегии", *self.player_b_strategy_labels, "MIN выигрыш A"),
        )
        # Добавляем стратегии игрока A со столбцом MIN выигрыша A.
        strategy_table.add_rows(
            rows=(
                (self.player_a_strategy_labels[i], *self.game_matrix[i], min_winning)
                for i, min_winning in enumerate(self.min_wins_player_a)
            )
        )

        # Добавляем заключительную строку с MAX проигрышами B.
        strategy_table.add_row(("MAX проигрыш B", *self.max_looses_player_b, " "))

        return str(strategy_table)

    def __repr__(self):
        return str(self)

    @property
    def min_wins_player_a(self) -> np.array:
        """Значения MIN выигрыша игрока А в матричной игре."""
        return np.array([min(row) for row in self.game_matrix])

    @property
    def max_looses_player_b(self) -> np.array:
        """Значения MAX проигрыша игрока B в матричной игре."""
        return np.array([max(column) for column in self.game_matrix.T])

    @property
    def lower_game_price(self) -> tuple[IndexType, ValueType]:
        """Возвращает индекс и значение нижней цены игры (максимин, max_j min_i c_{ij})"""
        low_price_index: int = np.argmax(self.min_wins_player_a, axis=0)
        return low_price_index, self.min_wins_player_a[low_price_index]

    @property
    def upper_game_price(self) -> tuple[IndexType, ValueType]:
        """Возвращает индекс и значение верхней цены игры (минимакс, min_i max_j c_{ij})"""
        upper_price_index: int = np.argmin(self.max_looses_player_b, axis=0)
        return upper_price_index, self.max_looses_player_b[upper_price_index]

    def normalize_matrix(self) -> None:
        """Приводит исходную матрицу к нормализованной (с неотрицательными коэффициентами) in-place."""
        min_element = np.min(self.game_matrix)
        if min_element < 0:
            self.game_matrix += -min_element

    def drop_duplicate_strategies(self) -> None:
        """Удаляет дублирующиеся стратегии игроков A и B in-place."""
        # Удаление дубликатов строк.
        _, idx_rows = np.unique(self.game_matrix, axis=0, return_index=True)
        self.game_matrix = self.game_matrix[np.sort(idx_rows)]
        self.player_a_strategy_labels = [self.player_a_strategy_labels[i] for i in np.sort(idx_rows)]

        # Удаление дубликатов столбцов.
        _, idx_cols = np.unique(self.game_matrix, axis=1, return_index=True)
        self.game_matrix = self.game_matrix[:, np.sort(idx_cols)]
        self.player_b_strategy_labels = [self.player_b_strategy_labels[i] for i in np.sort(idx_cols)]

    def reduce_dimension(self, mode="dominant_absorption") -> "GameMatrix":
        """
        Сводит к минимуму размерность матричной игры.
        :param str mode: Выбор метода уменьшения размерности матричной игры:
            - `dominant_absorption` - поглощение доминируемых стратегий;
            - `nbr_drop` - удаление NBR-стратегий.
        :returns: Экземпляр GameMatrix с уменьшенной размерностью
        """
        self.drop_duplicate_strategies()
        match mode:
            case "dominant_absorption":
                reduced_matrix: GameMatrix = self._dominant_absorption_reduce()
            case "nbr_drop":
                reduced_matrix: GameMatrix = self._nbr_drop_reduce()
            case _:
                exc_msg = f"Invalid mode of reducing matrix: {mode}"
                raise MatrixGameException(exc_msg)

        return reduced_matrix

    def _dominant_absorption_reduce(self) -> "GameMatrix":
        """Сводит к минимуму размерность матричной игры поглощением доминируемых стратегий."""
        # Поиск и удаление доминируемых строк
        dominated_rows: set[int] = self.__find_dominated_rows()
        if dominated_rows:
            # Обновление индексов строк для удаления.
            dominated_rows: list[int] = sorted(dominated_rows)
            # Удаление доминируемых строк.
            reduced_matrix: np.array = np.delete(self.game_matrix, dominated_rows, axis=0)
            # Обновление соответствующих стратегий игрока A.
            player_a_strategy_labels = [
                label for i, label in enumerate(self.player_a_strategy_labels) if i not in dominated_rows
            ]
            return GameMatrix(
                reduced_matrix,
                player_a_strategy_labels,
                self.player_b_strategy_labels,
            )._dominant_absorption_reduce()

        # Поиск и удаление доминируемых столбцов.
        dominated_columns: set[int] = self.__find_dominated_columns()
        if dominated_columns:
            # Обновление индексов столбцов для удаления.
            dominated_columns: list[int] = sorted(dominated_columns)
            # Удаление доминируемых строк и столбцов.
            reduced_matrix: np.array = np.delete(self.game_matrix, dominated_columns, axis=1)
            # Обновление соответствующих стратегий игрока B.
            player_b_strategy_labels = [
                label for i, label in enumerate(self.player_b_strategy_labels) if i not in dominated_columns
            ]
            return GameMatrix(
                reduced_matrix,
                self.player_a_strategy_labels,
                player_b_strategy_labels,
            )._dominant_absorption_reduce()

        return copy.deepcopy(self)

    def _nbr_drop_reduce(self):
        """Сводит к минимуму размерность матричной игры удалением NBR-стратегий."""
        exc_msg = "Этот метод я пока не прошёл :("
        raise NotImplementedError(exc_msg)

    def __find_dominated_rows(self) -> set[int]:
        """Возвращает множество индексов доминируемых строк в текущей матрице."""
        dominated_rows = set()
        # Поиск и удаление доминируемых строк
        for i in range(self.game_matrix.shape[0]):
            for j in range(self.game_matrix.shape[0]):
                if i == j or i in dominated_rows:
                    continue

                if all(self.game_matrix[j] >= self.game_matrix[i]):
                    dominated_rows.add(i)
                    msg = (
                        f"Поглощение стратегии {self.player_a_strategy_labels[i]} "
                        f"доминирующей стратегией {self.player_a_strategy_labels[j]}"
                    )
                    _logger.info(msg)

        return dominated_rows

    def __find_dominated_columns(self) -> set[int]:
        """Возвращает множество индексов доминируемых столбцов в текущей матрице."""
        dominated_columns = set()
        for i in range(self.game_matrix.shape[1]):
            for j in range(self.game_matrix.shape[1]):
                if i == j or i in dominated_columns:
                    continue

                if all(self.game_matrix[:, j] <= self.game_matrix[:, i]):
                    dominated_columns.add(i)
                    msg = (
                        f"Поглощение стратегии {self.player_b_strategy_labels[i]} "
                        f"доминирующей стратегией {self.player_b_strategy_labels[j]}"
                    )
                    _logger.info(msg)

        return dominated_columns
