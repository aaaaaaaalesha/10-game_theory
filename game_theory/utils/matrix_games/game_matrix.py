"""Прямоугольная матрица игры двух лиц."""
import copy
import logging
import operator

import numpy as np
from prettytable import PrettyTable

from .exceptions import MatrixGameException
from .types import ComparisonOperator, IndexType, LabelType, ValueType

_logger = logging.getLogger(__name__)


class GameMatrix:
    """Реализация матрицы игры для двух игроков: A и B"""

    def __init__(
        self,
        matrix: np.array,
        player_a_strategy_labels: list[LabelType] | None = None,
        player_b_strategy_labels: list[LabelType] | None = None,
    ):
        self.matrix: np.array = matrix
        self.player_a_strategy_labels = player_a_strategy_labels
        self.player_b_strategy_labels = player_b_strategy_labels
        if player_a_strategy_labels is None:
            self.player_a_strategy_labels: list[LabelType] = [f"a{i + 1}" for i in range(matrix.shape[0])]
        if player_b_strategy_labels is None:
            self.player_b_strategy_labels: list[LabelType] = [f"b{i + 1}" for i in range(matrix.shape[1])]

    def __str__(self):
        strategy_table = PrettyTable(
            title="Таблица стратегий (игрока А)",
            field_names=("Стратегии", *self.player_b_strategy_labels, "MIN выигрыш A"),
        )
        # Добавляем стратегии игрока A со столбцом MIN выигрыша A.
        strategy_table.add_rows(
            rows=(
                (self.player_a_strategy_labels[i], *self.matrix[i], min_winning)
                for i, min_winning in enumerate(self.min_wins_player_a)
            )
        )

        # Добавляем заключительную строку с MAX проигрышами B.
        strategy_table.add_row(("MAX проигрыш B", *self.max_looses_player_b, " "))

        return str(strategy_table)

    def __repr__(self):
        """Для корректного отображения в Jupyter."""
        return str(self)

    def __eq__(self, other: "GameMatrix"):
        """Переопределение оператора `==` для экземпляров текущего класса."""
        return all(
            (
                (self.matrix == other.matrix).all(),
                self.player_a_strategy_labels == other.player_a_strategy_labels,
                self.player_b_strategy_labels == other.player_b_strategy_labels,
            )
        )

    @property
    def min_wins_player_a(self) -> np.array:
        """Значения MIN выигрыша игрока А в матричной игре."""
        return np.array([min(row) for row in self.matrix])

    @property
    def max_looses_player_b(self) -> np.array:
        """Значения MAX проигрыша игрока B в матричной игре."""
        return np.array([max(column) for column in self.matrix.T])

    @property
    def lowest_game_price(self) -> tuple[IndexType, ValueType]:
        """Возвращает индекс и значение нижней цены игры (максимин, max_j min_i c_{ij})"""
        low_price_index: IndexType = np.argmax(self.min_wins_player_a, axis=0)
        return low_price_index, self.min_wins_player_a[low_price_index]

    @property
    def highest_game_price(self) -> tuple[IndexType, ValueType]:
        """Возвращает индекс и значение верхней цены игры (минимакс, min_i max_j c_{ij})"""
        upper_price_index: IndexType = np.argmin(self.max_looses_player_b, axis=0)
        return upper_price_index, self.max_looses_player_b[upper_price_index]

    def normalize_matrix(self) -> None:
        """Приводит исходную матрицу к нормализованной (с неотрицательными коэффициентами) in-place."""
        min_element: ValueType = np.min(self.matrix)
        if min_element < 0:
            self.matrix += -min_element

    def drop_duplicate_strategies(self) -> None:
        """Удаляет дублирующиеся стратегии игроков A и B in-place."""
        # Удаление дубликатов строк.
        _, idx_rows = np.unique(self.matrix, axis=0, return_index=True)
        self.matrix = self.matrix[np.sort(idx_rows)]
        self.player_a_strategy_labels = [self.player_a_strategy_labels[i] for i in np.sort(idx_rows)]

        # Удаление дубликатов столбцов.
        _, idx_cols = np.unique(self.matrix, axis=1, return_index=True)
        self.matrix = self.matrix[:, np.sort(idx_cols)]
        self.player_b_strategy_labels = [self.player_b_strategy_labels[i] for i in np.sort(idx_cols)]

    def reduce_dimension(self, method="dominant_absorption") -> "GameMatrix":
        """
        Сводит к минимуму размерность матричной игры.
        :param str method: Выбор метода уменьшения размерности матричной игры:
            - `dominant_absorption` - поглощение доминируемых стратегий;
            - `nbr_drop` - удаление NBR-стратегий.
        :returns: Экземпляр GameMatrix с уменьшенной размерностью.
        """
        self.drop_duplicate_strategies()
        if method not in ("dominant_absorption", "nbr_drop"):
            exc_msg = f"Invalid mode of reducing matrix: {method}"
            raise MatrixGameException(exc_msg)

        reduced_matrix: GameMatrix = self._base_game_reduce(method=method)
        return reduced_matrix

    def _base_game_reduce(self, method="dominant_absorption") -> "GameMatrix":
        """
        Сводит к минимуму размерность матричной игры поглощением стратегий.
        :returns: Экземпляр GameMatrix с уменьшенной размерностью.
        """
        match method:
            case "dominant_absorption":
                useless_finder = self.__find_dominated_vectors
            case "nbr_drop":
                useless_finder = self.__find_nbr_vectors
            case _:
                exc_msg = f"Invalid mode of reducing matrix: {method}"
                raise MatrixGameException(exc_msg)

        # Поиск и удаление доминируемых строк.
        useless_rows: set[IndexType] = useless_finder(axis=0)
        if useless_rows:
            # Обновление индексов строк для удаления.
            useless_rows: list[IndexType] = sorted(useless_rows)
            # Удаление доминируемых строк.
            reduced_matrix: np.array = np.delete(self.matrix, useless_rows, axis=0)
            # Обновление соответствующих стратегий игрока A.
            player_a_strategy_labels = [
                label for i, label in enumerate(self.player_a_strategy_labels) if i not in useless_rows
            ]
            return GameMatrix(
                reduced_matrix,
                player_a_strategy_labels,
                self.player_b_strategy_labels,
            )._base_game_reduce(method=method)

        # Поиск и удаление доминируемых столбцов.
        useless_columns: set[IndexType] = useless_finder(axis=1)
        if useless_columns:
            # Обновление индексов столбцов для удаления.
            useless_columns: list[IndexType] = sorted(useless_columns)
            # Удаление доминируемых строк и столбцов.
            reduced_matrix: np.array = np.delete(self.matrix, useless_columns, axis=1)
            # Обновление соответствующих стратегий игрока B.
            player_b_strategy_labels = [
                label for i, label in enumerate(self.player_b_strategy_labels) if i not in useless_columns
            ]
            return GameMatrix(
                reduced_matrix,
                self.player_a_strategy_labels,
                player_b_strategy_labels,
            )._base_game_reduce(method=method)

        return copy.deepcopy(self)

    def __find_dominated_vectors(self, axis=0) -> set[IndexType]:
        """
        Возвращает множество индексов доминируемых строк (столбцов) в текущей матрице.
        :param int axis: Выбор координаты (строка/столбец); принимает 0 или 1.
            - axis = 0 (default) - нахождение доминирующих строк;
            - axis = 1 - нахождение доминирующих столбцов.
        :return: Множество индексов доминируемых срок/столбцов в текущей матрице.
        """
        match axis:
            case 0:
                # `>=` для строк; стратегии игрока A по строкам.
                comparison_op: ComparisonOperator = operator.ge
                player_strategy_labels: list[LabelType] = self.player_a_strategy_labels
                game_matrix: np.array = self.matrix
            case 1:
                # `<=` для столбцов; стратегии игрока B по столбцам.
                comparison_op: ComparisonOperator = operator.le
                player_strategy_labels: list[LabelType] = self.player_b_strategy_labels
                game_matrix: np.array = self.matrix.T
            case _:
                exc_msg = f"Invalid parameter `axis`: {axis}. Must be 0 or 1."
                raise MatrixGameException(exc_msg)

        dominated_vectors_indexes = set()
        # Поиск доминируемых векторов (строк/столбцов).
        for i in range(self.matrix.shape[axis]):
            for j in range(self.matrix.shape[axis]):
                if i == j or i in dominated_vectors_indexes:
                    continue

                if all(comparison_op(game_matrix[j], game_matrix[i])):
                    dominated_vectors_indexes.add(i)
                    msg = (
                        f"{player_strategy_labels[j]} ≻ {player_strategy_labels[i]}: "
                        f"поглощение стратегии {player_strategy_labels[i]} "
                        f"доминирующей стратегией {player_strategy_labels[j]}"
                    )
                    _logger.info(msg)

        return dominated_vectors_indexes

    def __find_nbr_vectors(self, axis=0) -> set[IndexType]:
        """
        Возвращает множество индексов NBR-строк (столбцов) в текущей матрице.
        :param int axis: Выбор координаты (строка/столбец); принимает 0 или 1.
            - axis = 0 (default) - нахождение NBR-строк;
            - axis = 1 - нахождение NBR-столбцов.
        :return: Множество индексов NBR-срок/столбцов в текущей матрице.
        """
        match axis:
            case 0:
                # `np.argmax` для выбора лучших стратегий игрока A.
                arg_extremum = np.argmax
                player_strategy_labels: list[LabelType] = self.player_a_strategy_labels
                game_matrix: np.array = self.matrix.T
            case 1:
                # `np.argmax` для выбора лучших стратегий игрока B.
                arg_extremum = np.argmin
                player_strategy_labels: list[LabelType] = self.player_b_strategy_labels
                game_matrix: np.array = self.matrix
            case _:
                exc_msg = f"Invalid parameter `axis`: {axis}. Must be 0 or 1."
                raise MatrixGameException(exc_msg)

        # Для фиксированной стратегии игрока A (B) смотрим, какие при этом стратегии использует игрок B (A).
        # Те из них, что он не использует - вычёркиваем (логично же, ну).
        player_used_strategies: set[IndexType] = {arg_extremum(vec) for vec in game_matrix}
        useless_strategies: set[IndexType] = set(range(game_matrix.shape[1])) - player_used_strategies
        if useless_strategies:
            msg = f"Удаление NBR-стратегий {[player_strategy_labels[i] for i in useless_strategies]}"
            _logger.info(msg)
        return useless_strategies
