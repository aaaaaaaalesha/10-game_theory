import logging

import numpy as np
from prettytable import PrettyTable

from .types import IndexType, SizeType

_logger = logging.getLogger(__name__)


class BimatrixGame:
    def __init__(self, bimatrix: np.ndarray, title="Биматричная игра"):
        self.bimatrix = bimatrix
        self.__title = title

    @classmethod
    def from_random_values(cls, shape: tuple[SizeType, SizeType] = (10, 10), low=-100, high=100) -> "BimatrixGame":
        """Генерирует случайные пары значений от 0 до 100 для матрицы заданного размера."""
        first_values = np.random.random_integers(low, high, shape)  # noqa: NPY002
        second_values = np.random.random_integers(low, high, shape)  # noqa: NPY002
        return cls(
            np.array([list(zip(first_row, second_row)) for first_row, second_row in zip(first_values, second_values)]),
            title=f"Случайная биматричная игры размера {shape[0]}x{shape[1]}",
        )

    def __str__(self):
        digits = 3
        table = PrettyTable(
            title=self.__title,
            header=False,
            float_format=f".{digits}",
        )

        pareto_indices = self.pareto_indices
        nash_indices = self.nash_indices
        for i, row in enumerate(self.bimatrix):
            formatted_row = []
            for j, bivalue in enumerate(row):
                formatted_value = (round(bivalue[0], digits), round(bivalue[1], digits))
                if (i, j) in pareto_indices:
                    # Полужирное выделения для Парето.
                    formatted_value = "\033[1m" + str(formatted_value) + "\033[0m"
                if (i, j) in nash_indices:
                    # Подчёркивание для Нэша.
                    formatted_value = "\033[4m" + str(formatted_value) + "\033[0m"
                formatted_row.append(formatted_value)
            table.add_row(formatted_row)

        return (
            f"\033[1mЖирным\033[0m выделены ситуации, оптимальные по Парето\n"
            f"\033[4mПодчеркнутым\033[0m - ситуации, равновесные по Нэшу\n{table}"
        )

    def __repr__(self):
        return str(self)

    @property
    def pareto_indices(self) -> set[tuple[IndexType, IndexType]]:
        """Возвращает множество индексов ситуаций, оптимальных по Парето."""
        return {
            (i, j)
            for i in range(self.bimatrix.shape[0])
            for j in range(self.bimatrix.shape[1])
            if self.__is_pareto_optimal(i, j)
        }

    @property
    def nash_indices(self) -> set[tuple[IndexType, IndexType]]:
        """Возвращает множество индексов ситуаций, равновесных по Нэшу."""
        return {
            (i, j)
            for i in range(self.bimatrix.shape[0])
            for j in range(self.bimatrix.shape[1])
            if self.__is_nash_optimal(i, j)
        }

    def __is_pareto_optimal(self, first_index: IndexType, second_index: IndexType) -> bool:
        """Проверяет, является ли ситуация с данными индексами равновесной по Нэшу."""
        m, n, _ = self.bimatrix.shape
        first, second = self.bimatrix[first_index, second_index]
        return all(
            any(
                (
                    (self.bimatrix[i, j][0] < first or self.bimatrix[i, j][1] < second),
                    (self.bimatrix[i, j][0] <= first and self.bimatrix[i, j][1] <= second),
                )
            )
            for i in range(m)
            for j in range(n)
        )

    def __is_nash_optimal(self, first_index: IndexType, second_index: IndexType) -> bool:
        """Проверяет, является ли ситуация с данными индексами равновесной по Нэшу."""
        m, n, _ = self.bimatrix.shape
        first, second = self.bimatrix[first_index, second_index]
        if any(self.bimatrix[i][second_index][0] > first for i in range(m)):
            return False

        return all(self.bimatrix[first_index][j][1] <= second for j in range(n))
