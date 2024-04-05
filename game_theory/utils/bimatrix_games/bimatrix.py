import logging
from typing import Any

import numpy as np
from prettytable import PrettyTable

from .types import IndexType, SizeType

_logger = logging.getLogger(__name__)


class BimatrixGame:
    ROUND_DIGITS: int = 3

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
        table = PrettyTable(
            title=self.__title,
            header=False,
            float_format=f".{self.ROUND_DIGITS}",
        )
        # Множества ситуаций, оптимальных по Парето и равновесных по Нэшу.
        pareto_indices: set = self.pareto_indices
        nash_indices: set = self.nash_indices
        for i, row in enumerate(self.bimatrix):
            formatted_row = []
            for j, bivalue in enumerate(row):
                formatted_value = (
                    round(bivalue[0], self.ROUND_DIGITS),
                    round(bivalue[1], self.ROUND_DIGITS),
                )
                if (i, j) in pareto_indices:
                    # Выделение полужирным курсивом для Парето.
                    formatted_value = self.__pareto_highlight(formatted_value)
                if (i, j) in nash_indices:
                    # Выделение подчёркиванием для Нэша.
                    formatted_value = self.__nash_highlight(formatted_value)
                formatted_row.append(formatted_value)
            table.add_row(formatted_row)

        return (
            f"{self.__pareto_highlight('Жирным курсивом')} выделены ситуации, оптимальные по Парето.\n"
            f"{self.__nash_highlight('Подчеркнутым')} - ситуации, равновесные по Нэшу.\n{table}\n\n"
            f"Равновесие Нэша: "
            f"{[tuple(self.bimatrix[i, j].astype(np.float16)) for i, j in nash_indices] if nash_indices else '--'}\n"
            f"Оптимальность Парето: "
            f"{[tuple(self.bimatrix[i, j].astype(np.float16)) for i, j in pareto_indices] if pareto_indices else '--'}"
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

    @staticmethod
    def __pareto_highlight(value: Any) -> str:
        """
        Возвращает строковое представление переданного значения,
        выделенного как Парето-оптимальное (жирным курсивом).
        """
        return f"\x1B[3m\033[1m{value}\033[0m\x1B[0m"

    @staticmethod
    def __nash_highlight(value: Any) -> str:
        """
        Возвращает строковое представление переданного значения,
        выделенного как равновесное по Нэшу (подчёркнуто).
        """
        return f"\033[4m{value}\033[0m"
