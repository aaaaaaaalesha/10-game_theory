import logging

import numpy as np
from prettytable import PrettyTable

from .types import SizeType

_logger = logging.getLogger(__name__)


class BimatrixGame:
    def __init__(self, bimatrix: np.ndarray):
        self.bimatrix = bimatrix

    @classmethod
    def from_random_values(cls, shape: tuple[SizeType, SizeType] = (10, 10), low=-100, high=100) -> "BimatrixGame":
        """Генерирует случайные пары значений от 0 до 100 для матрицы заданного размера."""
        first_values = np.random.random_integers(low, high, shape)  # noqa: NPY002
        second_values = np.random.random_integers(low, high, shape)  # noqa: NPY002
        return cls(
            np.array([list(zip(first_row, second_row)) for first_row, second_row in zip(first_values, second_values)])
        )

    def __str__(self):
        float_num_digits = 3
        table = PrettyTable(
            header=False,
            float_format=f".{float_num_digits}",
        )
        table.add_rows(
            [(round(bivalue[0], float_num_digits), round(bivalue[1], float_num_digits)) for bivalue in row]
            for row in self.bimatrix
        )
        return str(table)

    def __repr__(self):
        return str(self)
