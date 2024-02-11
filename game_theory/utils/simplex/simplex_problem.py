"""
Copyright 2020 Alexey Alexandrov

"""
import json
import logging
from pathlib import Path

import numpy as np

from .exceptions import SimplexProblemException
from .simplex_table import SimplexTable

_logger = logging.getLogger(__name__)


class SimplexProblem:
    """
    Класс для решения задачи ЛП симплекс-методом.
    """

    def __init__(self, input_path: Path):
        """
        Регистрирует входные данные из JSON-файла. Определяет условие задачи.
        :param input_path: Путь до JSON-файла с входными данными.
        """
        # Парсим JSON-файл с входными данными
        with input_path.open() as read_file:
            json_data = json.load(read_file)

        # Вектор-строка с - коэффициенты ЦФ.
        self.obj_func_coffs_ = np.array(json_data["obj_func_coffs"])
        # Матрица ограничений А.
        self.constraint_system_lhs_ = np.array(json_data["constraint_system_lhs"])
        # Вектор-столбец ограничений b.
        self.constraint_system_rhs_ = np.array(json_data["constraint_system_rhs"])
        # Направление задачи (min или max)
        self.func_direction_ = json_data["func_direction"]

        if len(self.constraint_system_rhs_) != self.constraint_system_rhs_.shape[0]:
            exc_msg = "Ошибка при вводе данных. Число строк в матрице" "и столбце ограничений не совпадает."
            raise SimplexProblemException(exc_msg)

        # Если задача на max, то меняем знаки ЦФ и направление задачи
        # (в конце возьмем решение со знаком минус и получим искомое).
        if self.func_direction_ == "max":
            self.obj_func_coffs_ *= -1

        # Инициализация симплекс-таблицы.
        self.simplex_table_ = SimplexTable(
            obj_func_coffs=self.obj_func_coffs_,
            constraint_system_lhs=self.constraint_system_lhs_,
            constraint_system_rhs=self.constraint_system_rhs_,
        )

    def __str__(self):
        """Условие задачи."""
        return "\n".join(
            (
                "Условие задачи:",
                "Найти вектор x = (x1,x2,..., xn)^T как решение след. задачи:",
                f"F = cx -> {self.func_direction_},",
                "Ax <= b,\nx1,x2, ..., xn >= 0",
                f"C = {self.obj_func_coffs_},",
                f"A =\n{self.constraint_system_lhs_},",
                f"b^T = {self.constraint_system_rhs_}.",
            )
        )

    def __repr__(self):
        """Условие задачи для отображения в Jupyter."""
        return str(self)

    def solve(self) -> float:
        """
        Запуск решения задачи.
        :returns: Значение целевой функции F после оптимизации.
        """
        _logger.info("Процесс решения:")
        self.reference_solution()
        self.optimal_solution()

        last_row_ind: int = self.simplex_table_.main_table_.shape[0] - 1
        return self.simplex_table_.main_table_[last_row_ind][0]

    # Этап 1. Поиск опорного решения.
    def reference_solution(self):
        """Поиск опорного решения."""
        _logger.info(
            "\n".join(
                (
                    "Поиск опорного решения:",
                    "Исходная симплекс-таблица:",
                    str(self.simplex_table_),
                )
            )
        )
        while not self.simplex_table_.is_find_ref_solution():
            self.simplex_table_.search_ref_solution()

        _logger.info("Опорное решение найдено!")
        self.__output_solution()

    def optimal_solution(self) -> None:
        """Поиск оптимального решения."""
        _logger.info("Поиск оптимального решения:")
        while not self.simplex_table_.is_find_opt_solution():
            self.simplex_table_.optimize_ref_solution()

        # Если задача на max, то в начале свели задачу к поиску min, а теперь
        # возьмём это решение со знаком минус и получим ответ для max.
        if self.func_direction_ == "max":
            table_rows_count: int = self.simplex_table_.main_table_.shape[0]
            self.simplex_table_.main_table_[table_rows_count - 1][0] *= -1

        _logger.info("Оптимальное решение найдено!")
        self.__output_solution()

    def __output_solution(self) -> None:
        """
        Метод выводит текущее решение.
        Используется для вывода опорного и оптимального решений.
        """
        fict_vars = self.simplex_table_.top_row_[2:]
        last_row_ind = self.simplex_table_.main_table_.shape[0] - 1

        # Output dummy variables =0 values.
        _logger.info("".join([*[f"{var} = " for var in fict_vars], "0, "]))

        # Output rest variables and its values.
        _logger.info(
            ", ".join(
                [
                    f"{self.simplex_table_.left_column_[i]} = {self.simplex_table_.main_table_[i][0]:.2f}"
                    for i in range(last_row_ind)
                ]
            )
        )

        _logger.info(
            "Целевая функция: F = %.2f",
            self.simplex_table_.main_table_[last_row_ind][0],
        )
