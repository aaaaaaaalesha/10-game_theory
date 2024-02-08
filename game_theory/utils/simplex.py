"""
Copyright 2020 Alexey Alexandrov

Данный код был написан чёрт знает когда (см. выше).
Стиль написания не осуждать (хотя я cам осуждаю).
"""

import json

import numpy as np

from .exceptions import SimplexException
from .simplex_table import SimplexTable


class Simplex:
    """
    Класс для решения задачи ЛП симплекс-методом.
    """

    def __init__(self, path_to_file):
        """
        Регистрирует входные данные из JSON-файла. Определяет условие задачи.
        :param path_to_file: путь до JSON-файла с входными данными.
        """
        # Парсим JSON-файл с входными данными
        with open(path_to_file, 'r') as read_file:
            json_data = json.load(read_file)
            # Вектор-строка с - коэффициенты ЦФ.
            self.obj_func_coffs_ = np.array(json_data['obj_func_coffs'])
            # Матрица ограничений А.
            self.constraint_system_lhs_ = np.array(json_data['constraint_system_lhs'])
            # Вектор-столбец ограничений b.
            self.constraint_system_rhs_ = np.array(json_data['constraint_system_rhs'])
            # Направление задачи (min или max)
            self.func_direction_ = json_data['func_direction']

            if len(self.constraint_system_rhs_) != self.constraint_system_rhs_.shape[0]:
                raise SimplexException(
                    'Ошибка при вводе данных. Число строк в матрице' 
                    'и столбце ограничений не совпадает.'
                )

            # Если задача на max, то меняем знаки ЦФ и направление задачи
            # (в конце возьмем решение со знаком минус и получим искомое).
            if self.func_direction_ == 'max':
                self.obj_func_coffs_ *= -1

            # Инициализация симплекс-таблицы.
            self.simplex_table_ = SimplexTable(
                obj_func_coffs=self.obj_func_coffs_,
                constraint_system_lhs=self.constraint_system_lhs_,
                constraint_system_rhs=self.constraint_system_rhs_,
            )

    def __str__(self):
        """Условие задачи."""
        return '\n'.join((
            f'Условие задачи:',
            f'{"-" * 60}',
            f'Найти вектор x = (x1,x2,..., xn)^T как решение след. задачи:',
            f'F = cx -> {self.func_direction_},',
            f'Ax <= b,\nx1,x2, ..., xn >= 0',
            f'C = {self.obj_func_coffs_},',
            f'A =\n{self.constraint_system_lhs_},',
            f'b^T = {self.constraint_system_rhs_}.',
            f'{"-" * 60}',
        ))

    # Этап 1. Поиск опорного решения.
    def reference_solution(self):
        """Поиск опорного решения."""
        print('Процесс решения:\n1) Поиск опорного решения:')
        print('Исходная симплекс-таблица:', self.simplex_table_, sep='\n')
        while not self.simplex_table_.is_find_ref_solution():
            self.simplex_table_.search_ref_solution()

        print('Опорное решение найдено!')
        self.output_solution()

    # Этап 2. Поиск оптимального решения.
    def optimal_solution(self):
        """Метод производит отыскание оптимального решения."""
        print('2) Поиск оптимального решения:')
        while not self.simplex_table_.is_find_opt_solution():
            self.simplex_table_.optimize_ref_solution()

        # Если задача на max, то в начале свели задачу к поиску min, а теперь
        # возьмём это решение со знаком минус и получим ответ для max.
        if self.func_direction_ == 'max':
            table_rows_count: int = self.simplex_table_.main_table_.shape[0]
            self.simplex_table_.main_table_[table_rows_count - 1][0] *= -1

        print('Оптимальное решение найдено!')
        self.output_solution()

    def output_solution(self):
        """
        Метод выводит текущее решение, используется для вывода опорного и оптимального решений.
        """
        fict_vars = self.simplex_table_.top_row_[2:]
        last_row_ind = self.simplex_table_.main_table_.shape[0] - 1

        for var in fict_vars:
            print(var, '= ', end='')

        print(0, end=', ')

        for i in range(last_row_ind):
            print(
                self.simplex_table_.left_column_[i], 
                '= ', 
                round(self.simplex_table_.main_table_[i][0], 1), 
                end=', ',
            )

        print('\nЦелевая функция: F =', round(self.simplex_table_.main_table_[last_row_ind][0], 1))
