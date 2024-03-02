"""Итерационный численный метод Брауна-Робинсон решения (n x m)-игры с нулевой суммой."""
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from game_theory.utils.matrix_games.game_matrix import GameMatrix
from game_theory.utils.matrix_games.types import IndexType, LabelType, ValueType

from .labels import (
    ACCURACY_LABEL,
    CHOSEN_A_LABEL,
    CHOSEN_B_LABEL,
    HIGHEST_ESTIMATION_LABEL,
    ITERATION_LABEL,
    LOWEST_ESTIMATION_LABEL,
)

_logger = logging.getLogger(__name__)


class BrownRobinson:
    """Класс инкапсулирует решение матричной игры методом Брауна-Робинсон."""

    def __init__(self, game_matrix: GameMatrix, accuracy: float = 0.1):
        self.game_matrix: GameMatrix = game_matrix

        # Случайным образом производим выбор игроков A и B.
        self.player_a_strategy_index: IndexType = random.randint(0, game_matrix.shape[0] - 1)
        self.player_b_strategy_index: IndexType = random.randint(0, game_matrix.shape[1] - 1)

        # Накопленные значения выигрыша/проигрыша игроков A и B.
        self.player_a_accumulated_values: np.ndarray[ValueType] = self.player_a_strategy_values
        self.player_b_accumulated_values: np.ndarray[ValueType] = self.player_b_strategy_values

        # Текущая усреднённая верхняя и нижняя цены игры (ВЦИ и НЦИ).
        self.highest_price_estimation: float = max(self.player_a_strategy_values)
        self.lowest_price_estimation: float = min(self.player_b_strategy_values)

        # Заполняем строку первой итерации метода.
        self.solution_table = pd.DataFrame.from_records(
            [
                {
                    ITERATION_LABEL: 1,
                    CHOSEN_A_LABEL: self.player_a_strategy_labels[self.player_a_strategy_index],
                    CHOSEN_B_LABEL: self.player_b_strategy_labels[self.player_b_strategy_index],
                    **dict(zip(self.player_a_strategy_labels, self.player_a_accumulated_values)),
                    **dict(zip(self.player_b_strategy_labels, self.player_b_accumulated_values)),
                    HIGHEST_ESTIMATION_LABEL: self.highest_price_estimation,
                    LOWEST_ESTIMATION_LABEL: self.lowest_price_estimation,
                    ACCURACY_LABEL: self.highest_price_estimation - self.lowest_price_estimation,
                }
            ]
        )

        # Точность вычислений численного метода.
        self.accuracy: float = accuracy
        # Метка решённой задачи.
        self.is_solved = False

    @property
    def player_a_strategy_labels(self) -> list[LabelType]:
        """Возвращает список меток стратегий игрока A."""
        return self.game_matrix.player_a_strategy_labels

    @property
    def player_b_strategy_labels(self) -> list[LabelType]:
        """Возвращает список меток стратегий игрока B."""
        return self.game_matrix.player_b_strategy_labels

    @property
    def player_a_strategy_values(self) -> np.ndarray[ValueType]:
        """Значения выигрышей игрока A при текущей стратегии."""
        return self.game_matrix.matrix.T[self.player_b_strategy_index].copy()

    @property
    def player_b_strategy_values(self) -> np.ndarray[ValueType]:
        """Значения проигрышей игрока B при текущей стратегии."""
        return self.game_matrix.matrix[self.player_a_strategy_index].copy()

    def solve(self, out_path: Path | None = None) -> pd.DataFrame:
        if self.is_solved:
            return self.__out_result(out_path)

        prev_row: pd.Series = self.solution_table.iloc[-1]
        while prev_row[ACCURACY_LABEL] > self.accuracy:
            prev_row: pd.Series = self.__perform_iteration(prev_row)

        self.is_solved = True
        return self.__out_result(out_path)

    def __out_result(self, out_path: Path | None = None) -> pd.DataFrame:
        # Экспортируем результат в CSV-таблицу, если передан путь.
        if isinstance(out_path, Path):
            self.solution_table.to_csv(out_path, index=False)

        return self.solution_table

    def __perform_iteration(self, row: pd.Series) -> pd.Series:
        # Находим лучшие стратегии игроков A и B.
        max_price_indexes: np.ndarray[IndexType] = np.flatnonzero(
            self.player_a_accumulated_values == np.max(self.player_a_accumulated_values)
        )
        min_price_indexes: np.ndarray[IndexType] = np.flatnonzero(
            self.player_b_accumulated_values == np.min(self.player_b_accumulated_values)
        )
        # Если их несколько, рандомим между ними.
        self.player_a_strategy_index = (
            random.choice(max_price_indexes) if len(max_price_indexes) > 1 else max_price_indexes[0]
        )
        self.player_b_strategy_index = (
            random.choice(min_price_indexes) if len(min_price_indexes) > 1 else min_price_indexes[0]
        )

        # Добавляем выбранные стратегии к накопленным ранее.
        self.player_a_accumulated_values += self.player_a_strategy_values
        self.player_b_accumulated_values += self.player_b_strategy_values

        iteration_k: int = row[ITERATION_LABEL] + 1
        # Храним минимальную верхнюю и максимальную нижнюю цены игры
        # (но в строке итерации выводим именно текущие оценки для ЦИ).
        high_price_estimate: float = max(self.player_a_accumulated_values) / iteration_k
        low_price_estimate: float = min(self.player_b_accumulated_values) / iteration_k
        self.highest_price_estimation = min(row[HIGHEST_ESTIMATION_LABEL], high_price_estimate)
        self.lowest_price_estimation = max(row[LOWEST_ESTIMATION_LABEL], low_price_estimate)
        # Наполняем строку итерации.
        new_row = pd.Series(
            [
                iteration_k,
                self.player_a_strategy_labels[self.player_a_strategy_index],
                self.player_b_strategy_labels[self.player_b_strategy_index],
                *self.player_a_accumulated_values,
                *self.player_b_accumulated_values,
                high_price_estimate,
                low_price_estimate,
                self.highest_price_estimation - self.lowest_price_estimation,
            ],
            index=row.index,
        )
        # Добавляем строку к таблице и возвращаем её.
        self.solution_table.loc[len(self.solution_table)] = new_row
        return new_row
