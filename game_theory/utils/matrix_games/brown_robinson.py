"""Итерационный численный метод Брауна-Робинсон решения (n x m)-игры с нулевой суммой."""
import logging
from pathlib import Path

import pandas as pd

from .game_matrix import GameMatrix

_logger = logging.getLogger(__name__)


class BrownRobinson:
    """Класс инкапсулирует решение матричной игры методом Брауна-Робинсон."""

    def __init__(self, game_matrix: GameMatrix):
        self.game_matrix: GameMatrix = game_matrix
        game_matrix.player_a_strategy_labels = [
            f"x{a_label[1:]}" for a_label in self.game_matrix.player_a_strategy_labels
        ]
        game_matrix.player_b_strategy_labels = [
            f"y{b_label[1:]}" for b_label in self.game_matrix.player_b_strategy_labels
        ]
        # Таблица итераций расчётов алгоритма Брауна-Робинсон.
        self.solution_table: pd.DataFrame = pd.DataFrame(
            columns=(
                "Итерация k",
                "Выбор игрока A",
                "Выбор игрока B",
                *game_matrix.player_a_strategy_labels,
                *game_matrix.player_b_strategy_labels,
                "Усреднённая оценка ВЦИ",
                "Усреднённая оценка НЦИ",
                "Погрешность ε",
            )
        )
        self.is_solved = False

    def solve(self, out_path: Path | None = None) -> pd.DataFrame:
        if self.is_solved:
            return self.__out_result(out_path)

        # TODO: implement it!
        self.is_solved = True
        return self.__out_result(out_path)

    def __out_result(self, out_path: Path | None = None) -> pd.DataFrame:
        # Экспортируем результат в CSV-таблицу, если передан путь.
        if isinstance(out_path, Path):
            self.solution_table.to_csv(out_path, index=False)

        return self.solution_table
