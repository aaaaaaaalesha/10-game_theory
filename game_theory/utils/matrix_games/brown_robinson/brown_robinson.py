"""Итерационный численный метод Брауна-Робинсон решения (n x m)-игры с нулевой суммой."""
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from game_theory.utils.matrix_games.exceptions import MatrixGameException
from game_theory.utils.matrix_games.game_matrix import GameMatrix
from game_theory.utils.matrix_games.types import IndexType, LabelType, ValueType

from .labels import (
    ACCURACY_LABEL,
    CHOSEN_A_LABEL,
    CHOSEN_B_LABEL,
    ITERATION_LABEL,
    MAXMIN_ESTIMATION_LABEL,
    MINMAX_ESTIMATION_LABEL,
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
        self.minmax_price_estimation: float = max(self.player_a_strategy_values)
        self.maxmin_price_estimation: float = min(self.player_b_strategy_values)

        # Заполняем строку первой итерации метода.
        self.solution_table = pd.DataFrame.from_records(
            [
                {
                    ITERATION_LABEL: 1,
                    CHOSEN_A_LABEL: self.player_a_strategy_labels[self.player_a_strategy_index],
                    CHOSEN_B_LABEL: self.player_b_strategy_labels[self.player_b_strategy_index],
                    **dict(zip(self.player_a_strategy_labels, self.player_a_accumulated_values)),
                    **dict(zip(self.player_b_strategy_labels, self.player_b_accumulated_values)),
                    MINMAX_ESTIMATION_LABEL: self.minmax_price_estimation,
                    MAXMIN_ESTIMATION_LABEL: self.maxmin_price_estimation,
                    ACCURACY_LABEL: self.minmax_price_estimation - self.maxmin_price_estimation,
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

    @property
    def mixed_strategies(self) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
        """
        Возвращает кортеж смешанных стратегий игроков, если решение было найдено и
        (None, None), если задача ещё не была решена.
        """
        if not self.is_solved:
            return None, None

        iterations_count: int = len(self.solution_table)
        mixed_strategies_player_a: pd.Series = (
            self.solution_table[CHOSEN_A_LABEL].value_counts().sort_index() / iterations_count
        )
        mixed_strategies_player_b: pd.Series = (
            self.solution_table[CHOSEN_B_LABEL].value_counts().sort_index() / iterations_count
        )
        return tuple(mixed_strategies_player_a.values), tuple(mixed_strategies_player_b.values)

    @property
    def game_price_estimation(self) -> ValueType | None:
        """Оценка для цены игры на данной итерации алгоритма."""
        last_row: pd.Series = self.solution_table.iloc[-1]
        # В качестве оценки берём нижнюю цену игры.
        return last_row[MAXMIN_ESTIMATION_LABEL]

    def solve(
        self,
        mode="random",
        out: Path | None = None,
    ) -> pd.DataFrame:
        """
        Производит решение матричной игры за обоих игроков методом Брауна-Робинсон.
        :param out: Опциональный путь до файла для записи таблицы итераций алгоритма.
        :param str mode: Регламентирует, как выбирать стратегии в случае коллизий.
            - "random" (default) - использует случайную из лучших стратегий;
            - "previous" - использует стратегию с прошлой итерации.
        :return:
        """
        if self.is_solved:
            return self.__write_result(out)

        # Берём предыдущую строку итерации алгоритма.
        last_row: pd.Series = self.solution_table.iloc[-1]
        # Проделываем итерации алгоритма, пока не достигнем величины, меньшей ε.
        while last_row[ACCURACY_LABEL] > self.accuracy:
            # Производим очередную итерацию алгоритма.
            last_row: pd.Series = self.__perform_iteration(last_row, mode=mode)
            # Добавляем очередную строку в таблицу.
            self.solution_table.loc[len(self.solution_table)] = last_row

        self.is_solved = True
        return self.__write_result(out)

    def __write_result(self, out_path: Path | None = None) -> pd.DataFrame:
        # Округляем до 3 значащих цифр после запятой.
        self.solution_table = self.solution_table.round(3)
        # Экспортируем результат в CSV-таблицу, если передан путь.
        if isinstance(out_path, Path):
            self.solution_table.to_csv(out_path, index=False)

        return self.solution_table

    def __perform_iteration(self, previous_row: pd.Series, mode="random") -> pd.Series:
        """
        Производит очередную итерацию алгоритма Брауна-Робинсон.
        :param pd.Series previous_row: Строка предыдущей итерации алгоритма.
        :param str mode: Регламентирует, как выбирать стратегии в случае коллизий.
            - "random" (default) - использует случайную из лучших стратегий;
            - "previous" - использует стратегию с прошлой итерации.
        :return: Вычисленное значение ошибки (точности) ε на данной итерации.
        """
        # Выбираем лучшие стратегии игроков в новой итерации.
        (self.player_a_strategy_index, self.player_b_strategy_index) = self.__choose_strategies(mode=mode)
        # Добавляем выбранные стратегии к накопленным ранее.
        self.player_a_accumulated_values += self.player_a_strategy_values
        self.player_b_accumulated_values += self.player_b_strategy_values

        iteration_k: int = previous_row[ITERATION_LABEL] + 1
        # Храним минимальную верхнюю и максимальную нижнюю цены игры
        # (но в строке итерации выводим именно текущие оценки для ЦИ).
        high_price_estimate: float = max(self.player_a_accumulated_values) / iteration_k
        self.minmax_price_estimation = min(self.minmax_price_estimation, high_price_estimate)
        low_price_estimate: float = min(self.player_b_accumulated_values) / iteration_k
        self.maxmin_price_estimation = max(self.maxmin_price_estimation, low_price_estimate)

        # Добавляем строку новой итерации к таблице.
        return pd.Series(
            [
                iteration_k,
                self.player_a_strategy_labels[self.player_a_strategy_index],
                self.player_b_strategy_labels[self.player_b_strategy_index],
                *self.player_a_accumulated_values,
                *self.player_b_accumulated_values,
                high_price_estimate,
                low_price_estimate,
                self.minmax_price_estimation - self.maxmin_price_estimation,
            ],
            index=previous_row.index,
        )

    def __choose_strategies(self, mode="random") -> tuple[IndexType, IndexType]:
        """
        Выбирает индексы лучших стратегий игроков для следующей итерации алгоритма.
        :param str mode: Регламентирует, как выбирать стратегии в случае коллизий.
          - "random" (default) - использует случайную из лучших стратегий;
          - "previous" - использует стратегию с прошлой итерации.
        :return: Кортеж из двух индексов лучших стратегий для игроков A и B соответственно.
        """
        # Находим лучшие стратегии игрока A.
        max_strategy_indexes: np.ndarray[IndexType] = np.flatnonzero(
            self.player_a_accumulated_values == np.max(self.player_a_accumulated_values)
        )
        # Находим лучшие стратегии игрока B.
        min_strategy_indexes: np.ndarray[IndexType] = np.flatnonzero(
            self.player_b_accumulated_values == np.min(self.player_b_accumulated_values)
        )

        chosen_strategy_indexes = [self.player_a_strategy_index, self.player_b_strategy_index]
        for i, best_strategy_indexes in enumerate((max_strategy_indexes, min_strategy_indexes)):
            if len(best_strategy_indexes) == 1:
                chosen_strategy_indexes[i] = best_strategy_indexes[0]
                continue

            # Если происходит коллизия лучших стратегий, производим отбор
            match mode:
                case "previous":
                    # Ничего не делаем, оставляем предыдущую стратегию.
                    continue
                case "random":
                    # Если их несколько, рандомим между ними.
                    chosen_strategy_indexes[i] = random.choice(best_strategy_indexes)
                case _:
                    err_msg = f"Invalid mode {mode}"
                    raise MatrixGameException(err_msg)

        return tuple(chosen_strategy_indexes)
