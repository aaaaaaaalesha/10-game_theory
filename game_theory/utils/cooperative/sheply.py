import logging
from itertools import combinations
from math import factorial
from typing import Generator

from numpy import isclose

from .types import Coalition, SizeType, ValueType

_logger = logging.getLogger(__name__)


class CooperativeGame:
    def __init__(self, players_count: SizeType, char_values: tuple[ValueType, ...]) -> None:
        if len(char_values) != (coalitions_count := 2**players_count):
            err_msg = (
                f"The `char_values` length must be equal to "
                f"`2 ^ players_count = {coalitions_count}` but got {len(char_values)} instead"
            )
            raise ValueError(err_msg)

        self.players_count: SizeType = players_count
        self.total_coalition = tuple(range(1, players_count + 1))
        # Characteristic function mapping coalition to characteristic function value.
        self.__char_mapping: dict[Coalition, ValueType] = dict(zip(self.coalitions_generator, char_values))
        self.shapley_vector = None

    def char_function(self, coalition: Coalition) -> ValueType:
        return self.__char_mapping[coalition]

    @property
    def coalitions_generator(self) -> Generator[Coalition, None, None]:
        """Generates all combinations from Ø to total coalition in standard order."""
        player_labels = tuple(range(1, self.players_count + 1))
        yield from (coalition for k in range(self.players_count + 1) for coalition in combinations(player_labels, k))

    @property
    def coalitions_pairs_generator(self) -> Generator[tuple[Coalition, Coalition], None, None]:
        yield from (coalition_pair for coalition_pair in combinations(self.coalitions_generator, 2))

    def is_superadditive_game(self) -> bool:
        v = self.char_function
        return all(
            (
                v(tuple(first_subset | second_subset)) >= v(first_subcoalition) + v(second_subcoalition)
                for first_subcoalition, second_subcoalition in self.coalitions_pairs_generator
                # `first_subcoalition` and `second_subcoalition` are disjoint.
                if not ((first_subset := set(first_subcoalition)) & (second_subset := set(second_subcoalition)))
            )
        )

    def is_convex(self) -> tuple[bool, tuple | None]:
        v = self.char_function
        for first_subcoalition, second_subcoalition in self.coalitions_pairs_generator:
            first_subset, second_subset = set(first_subcoalition), set(second_subcoalition)
            union_coalition = tuple(first_subset | second_subset)
            intersection_coalition = tuple(first_subset & second_subset)
            if v(union_coalition) + v(intersection_coalition) < v(first_subcoalition) + v(second_subcoalition):
                msg = (
                    f"Игра не является выпуклой, так как, к примеру, "
                    f"для коалиций S = {first_subset} и Т = {second_subset} имеем\n"
                    f"v({union_coalition}) + v({intersection_coalition}) < v({first_subset}) + v({second_subset})\n"
                    f"{v(union_coalition)} + {v(intersection_coalition)} "
                    f"< {v(first_subcoalition)} + {v(second_subcoalition)}"
                )
                _logger.info(msg)
                return False, (first_subcoalition, second_subcoalition)

        return True, None

    def get_shapley_vector(self):
        if not self.is_superadditive_game():
            err_msg = "Game is not superadditive"
            raise ValueError(err_msg)

        if self.shapley_vector:
            return self.shapley_vector

        n = self.players_count
        v = self.char_function
        multiplier = 1 / factorial(n)
        self.shapley_vector = tuple(
            multiplier
            * sum(
                factorial(len(s) - 1) * factorial(n - len(s)) * (v(s) - v(tuple(set(s) - {i})))
                for s in self.coalitions_generator
                if i in s
            )
            for i in self.total_coalition
        )
        return self.shapley_vector

    def is_individual_rationalization(self):
        return isclose(sum(self.shapley_vector), self.char_function(self.total_coalition))

    def is_group_rationalization(self):
        return all(self.shapley_vector[i - 1] >= self.char_function((i,)) for i in self.total_coalition)
