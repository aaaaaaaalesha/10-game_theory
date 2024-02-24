"""Сервисные функции для осуществления возврата к смешанным стратегиям исходной матричной игры."""
import logging

from prettytable import PrettyTable

from .game_matrix import GameMatrix
from .types import LabelType, ValueType

_logger = logging.getLogger(__name__)


def check_resulted_game_price(game_matrix: GameMatrix, game_price_value: ValueType) -> bool:
    """
    Проверяет, что полученное значение цены игры корректно
    (то есть лежит в отрезке между верхней и нижней ценами игры).
    :param GameMatrix game_matrix: Матрица игры, для которой рассматривается цена игры.
    :param ValueType game_price_value: Значение цены игры, которое нужно проверить.
    :returns: `True`, если цена игры лежит в отрезке, `False` - иначе.
    """
    _, lowest_game_price = game_matrix.lowest_game_price
    _, highest_game_price = game_matrix.highest_game_price
    resulted_game_price = game_price_value

    msg = f"Цена игры: {lowest_game_price} <= {resulted_game_price:.3f} <= {highest_game_price}"
    _logger.info(msg)
    return lowest_game_price <= resulted_game_price <= highest_game_price


def get_resulted_mixed_strategies(
    player_labels: list[LabelType],
    labels_to_probability: dict[LabelType, float],
    player_name="A",
) -> PrettyTable:
    """
    Расставляет смешанные стратегии игрока 'по местам' возвращаясь к исходной задаче
    (до уменьшения размерности матрицы игры).
    :param list[LabelType] player_labels: Метки стратегий игрока в исходной матричной игре.
    :param dict[LabelType, float] labels_to_probability: Словарь, отображающий метку стратегии игрока
    в соответствующее значение смешанной стратегии.
    :param player_name: Метка имени игрока (по умолчанию 'A').
    :return: Объект класса `PrettyTable` для вывода результата.
    """
    mixed_strategies_table = PrettyTable(
        title=f"Смешанные стратегии игрока {player_name}",
        field_names=player_labels,
        float_format=".2",
    )
    # Значения смешанных стратегий игрока.
    mixed_strategies_table.add_row([labels_to_probability.get(label, 0.0) for label in player_labels])
    return mixed_strategies_table
