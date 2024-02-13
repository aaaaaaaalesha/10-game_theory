"""Simplex exceptions."""


class SimplexProblemException(Exception):
    """Для решения прямой задачи симплекс-методом."""


class DualProblemException(SimplexProblemException):
    """Для решения двойственной задачи симплекс-методом."""


class AlreadySolvedException(SimplexProblemException):
    """Исключение для ограничения попытки повторного запуска алгоритмов."""
