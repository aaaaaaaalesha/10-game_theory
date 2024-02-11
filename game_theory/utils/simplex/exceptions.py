"""Simplex exceptions."""


class SimplexProblemException(Exception):
    """Для решения прямой задачи симплекс-методом."""


class DualProblemException(SimplexProblemException):
    """Для решения двойственной задачи симплекс-методом."""
