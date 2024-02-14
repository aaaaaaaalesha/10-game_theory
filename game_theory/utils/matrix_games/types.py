"""Подсказки типов для матричных игр."""

from typing import Any, Callable, TypeAlias

# Type to annotate values in game matrix.
ValueType: TypeAlias = int | float
# Type to annotate indexes in game matrix.
IndexType: TypeAlias = int
# Type to annotate comparison operator.
ComparisonOperator: TypeAlias = Callable[[Any, Any], bool]
