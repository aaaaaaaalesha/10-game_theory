"""Подсказки типов для матричных игр."""

from typing import Annotated, Any, Callable, TypeAlias

from annotated_types import Gt

# Type to annotate values in game matrix.
ValueType: TypeAlias = int | float
# Type to annotate indexes in game matrix.
IndexType: TypeAlias = int
# Type to annotate labels of strategies for players.
LabelType: TypeAlias = str
# Type to annotate sizes or shapes.
SizeType: TypeAlias = Annotated[int, Gt(0)]
# Type to annotate comparison operator.
ComparisonOperator: TypeAlias = Callable[[Any, Any], bool]
