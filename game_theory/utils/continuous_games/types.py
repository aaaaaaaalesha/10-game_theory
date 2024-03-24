"""Подсказки типов для непрерывных игр."""

from typing import Annotated, TypeAlias

from annotated_types import Gt

# Type to annotate values in game matrix.
ValueType: TypeAlias = int | float
# Type to annotate indexes in game matrix.
IndexType: TypeAlias = int
# Type to annotate sizes or shapes.
SizeType: TypeAlias = Annotated[int, Gt(0)]
