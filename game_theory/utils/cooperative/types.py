from typing import Annotated, TypeAlias

from annotated_types import Gt

Coalition: TypeAlias = tuple[int, ...]
SizeType: TypeAlias = Annotated[int, Gt(0)]
ValueType: TypeAlias = int
