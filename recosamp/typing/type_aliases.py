from typing import TypeAlias

from torch import Tensor


IntTensorDict: TypeAlias = dict[int, Tensor]

StringTensorDict: TypeAlias = dict[str, Tensor]
