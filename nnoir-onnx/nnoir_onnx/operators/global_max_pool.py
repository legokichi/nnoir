from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Function, MaxPooling2D
from numpy.typing import NDArray

from .utils import *


class OpGlobalMaxPool(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpGlobalMaxPool, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [x] = self.node.input

        _input = env[x]
        in_h = _input.shape[2]
        in_w = _input.shape[3]
        kh = in_h
        kw = in_w
        sy = 1
        sx = 1
        pad_h = (0, 0)
        pad_w = (0, 0)

        return [
            MaxPooling2D(
                list(self.node.input),
                list(self.node.output),
                kernel=(kh, kw),
                stride=(sy, sx),
                pad_h=pad_h,
                pad_w=pad_w,
            )
        ]
