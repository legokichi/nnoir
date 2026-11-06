from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from nnoir.functions import AddConstant, ClippedReLU, Function, MulConstant
from numpy.typing import NDArray

from .utils import *

FLT_MAX = 3.4028234663852886e38


class OpClip(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpClip, self).__init__(node, *args)

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        if self.opset_version < 6:
            raise UnsupportedONNXOperation(self.node, "only opset_version >= 6 is supported")

        _min: Optional[float] = None
        _max: Optional[float] = None

        if self.opset_version >= 11:
            if len(self.node.input) >= 2 and self.node.input[1] != "":
                _min = float(np.asarray(constants[self.node.input[1]]).ravel()[0])

            if len(self.node.input) >= 3 and self.node.input[2] != "":
                _max = float(np.asarray(constants[self.node.input[2]]).ravel()[0])

        else:
            # opset_version 6
            for attr in self.node.attribute:
                if attr.name == "max":
                    _max = attr.f
                if attr.name == "min":
                    _min = attr.f

        x = self.node.input[0]
        outputs = list(self.node.output)

        has_min = _min is not None and _min > -FLT_MAX
        has_max = _max is not None and _max < FLT_MAX

        if not has_min and not has_max:
            return [MulConstant([x], outputs, value=1.0)]  # type: ignore

        if has_min and _min == 0.0:
            return [ClippedReLU([x], outputs, upper=_max if has_max else FLT_MAX)]  # type: ignore

        def gen_tmp() -> str:
            name = gen_unregisterd_node_name(env)
            register_node(env, name, env[x])
            return name

        if has_min:
            # Clip(x, min, max) = ClippedReLU(x - min, upper=max - min) + min
            upper = min(_max - _min, FLT_MAX) if has_max else FLT_MAX
            t0 = gen_tmp()
            t1 = gen_tmp()
            return [
                AddConstant([x], [t0], value=-_min),  # type: ignore
                ClippedReLU([t0], [t1], upper=upper),  # type: ignore
                AddConstant([t1], outputs, value=_min),  # type: ignore
            ]
        else:
            # Clip(x, -inf, max) = max - ClippedReLU(max - x)
            t0 = gen_tmp()
            t1 = gen_tmp()
            t2 = gen_tmp()
            t3 = gen_tmp()
            return [
                MulConstant([x], [t0], value=-1.0),  # type: ignore
                AddConstant([t0], [t1], value=_max),  # type: ignore
                ClippedReLU([t1], [t2], upper=FLT_MAX),  # type: ignore
                MulConstant([t2], [t3], value=-1.0),  # type: ignore
                AddConstant([t3], outputs, value=_max),  # type: ignore
            ]
