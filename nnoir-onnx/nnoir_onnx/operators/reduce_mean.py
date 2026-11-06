from typing import Any, Dict, List, Optional, Tuple

import onnx
from nnoir.functions import Function, MulConstant, Sum
from numpy.typing import NDArray

from .utils import *


class OpReduceMean(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpReduceMean, self).__init__(node, *args)

        self.axes = None
        self.keepdims = True
        self.noop_with_empty_axes = False
        for attr in node.attribute:
            if attr.name == "axes":
                self.axes = attr.ints
            if attr.name == "keepdims":
                self.keepdims = attr.i > 0
            if attr.name == "noop_with_empty_axes":
                self.noop_with_empty_axes = attr.i > 0

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        x = self.node.input[0]
        [y] = self.node.output
        axes = self.axes
        if axes is None and len(self.node.input) > 1:
            # Opset >= 18: axes is given as an input tensor instead of an attribute
            axes_name = self.node.input[1]
            if axes_name in constants:
                axes = list(constants[axes_name])
            else:
                axes = list(env[axes_name])
        if axes is not None and len(axes) == 0:
            if self.noop_with_empty_axes:
                return [MulConstant([x], list(self.node.output), value=1.0)]  # type: ignore
            axes = None
        if axes is None:
            axes = tuple(range(len(env[x].shape)))

        n = 1
        for i in axes:
            n *= env[x].shape[i]

        internal_node = gen_unregisterd_node_name(env)
        register_node(env, internal_node, env[y])

        return [
            Sum(
                [x],
                [internal_node],  # type: ignore
                axes=list(map(int, axes)),
                keepdims=self.keepdims,
            ),
            MulConstant([internal_node], list(self.node.output), value=float(1.0 / n)),  # type: ignore
        ]
