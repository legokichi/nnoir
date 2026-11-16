from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Concat, Constant, Function
from numpy.typing import NDArray

from .utils import *


class OpConcat(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpConcat, self).__init__(node, *args)

        self.axis = None
        for attr in node.attribute:
            if attr.name == "axis":
                self.axis = attr.i

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [y] = self.node.output
        axis = len(env[y].shape) + self.axis if self.axis < 0 else self.axis  # type: ignore
        functions: List[Function] = []
        inputs = []
        for inp in self.node.input:
            if inp in constants:
                name = gen_unregisterd_node_name(env)
                register_node(env, name, constants[inp])
                functions.append(Constant([], [name], value=constants[inp]))  # type: ignore
                inputs.append(name)
            else:
                inputs.append(inp)
        functions.append(Concat(inputs, list(self.node.output), axis=int(axis)))  # type: ignore
        return functions
