from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import Concat, Function, Reshape, Slice
from numpy.typing import NDArray

from .utils import *


class OpGather(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpGather, self).__init__(node, *args)

        self.axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                self.axis = attr.i

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        [v0, indices] = self.node.input
        [v1] = self.node.output
        v0_shape = env[v0].shape
        rank = len(v0_shape)
        axis = self.axis + rank if self.axis < 0 else self.axis

        if indices not in constants:
            raise UnsupportedONNXOperation(self.node, "indices must be constant")

        indices_v = np.asarray(constants[indices])
        if indices_v.dtype not in (np.int64, np.int32):
            raise UnsupportedONNXOperation(self.node, "indices must be integer")
        indices_v = indices_v.astype(np.int64)

        dim = v0_shape[axis]
        flat_indices = [int(i) + dim if i < 0 else int(i) for i in indices_v.ravel()]
        if len(flat_indices) == 0:
            raise UnsupportedONNXOperation(self.node, "indices must not be empty")

        # Slice the data at each index, keeping the gathered axis as length 1.
        functions: List[Function] = []
        slice_names = []
        for index in flat_indices:
            slice_shape = list(v0_shape)
            slice_shape[axis] = 1
            slice_name = gen_unregisterd_node_name(env)
            register_node(env, slice_name, np.zeros(slice_shape).astype(np.float32))

            starts = [0] * rank
            starts[axis] = index
            ends = list(v0_shape)
            ends[axis] = index + 1

            functions.append(Slice([v0], [slice_name], starts=starts, ends=ends))  # type: ignore
            slice_names.append(slice_name)

        if len(slice_names) == 1:
            gathered_name = slice_names[0]
        else:
            gathered_shape = list(v0_shape)
            gathered_shape[axis] = len(flat_indices)
            gathered_name = gen_unregisterd_node_name(env)
            register_node(env, gathered_name, np.zeros(gathered_shape).astype(np.float32))
            functions.append(Concat(slice_names, [gathered_name], axis=int(axis)))  # type: ignore

        functions.append(
            Reshape([gathered_name], list(self.node.output), shape=list(map(int, env[v1].shape)))  # type: ignore
        )
        return functions
