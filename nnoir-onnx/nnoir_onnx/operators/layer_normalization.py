from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx
from nnoir.functions import (
    Add,
    AddConstant,
    Constant,
    Div,
    Function,
    Mul,
    MulConstant,
    Reshape,
    Sqrt,
    Sub,
    Sum,
)
from numpy.typing import NDArray

from .utils import *


class OpLayerNormalization(Op):
    def __init__(self, node: onnx.NodeProto, *args: Any):
        super(OpLayerNormalization, self).__init__(node, *args)

        self.axis = -1
        self.epsilon = 1e-5
        for attr in node.attribute:
            if attr.name == "axis":
                self.axis = attr.i
            if attr.name == "epsilon":
                self.epsilon = attr.f

    def to_function(self, env: Dict[str, NDArray[Any]], constants: Dict[str, NDArray[Any]]) -> List[Function]:
        x = self.node.input[0]
        scale_name = self.node.input[1] if len(self.node.input) >= 2 and self.node.input[1] != "" else None
        bias_name = self.node.input[2] if len(self.node.input) >= 3 and self.node.input[2] != "" else None

        if scale_name is not None and scale_name not in constants:
            raise UnsupportedONNXOperation(self.node, "scale must be constant")
        if bias_name is not None and bias_name not in constants:
            raise UnsupportedONNXOperation(self.node, "bias must be constant")

        x_shape = env[x].shape
        rank = len(x_shape)
        axis = self.axis + rank if self.axis < 0 else self.axis
        axes = list(range(axis, rank))

        n = 1
        for i in axes:
            n *= x_shape[i]

        keepdims_shape = tuple(1 if i in axes else s for i, s in enumerate(x_shape))

        def gen_tmp(shape: Tuple[int, ...]) -> str:
            name = gen_unregisterd_node_name(env)
            register_node(env, name, np.zeros(shape).astype(np.float32))
            return name

        functions: List[Function] = []

        # mean = Sum(x, axes) / N
        t_sum = gen_tmp(keepdims_shape)
        t_mean = gen_tmp(keepdims_shape)
        functions.append(Sum([x], [t_sum], axes=axes, keepdims=True))  # type: ignore
        functions.append(MulConstant([t_sum], [t_mean], value=float(1.0 / n)))  # type: ignore

        # d = x - mean
        t_d = gen_tmp(tuple(x_shape))
        functions.append(Sub([x, t_mean], [t_d]))  # type: ignore

        # var = Sum(d * d, axes) / N
        t_d2 = gen_tmp(tuple(x_shape))
        t_svar = gen_tmp(keepdims_shape)
        t_var = gen_tmp(keepdims_shape)
        functions.append(Mul([t_d, t_d], [t_d2]))  # type: ignore
        functions.append(Sum([t_d2], [t_svar], axes=axes, keepdims=True))  # type: ignore
        functions.append(MulConstant([t_svar], [t_var], value=float(1.0 / n)))  # type: ignore

        # x_hat = d / sqrt(var + epsilon)
        t_vare = gen_tmp(keepdims_shape)
        t_std = gen_tmp(keepdims_shape)
        t_hat = gen_tmp(tuple(x_shape))
        functions.append(AddConstant([t_var], [t_vare], value=float(self.epsilon)))  # type: ignore
        functions.append(Sqrt([t_vare], [t_std]))  # type: ignore
        functions.append(Div([t_d, t_std], [t_hat]))  # type: ignore

        # y = x_hat * scale (+ bias)
        current = t_hat
        if scale_name is not None:
            t_scale = gen_unregisterd_node_name(env)
            register_node(env, t_scale, constants[scale_name])
            t_scaled = gen_tmp(tuple(x_shape))
            functions.append(Constant([], [t_scale], value=constants[scale_name]))  # type: ignore
            functions.append(Mul([current, t_scale], [t_scaled]))  # type: ignore
            current = t_scaled

        y = self.node.output[0]
        if bias_name is not None:
            # Note: Bias is not used here because its `b` parameter is stored
            # encoded and only decoded on load, which breaks in-memory runs.
            t_bias = gen_unregisterd_node_name(env)
            register_node(env, t_bias, constants[bias_name])
            functions.append(Constant([], [t_bias], value=constants[bias_name]))  # type: ignore
            functions.append(Add([current, t_bias], [y]))  # type: ignore
        else:
            functions.append(MulConstant([current], [y], value=1.0))  # type: ignore

        # Optional Mean output
        if len(self.node.output) > 1 and self.node.output[1] != "" and self.node.output[1] in env:
            functions.append(
                Reshape([t_mean], [self.node.output[1]], shape=list(map(int, env[self.node.output[1]].shape)))  # type: ignore
            )

        # Optional InvStdDev output
        if len(self.node.output) > 2 and self.node.output[2] != "" and self.node.output[2] in env:
            t_one = gen_unregisterd_node_name(env)
            register_node(env, t_one, np.ones(keepdims_shape).astype(np.float32))
            t_inv = gen_tmp(keepdims_shape)
            functions.append(Constant([], [t_one], value=np.ones(keepdims_shape).astype(np.float32)))  # type: ignore
            functions.append(Div([t_one, t_std], [t_inv]))  # type: ignore
            functions.append(
                Reshape([t_inv], [self.node.output[2]], shape=list(map(int, env[self.node.output[2]].shape)))  # type: ignore
            )

        return functions
