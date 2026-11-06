from typing import Any, Dict, List

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_layer_normalization_00() -> None:
    """
    opset version >= 17, axis=-1
    """

    x_shape = (2, 16)

    class LayerNormalizationTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("LayerNormalization", inputs=["x", "scale", "bias"], outputs=["y"], axis=-1)

            inputs = [info("x", TensorProto.FLOAT, x_shape)]
            outputs = [info("y", TensorProto.FLOAT, x_shape)]

            scale = from_array((np.random.rand(16) + 0.5).astype(np.float32), "scale")
            bias = from_array((np.random.randn(16) * 0.1).astype(np.float32), "bias")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[scale, bias])
            return make_model(graph, opset_imports=[make_opsetid("", 17)])

    x = np.random.randn(*x_shape).astype(np.float32)
    inputs = {"x": x}
    outputs = ["y"]
    LayerNormalizationTester(inputs, outputs).run()


def test_layer_normalization_01() -> None:
    """
    opset version >= 17, axis=1 (normalize over multiple trailing axes)
    """

    x_shape = (2, 3, 4)

    class LayerNormalizationTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("LayerNormalization", inputs=["x", "scale", "bias"], outputs=["y"], axis=1)

            inputs = [info("x", TensorProto.FLOAT, x_shape)]
            outputs = [info("y", TensorProto.FLOAT, x_shape)]

            scale = from_array((np.random.rand(3, 4) + 0.5).astype(np.float32), "scale")
            bias = from_array((np.random.randn(3, 4) * 0.1).astype(np.float32), "bias")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[scale, bias])
            return make_model(graph, opset_imports=[make_opsetid("", 17)])

    x = np.random.randn(*x_shape).astype(np.float32)
    inputs = {"x": x}
    outputs = ["y"]
    LayerNormalizationTester(inputs, outputs).run()


def test_layer_normalization_02() -> None:
    """
    opset version >= 17, with Mean and InvStdDev outputs
    """

    x_shape = (2, 16)

    class LayerNormalizationTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node(
                "LayerNormalization", inputs=["x", "scale", "bias"], outputs=["y", "mean", "invstd"], axis=-1
            )

            inputs = [info("x", TensorProto.FLOAT, x_shape)]
            outputs = [
                info("y", TensorProto.FLOAT, x_shape),
                info("mean", TensorProto.FLOAT, (2,)),
                info("invstd", TensorProto.FLOAT, (2,)),
            ]

            scale = from_array((np.random.rand(16) + 0.5).astype(np.float32), "scale")
            bias = from_array((np.random.randn(16) * 0.1).astype(np.float32), "bias")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[scale, bias])
            return make_model(graph, opset_imports=[make_opsetid("", 17)])

    x = np.random.randn(*x_shape).astype(np.float32)
    inputs = {"x": x}
    outputs = ["y", "mean", "invstd"]
    LayerNormalizationTester(inputs, outputs).run()
