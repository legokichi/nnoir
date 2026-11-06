from typing import Any, Dict, List

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto
from onnx.helper import make_graph, make_model, make_node, make_opsetid, make_tensor_value_info
from onnx.numpy_helper import from_array
from util import Base

info = make_tensor_value_info


def test_reduce_mean_00() -> None:
    """
    opset version >= 1
    """

    class ReduceMeanTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceMean", inputs=["v0"], outputs=["v1"])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 1, 1, 1))]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceMeanTester({"v0": v0}, outputs).run()


def test_reduce_mean_01() -> None:
    """
    opset version >= 1
    """

    class ReduceMeanTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceMean", inputs=["v0"], outputs=["v1"], keepdims=0)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, [])]  # the shape is scalar

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph)
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceMeanTester({"v0": v0}, outputs).run()


def test_reduce_mean_02() -> None:
    """
    opset version >= 1 && < 18
    """

    class ReduceMeanTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceMean", inputs=["v0"], outputs=["v1"], axes=[1, 2])
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, [1, 1, 1, 5])]

            graph = make_graph([node], "add_graph", inputs, outputs)
            model = make_model(graph, opset_imports=[make_opsetid("", 13)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceMeanTester({"v0": v0}, outputs).run()


def test_reduce_mean_03() -> None:
    """
    opset version >= 18 (axes as an input)
    """

    class ReduceMeanTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceMean", inputs=["v0", "axes"], outputs=["v1"], keepdims=1)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3, 1, 1))]

            axes = from_array(np.array([2, 3], dtype=np.int64), "axes")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[axes])
            model = make_model(graph, opset_imports=[make_opsetid("", 18)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceMeanTester({"v0": v0}, outputs).run()


def test_reduce_mean_04() -> None:
    """
    opset version >= 18 (axes as an input, keepdims=0)
    """

    class ReduceMeanTester(Base):
        def create_onnx(self) -> onnx.ModelProto:
            node = make_node("ReduceMean", inputs=["v0", "axes"], outputs=["v1"], keepdims=0)
            inputs = [info("v0", TensorProto.FLOAT, (1, 3, 4, 5))]
            outputs = [info("v1", TensorProto.FLOAT, (1, 3))]

            axes = from_array(np.array([2, 3], dtype=np.int64), "axes")

            graph = make_graph([node], "add_graph", inputs, outputs, initializer=[axes])
            model = make_model(graph, opset_imports=[make_opsetid("", 18)])
            return model

    v0 = np.random.rand(1, 3, 4, 5).astype(np.float32)

    outputs = ["v1"]
    ReduceMeanTester({"v0": v0}, outputs).run()
