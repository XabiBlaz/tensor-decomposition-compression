"""Fixed-shape ONNX export with numerical parity before advertising the artifact."""

from pathlib import Path

import torch

from .tasks.vision import evaluation_mode


def export_onnx(model, inputs, path, *, rtol=1e-4, atol=1e-5):
    import numpy as np
    import onnx
    import onnxruntime as ort
    if not isinstance(inputs, torch.Tensor):
        raise ValueError("ONNX export currently supports tensor-in/tensor-out vision adapters.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_mode(model), torch.no_grad():
        expected = model(inputs)
        if not isinstance(expected, torch.Tensor):
            raise ValueError("This output contract needs its own ONNX adapter.")
        torch.onnx.export(model, inputs, str(path), input_names=["images"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": inputs.detach().cpu().numpy()})[0]
    np.testing.assert_allclose(actual, expected.detach().cpu().numpy(), rtol=rtol, atol=atol)
    return {"status": "verified", "runtime": "onnxruntime_cpu", "shape": list(inputs.shape),
            "rtol": rtol, "atol": atol, "max_absolute_error": float(np.abs(actual - expected.cpu().numpy()).max()),
            "artifact_bytes": path.stat().st_size}
