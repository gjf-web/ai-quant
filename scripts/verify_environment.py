"""Offline smoke checks for the D2L environment and existing model."""

import platform
import sys
from importlib.metadata import distributions, version
from pathlib import Path

import numpy as np
import torch
from d2l import torch as d2l
from packaging.requirements import Requirement
from torchvision.ops import nms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import build_model


def main():
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    for package in ("torch", "torchvision", "d2l", "notebook", "ipykernel", "scikit-learn"):
        print(f"{package}: {version(package)}")

    # Check required distributions separately from the known torch WHEEL tag error.
    for distribution in distributions():
        for requirement_text in distribution.requires or []:
            requirement = Requirement(requirement_text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            installed = version(requirement.name)
            assert requirement.specifier.contains(installed, prereleases=True), (
                f"{distribution.metadata['Name']} requires {requirement}; found {installed}"
            )
    print("Installed dependency versions: PASS (optional extras excluded)")

    # d/dx sum(x_i^2) = 2x, an exact reference for automatic differentiation.
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x.square().sum()).backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    np.testing.assert_array_equal(x.detach().numpy(), [1.0, 2.0, 3.0])

    # Exercise torchvision's compiled extension, not just the Python import.
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]])
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]

    features = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    labels = np.array([0, 0, 1, 1])
    model = build_model()
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    print("CPU autograd, NumPy bridge, torchvision NMS and model fit: PASS")

    print(f"D2L default device: {d2l.try_gpu()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    available = torch.backends.mps.is_available()
    print(f"MPS available: {available}")
    if available:
        # float32 keeps this check compatible with Apple MPS.
        matrix = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
        gpu_matrix = matrix.to("mps").requires_grad_()
        result = gpu_matrix @ gpu_matrix.T
        torch.testing.assert_close(result.cpu(), matrix @ matrix.T, rtol=1e-4, atol=1e-5)
        gpu_matrix.square().sum().backward()
        torch.testing.assert_close(gpu_matrix.grad.cpu(), 2 * matrix, rtol=1e-4, atol=1e-5)
        print("MPS matrix multiplication and autograd vs CPU: PASS")
    print("Environment verification: PASS")


if __name__ == "__main__":
    main()
