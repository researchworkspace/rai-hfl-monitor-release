import time

import torch
from captum.attr import IntegratedGradients

from raimon.fl import MLP, integrated_gradients


def test_matches_captum_and_scales_linearly():
    torch.manual_seed(0)
    model = MLP(20, 16).eval()
    x = torch.randn(300, 20)
    ref = IntegratedGradients(model).attribute(x, baselines=torch.zeros(1, 20), n_steps=32)
    ours = integrated_gradients(model, x, 32)
    assert torch.allclose(ours, ref.detach(), atol=1e-5, rtol=1e-4)
    t = time.time()
    integrated_gradients(MLP(150, 64).eval(), torch.randn(10_000, 150), 32)
    assert time.time() - t < 10
