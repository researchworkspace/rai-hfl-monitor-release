import numpy as np
import torch
from torch import nn

from raimon.fl import MLP, get_vec, local_train_dp


def test_dp_sgd_without_noise_matches_clipped_autograd():
    """With sigma = 0 and q = 1, one step must equal mean of per-example clipped autograd gradients."""
    torch.manual_seed(0)
    model = MLP(5, hidden=8)
    start = get_vec(model)
    x = np.random.default_rng(0).standard_normal((16, 5)).astype(np.float32)
    y = (np.random.default_rng(1).random(16) > 0.5).astype(np.int64)
    upd = local_train_dp(model, start, x, y, q=1.0, steps=1, lr=0.5, clip_norm=0.3, sigma=0.0,
                         accountant=None, gen=torch.Generator().manual_seed(0))
    ref = MLP(5, hidden=8)
    torch.nn.utils.vector_to_parameters(start, ref.parameters())
    total = torch.zeros_like(start)
    for i in range(16):
        ref.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(ref(torch.from_numpy(x[i:i + 1])),
                                                              torch.tensor([float(y[i])]))
        loss.backward()
        g = torch.cat([p.grad.reshape(-1) for p in ref.parameters()])
        total += g * min(1.0, 0.3 / (g.norm().item() + 1e-6))
    expected = -0.5 * total / 16
    assert torch.allclose(upd, expected, atol=1e-5)


def test_training_does_not_modify_start_vector():
    model = MLP(5, hidden=8)
    start = get_vec(model)
    before = start.clone()
    x = np.random.default_rng(0).standard_normal((32, 5)).astype(np.float32)
    y = (np.random.default_rng(1).random(32) > 0.5).astype(np.int64)
    local_train_dp(model, start, x, y, q=0.5, steps=3, lr=0.5, clip_norm=1.0, sigma=1.0, accountant=None,
                   gen=torch.Generator().manual_seed(0))
    assert torch.equal(start, before)
