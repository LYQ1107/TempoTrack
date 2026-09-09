import torch

from tempotrack_research.streaming.transport import unbalanced_sinkhorn


def test_uot_is_float32_log_domain_and_finite():
    cost = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float16)
    result = unbalanced_sinkhorn(cost, torch.ones(2), torch.ones(2), epsilon=.05, tau=.2, iterations=20)
    assert result.valid
    assert result.diagnostics["mode"] == "uot"
    assert result.diagnostics["epsilon"] == .05
