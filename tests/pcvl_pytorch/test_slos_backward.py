import torch

from merlin.pcvl_pytorch.slos_torchscript import build_slos_distribution_computegraph


def test_slos_compute_backward_supports_unitary_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    complex_dtype = torch.cfloat

    # small instance
    m = 4
    n_photons = 2

    # build graph
    graph = build_slos_distribution_computegraph(m, n_photons, dtype=dtype)

    # create batched random complex matrices and obtain unitary via QR
    B = 3
    # create random complex matrix with real and imag parts
    u = torch.randn((B, m, m), dtype=dtype, device=device) + 1j * torch.randn(
        (B, m, m), dtype=dtype, device=device
    )
    u = u.to(complex_dtype)
    # make unitary via batch QR
    q, r = torch.linalg.qr(u)

    # fix phases on r diagonal
    r_diag = torch.diagonal(r, dim1=-2, dim2=-1)
    safe_abs = torch.where(
        torch.abs(r_diag) == 0, torch.ones_like(r_diag), torch.abs(r_diag)
    )
    phases = r_diag / safe_abs
    q = q * phases.conj().unsqueeze(-2)

    # require grad on q
    q = q.requires_grad_(True)

    # simple input state: first n_photons modes are occupied
    input_state = [1] * n_photons + [0] * (m - n_photons)

    # compute probabilities for batch
    keys, amps = graph.compute(q, input_state)
    # amps shape: [B, K] where K is number of output states

    # create scalar loss (sum of probs) to allow backward
    probs = amps.real**2 + amps.imag**2
    loss = probs.sum()

    loss.backward()

    # check gradients exist on q
    assert q.grad is not None, "q.grad is None: backward did not populate gradients"
    # Check that gradient has non-zero entries (at least one)
    assert torch.any(q.grad.abs() > 1e-12).item(), "q.grad is all zeros"
