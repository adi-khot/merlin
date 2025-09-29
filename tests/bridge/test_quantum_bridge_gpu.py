import pytest
import torch
import perceval as pcvl

from merlin import QuantumLayer, OutputMappingStrategy
from merlin.bridge.QuantumBridge import QuantumBridge


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU not available"
)


def make_identity_layer(m: int, n_photons: int, device: torch.device) -> QuantumLayer:
    c = pcvl.Circuit(m)  # identity unitary
    return QuantumLayer(
        input_size=0,
        circuit=c,
        n_photons=n_photons,
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=True,
        device=device,
        dtype=torch.float32,
    )


def cpu_gpu_bridge_outputs(groups, statevec_fn, batch: int = 1):
    m = sum(2**g for g in groups)
    n_photons = len(groups)

    # CPU layer + bridge
    layer_cpu = make_identity_layer(m, n_photons, torch.device("cpu"))
    bridge_cpu = QuantumBridge(
        qubit_groups=groups,
        merlin_layer=layer_cpu,
        pl_state_fn=statevec_fn(torch.device("cpu")),
        wires_order="little",
        normalize=True,
        device=torch.device("cpu"),
    )

    # GPU layer + bridge
    device_gpu = torch.device("cuda")
    layer_gpu = make_identity_layer(m, n_photons, device_gpu)
    bridge_gpu = QuantumBridge(
        qubit_groups=groups,
        merlin_layer=layer_gpu,
        pl_state_fn=statevec_fn(device_gpu),
        wires_order="little",
        normalize=True,
        device=device_gpu,
    )

    x_cpu = torch.zeros(batch, 1)
    x_gpu = torch.zeros(batch, 1, device=device_gpu)

    out_cpu = bridge_cpu(x_cpu).detach().cpu()
    out_gpu = bridge_gpu(x_gpu)
    assert out_gpu.is_cuda, f"Expected output on CUDA, got {out_gpu.device}"
    return out_cpu, out_gpu.detach().cpu()


def test_bridge_gpu_matches_cpu_basis_state():
    groups = [1, 1]

    def statevec_fn(device):
        def _inner(_x: torch.Tensor):
            psi = torch.zeros(4, dtype=torch.complex64, device=device)
            psi[1] = 1.0 + 0.0j  # |01>
            return psi
        return _inner

    out_cpu, out_gpu = cpu_gpu_bridge_outputs(groups, statevec_fn, batch=1)
    assert out_cpu.shape == out_gpu.shape
    torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-5, atol=1e-6)


def test_bridge_gpu_matches_cpu_superposition_batched():
    groups = [1, 1]

    def statevec_fn(device):
        def _inner(x: torch.Tensor):
            psi = torch.zeros(4, dtype=torch.complex64, device=device)
            psi[0] = 1.0 + 0.0j
            psi[3] = 0.0 + 1.0j
            if isinstance(x, torch.Tensor) and x.dim() > 0 and x.shape[0] > 1:
                return psi.unsqueeze(0).expand(x.shape[0], -1)
            return psi
        return _inner

    out_cpu, out_gpu = cpu_gpu_bridge_outputs(groups, statevec_fn, batch=3)
    assert out_cpu.shape == out_gpu.shape
    torch.testing.assert_close(out_cpu, out_gpu, rtol=1e-5, atol=1e-6)
