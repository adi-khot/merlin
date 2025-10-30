import perceval as pcvl
import torch

class PhotonLossTransform(torch.nn.Module):
    """
    Linear map applying per-mode photon loss to a Fock probability vector.

    Args:
        simulation_keys: Iterable describing the raw Fock states produced by the
            simulator (as tuples or lists of integers).
        survival_probs: One survival probability per optical mode.
        dtype: Optional torch dtype for the transform matrix. Defaults to
            ``torch.float32``.
        device: Optional device used to stage the transform matrix.
    """

    def __init__(
            self,
            simulation_keys: list[tuple[int, ...]],
            survival_probs: list[float],
            *,
            dtype: torch.dtype | None = None,
            device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        #TODO initialize the PhotonLossTransform class

    def _build_transform(
            self,
    )
        #TODO for each Fock key, enumerate all possible loss outcomes per mode probability combination (*), constrained by the photon number in the key. Combine per-mode transforms via tensor products into a matrix L mapping simulated Fock states to post-loss states. Cache L and the corresponding loss-adjusted keys. Include a fast path returning an identity matrix when all p=1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[tuple[int, ...]]:
        #TODO

    @property
    def output_size(self) -> int:
        #TODO

    @property
    def is_identity(self) -> bool:
        #TODO

    def forward(self, distribution: torch.Tensor) -> torch.Tensor:
        """
        Apply the photon loss transform to a Fock probability vector.

        Args:
            distribution: A Fock probability vector as a 1D torch tensor.

        Returns:
            A Fock probability vector after photon loss.
        """
        #TODO apply the cached transform matrix to the input distribution



def resolve_photon_loss(experiment: pcvl.Experiment, n_modes: int) -> tuple[list[float], bool]:
    """Resolve photon loss from the experiment's noise model (and eventually Loss Channels).

    Args:
        experiment (pcvl.Experiment): The quantum experiment carrying the noise model.
        n_modes (int): Number of photonic modes to cover.

    Returns:
        list[float]: The survival probability for a photon in each mode.
    """
    survival_probs = [1.0] * n_modes  # Default: no loss
    empty_noise_model = True

    # TODO check experiement.noise for the NoiseModel, empty_noise_model = False if found

    # TODO check noise_model.brightness and noise_model.transmittance

    # TODO survival_probs = [brightness * transmittance] * n_modes

    return survival_probs, empty_noise_model
