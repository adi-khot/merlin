import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def set_seed():
    seed = int(os.environ.get("PYTEST_SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="module")
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")
