"""Common device selection and checkpoint utilities."""
from pathlib import Path
import random

import numpy as np
import torch
from .models import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(request='auto'):
    if request == 'auto':
        if torch.cuda.is_available():
            request = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            request = 'mps'
        else:
            request = 'cpu'
    return torch.device(request)


def load_model(checkpoint_path, device):
    # Load only checkpoints you trust. Even weights_only prevents many, but not all, risks.
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    model = build_model(checkpoint['model_name'], checkpoint['base_channels'])
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, checkpoint
