"""Training, evaluation and checkpointing for the pricing network."""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.data.simulate import FEATURE_COLUMNS, REFERENCE_YIELD, TARGET_COLUMN, YIELD_COLUMN
from src.models.pricing_arch import MacroCalibratedPricingNet
from src.utils.helpers import ensure_dir

CHAMPION_FILE = "champion.pt"
INTRADAY_FILE = "intraday.pt"


@dataclass
class Scaler:
    """Standardises features and target; expresses the yield relative to a reference."""

    feature_mean: List[float]
    feature_std: List[float]
    target_mean: float
    target_std: float
    yield_ref: float
    yield_scale: float

    @classmethod
    def fit(cls, frame):
        features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        std = features.std(axis=0)
        return cls(
            feature_mean=features.mean(axis=0).tolist(),
            feature_std=np.where(std == 0, 1.0, std).tolist(),
            target_mean=float(frame[TARGET_COLUMN].mean()),
            target_std=float(frame[TARGET_COLUMN].std()) or 1.0,
            yield_ref=REFERENCE_YIELD,
            yield_scale=float(frame[YIELD_COLUMN].std()) or 1.0,
        )

    def transform(self, frame):
        """Return (features, macro, target) as float32 arrays; target is None when absent."""
        features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
        x = (features - np.array(self.feature_mean)) / np.array(self.feature_std)
        macro = (frame[[YIELD_COLUMN]].to_numpy(dtype=float) - self.yield_ref) / self.yield_scale
        target = None
        if TARGET_COLUMN in frame.columns:
            target = (frame[[TARGET_COLUMN]].to_numpy(dtype=float) - self.target_mean) / self.target_std
            target = target.astype(np.float32)
        return x.astype(np.float32), macro.astype(np.float32), target

    def inverse_target(self, values):
        return np.asarray(values, dtype=float) * self.target_std + self.target_mean


def _select_parameters(model, params):
    if params == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return list(model.parameters())
    if params == "gamma":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        gamma = list(model.gamma_layer.parameters())
        for parameter in gamma:
            parameter.requires_grad_(True)
        return gamma
    raise ValueError(f"params must be 'all' or 'gamma', got {params!r}")


def train_model(frame, hidden_dim=32, epochs=300, lr=0.01, seed=0, model=None, scaler=None, params="all"):
    """Fit the network on `frame` and return (model, scaler).

    Pass an existing `model` and `scaler` with params="gamma" to recalibrate only
    the macro adjustment layer and leave the rest of the network frozen.
    """
    torch.manual_seed(seed)
    if scaler is None:
        scaler = Scaler.fit(frame)
    if model is None:
        model = MacroCalibratedPricingNet(input_dim=len(FEATURE_COLUMNS), hidden_dim=hidden_dim, output_dim=1)

    x, macro, target = scaler.transform(frame)
    if target is None:
        raise ValueError(f"Training data needs a '{TARGET_COLUMN}' column.")
    x, macro, target = torch.from_numpy(x), torch.from_numpy(macro), torch.from_numpy(target)

    optimiser = torch.optim.Adam(_select_parameters(model, params), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_fn(model(x, macro), target)
        loss.backward()
        optimiser.step()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model, scaler


def predict(model, scaler, frame):
    """Return predicted prices as a 1-D NumPy array."""
    x, macro, _ = scaler.transform(frame)
    model.eval()
    with torch.no_grad():
        scaled = model(torch.from_numpy(x), torch.from_numpy(macro)).numpy()
    return scaler.inverse_target(scaled).reshape(-1)


def evaluate(model, scaler, frame):
    """Return RMSE and MAE in price units."""
    errors = predict(model, scaler, frame) - frame[TARGET_COLUMN].to_numpy(dtype=float)
    return {"rmse": float(np.sqrt(np.mean(errors**2))), "mae": float(np.mean(np.abs(errors)))}


def save_checkpoint(path, model, scaler, metrics=None):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": asdict(scaler),
            "config": {"input_dim": len(FEATURE_COLUMNS), "hidden_dim": model.layer1.out_features},
            "metrics": metrics or {},
        },
        path,
    )
    return path


def load_checkpoint(path):
    """Return (model, scaler, metrics) from a checkpoint file."""
    payload = torch.load(Path(path), map_location="cpu")
    model = MacroCalibratedPricingNet(output_dim=1, **payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, Scaler(**payload["scaler"]), payload.get("metrics", {})


def load_latest_model(artifact_dir):
    """Load the intraday-recalibrated model if there is one, otherwise the champion.

    Returns (model, scaler, name). Raises FileNotFoundError when neither exists.
    """
    artifact_dir = Path(artifact_dir)
    for name in (INTRADAY_FILE, CHAMPION_FILE):
        if (artifact_dir / name).exists():
            model, scaler, _ = load_checkpoint(artifact_dir / name)
            return model, scaler, name
    raise FileNotFoundError(f"No model in {artifact_dir}. Run pipelines/weekly_retrain.py first.")
