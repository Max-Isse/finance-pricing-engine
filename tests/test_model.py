import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.simulate import TARGET_COLUMN, generate_pricing_data  # noqa: E402
from src.models.pricing_arch import MacroCalibratedPricingNet  # noqa: E402
from src.models.train import (  # noqa: E402
    evaluate,
    load_checkpoint,
    load_latest_model,
    predict,
    save_checkpoint,
    train_model,
)


def test_forward_returns_one_price_per_row():
    # Regression test: the network used to return one value per hidden unit, shape (n, 16).
    model = MacroCalibratedPricingNet(input_dim=5, hidden_dim=16, output_dim=1)
    features = torch.from_numpy(np.random.default_rng(0).normal(size=(10, 5)).astype(np.float32))
    yields = torch.ones(10, 1)
    assert tuple(model(features, yields).shape) == (10, 1)


def test_forward_accepts_one_dimensional_yields():
    model = MacroCalibratedPricingNet(input_dim=5, hidden_dim=8)
    assert tuple(model(torch.ones(4, 5), torch.ones(4)).shape) == (4, 1)


def test_gradient_reaches_the_macro_input():
    model = MacroCalibratedPricingNet(input_dim=5, hidden_dim=8)
    yields = torch.ones(4, 1, requires_grad=True)
    model(torch.ones(4, 5), yields).sum().backward()
    assert yields.grad is not None
    assert tuple(yields.grad.shape) == (4, 1)


def test_training_beats_a_mean_predictor():
    frame = generate_pricing_data(n_rows=2500, seed=1)
    train, holdout = frame.iloc[:2000], frame.iloc[2000:]
    model, scaler = train_model(train, epochs=200, seed=0)
    rmse = evaluate(model, scaler, holdout)["rmse"]
    assert rmse < 0.5 * holdout[TARGET_COLUMN].std()


def test_gamma_only_training_leaves_the_base_network_unchanged():
    frame = generate_pricing_data(n_rows=600, seed=2)
    model, scaler = train_model(frame, epochs=20, seed=0)
    base_before = model.layer1.weight.detach().numpy().copy()
    gamma_before = model.gamma_layer.weight.detach().numpy().copy()
    train_model(frame, model=model, scaler=scaler, params="gamma", epochs=20)
    assert np.array_equal(model.layer1.weight.detach().numpy(), base_before)
    assert not np.array_equal(model.gamma_layer.weight.detach().numpy(), gamma_before)
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_checkpoint_round_trip_gives_identical_predictions(tmp_path):
    frame = generate_pricing_data(n_rows=400, seed=3)
    model, scaler = train_model(frame, epochs=10, seed=0)
    save_checkpoint(tmp_path / "champion.pt", model, scaler, {"rmse": 1.0})
    loaded, loaded_scaler, metrics = load_checkpoint(tmp_path / "champion.pt")
    assert metrics == {"rmse": 1.0}
    assert np.allclose(predict(model, scaler, frame), predict(loaded, loaded_scaler, frame))
    assert load_latest_model(tmp_path)[2] == "champion.pt"


def test_load_latest_model_raises_without_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_latest_model(tmp_path)
