import torch
from torch.utils.data import DataLoader, TensorDataset

from cbct_n2n.models import UNet2D
from cbct_n2n.training.checkpoint import load_checkpoint, save_checkpoint
from cbct_n2n.training.trainer import train_one_epoch
import cbct_n2n.training.trainer as trainer_module


def test_checkpoint_round_trip_and_cpu_train_step(tmp_path):
    model = UNet2D(base_channels=4, depth=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    source = torch.randn(2, 1, 32, 32)
    target = source * 0.5
    loader = DataLoader(TensorDataset(source, target), batch_size=1)
    loss = train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    assert loss > 0
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, optimizer, None, 3, {"loss": loss}, {"x": 1})
    restored = UNet2D(base_channels=4, depth=2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = load_checkpoint(checkpoint, restored, restored_optimizer)
    assert state["epoch"] == 3
    assert state["config"] == {"x": 1}


def test_fit_resume_preserves_historical_best_checkpoint(tmp_path, monkeypatch):
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    config = {"training": {"epochs": 2, "amp": False, "patience": 2}}
    save_checkpoint(tmp_path / "best.pt", model, optimizer, None, 0, {"val_loss": 0.1}, config)
    monkeypatch.setattr(trainer_module, "train_one_epoch", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(trainer_module, "validate", lambda *args, **kwargs: 0.5)
    trainer_module.fit(model, [], [], optimizer, None, torch.device("cpu"), tmp_path, config, start_epoch=1)
    state = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert state["epoch"] == 0
    assert state["metrics"]["val_loss"] == 0.1
