from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
FINAL_CHAIN_CONFIGS = (
    "air3_mode2_1609_unet3d_h2f_128_r0.yaml",
    "air3_mode2_1609_unet3d_h2f_128b_r1.yaml",
    "air3_mode2_1609_unet3d_h2f_128b_r2.yaml",
    "air3_mode2_1609_unet3d_h2f_128b_r3.yaml",
    "air3_mode2_1609_unet3d_h2f_128b_r4.yaml",
    "1609v2_unet3d_h2f_128.yaml",
    "1609v2_unet3d_h2f_128b_r1.yaml",
    "1609v2_unet3d_h2f_128b_r2.yaml",
    "1609v2_unet3d_h2f_128b_r3.yaml",
)

#: AIReduceNoise3 scheme-B chain that ships: r0 -> r1b -> r2b -> r3b -> r4b
AIR3_RECURSIVE_ROUNDS = (1, 2, 3, 4)


def test_air3_r0_config_uses_final_manifest_four_domains_and_gpu0():
    config = yaml.safe_load(
        (ROOT / "configs" / "air3_mode2_1609_unet3d_h2f_128_r0.yaml").read_text(encoding="utf-8")
    )
    assert config["data"]["manifest"].endswith("splits_mode2_1609_balanced200_trainval.json")
    assert config["data"]["mappings"] == ["even->full", "odd->full"]
    assert config["data"]["residuals"] == ["bhc", "marfi"]
    assert config["training"]["device"] == "cuda:0"
    assert config["training"]["run_dir"] == "runs/air3_mode2_1609_unet3d_h2f_128_r0"


def test_air3_r1b_config_supervises_the_blended_target_on_the_same_manifest():
    """r1b must differ from r0 only in the supervision target and the run dir."""
    root = ROOT / "configs"
    r0 = yaml.safe_load((root / "air3_mode2_1609_unet3d_h2f_128_r0.yaml").read_text(encoding="utf-8"))
    r1b = yaml.safe_load((root / "air3_mode2_1609_unet3d_h2f_128b_r1.yaml").read_text(encoding="utf-8"))

    assert r1b["data"]["manifest"] == r0["data"]["manifest"]
    assert r1b["data"]["manifest"].endswith("splits_mode2_1609_balanced200_trainval.json")
    assert r1b["data"]["mappings"] == ["even->full.dn_3d_128b_r1", "odd->full.dn_3d_128b_r1"]
    assert r1b["data"]["residuals"] == ["bhc", "marfi"]
    assert r1b["training"]["run_dir"] == "runs/air3_mode2_1609_unet3d_h2f_128b_r1"
    assert r1b["training"]["device"] == "cuda:0"

    # control variables: everything except the supervision target and the run dir
    for section in ("reconstruction", "model"):
        assert r1b[section] == r0[section], section
    for key, value in r0["data"].items():
        if key in ("mappings", "residuals"):
            continue
        assert r1b["data"].get(key) == value, f"data.{key}"
    for key, value in r0["training"].items():
        if key == "run_dir":
            continue
        assert r1b["training"].get(key) == value, f"training.{key}"


def test_final_plan_uses_five_epoch_early_stopping_patience():
    for name in FINAL_CHAIN_CONFIGS:
        config = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert config["training"]["patience"] == 5, name


def test_air3_recursive_rounds_only_change_the_target_and_the_run_dir():
    """r1b..r4b must stay a controlled experiment against AIR3 r0."""
    root = ROOT / "configs"
    r0 = yaml.safe_load((root / "air3_mode2_1609_unet3d_h2f_128_r0.yaml").read_text(encoding="utf-8"))

    for round_index in AIR3_RECURSIVE_ROUNDS:
        name = f"air3_mode2_1609_unet3d_h2f_128b_r{round_index}.yaml"
        cfg = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        suffix = f"full.dn_3d_128b_r{round_index}"

        assert cfg["data"]["mappings"] == [f"even->{suffix}", f"odd->{suffix}"], name
        assert cfg["data"]["residuals"] == ["bhc", "marfi"], name
        assert cfg["data"]["manifest"] == r0["data"]["manifest"], name
        assert cfg["training"]["run_dir"] == f"runs/air3_mode2_1609_unet3d_h2f_128b_r{round_index}", name
        assert cfg["training"]["device"] == "cuda:0", name

        for section in ("reconstruction", "model"):
            assert cfg[section] == r0[section], f"{name}: {section}"
        for key, value in r0["data"].items():
            if key in ("mappings", "residuals"):
                continue
            assert cfg["data"].get(key) == value, f"{name}: data.{key}"
        for key, value in r0["training"].items():
            if key == "run_dir":
                continue
            assert cfg["training"].get(key) == value, f"{name}: training.{key}"
