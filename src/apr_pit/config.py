from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


def spatial_dimensions(config: dict[str, Any]) -> int:
    """Separate physical dimensions from the optional case-level V* input."""
    model = config["model"]
    conditioned = bool(model.get("velocity_conditioning", False))
    dimensions = int(model.get("spatial_dimensions", int(model["input_dim"]) - 1 - conditioned))
    if dimensions not in (2, 3):
        raise ValueError("model.spatial_dimensions must be 2 or 3")
    return dimensions


def resolve_inlet_velocity(config: dict[str, Any], override: float | None = None) -> float:
    """Resolve one physical inlet speed (m/s) for a whole attention batch.

    Legacy checkpoints keep their original unconditioned, zero-inlet behavior.
    Changing wind speed requires a conditioned, prescribed-inlet checkpoint;
    evaluation cannot silently change an open-portal boundary topology.
    """
    boundary = config.get("boundary", {})
    mode = boundary.get("inlet_mode", "velocity")
    if mode not in ("velocity", "open"):
        raise ValueError("boundary.inlet_mode must be 'velocity' or 'open'")
    conditioned = bool(config["model"].get("velocity_conditioning", False))
    if override is not None and not conditioned:
        raise ValueError("inlet_velocity override requires a velocity-conditioned checkpoint; retrain first")
    value = float(boundary.get("inlet_velocity", 0.0) if override is None else override)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("inlet_velocity must be a finite, nonnegative speed in m/s")
    if mode == "open" and value != 0.0:
        raise ValueError("An open-portal checkpoint uses V*=0; train a velocity-inlet model first")
    if not conditioned and value != 0.0:
        raise ValueError("Nonzero inlet_velocity requires model.velocity_conditioning: true")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and perform lightweight consistency checks."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    required = {"seed", "domain", "scales", "physics", "source", "model", "sampling", "training"}
    missing = required.difference(config)
    if missing:
        raise KeyError(f"Missing configuration sections: {sorted(missing)}")
    if config["model"]["d_model"] % config["model"]["attention_heads"] != 0:
        raise ValueError("model.d_model must be divisible by model.attention_heads")
    if config["model"]["correction_heads"] > config["model"]["attention_heads"]:
        raise ValueError("correction_heads cannot exceed attention_heads")
    dimensions = spatial_dimensions(config)
    expected = {2: (3, 5), 3: (4, 6)}
    expected_input, expected_output = expected[dimensions]
    expected_input += bool(config["model"].get("velocity_conditioning", False))
    if int(config["model"]["input_dim"]) != expected_input:
        raise ValueError(
            f"model.input_dim must be {expected_input} for {dimensions}D physics and its conditioning flag"
        )
    if int(config["model"]["output_dim"]) != expected_output:
        raise ValueError(
            f"model.output_dim must be {expected_output} for {dimensions}D physics"
        )
    velocity_scale = float(config["scales"]["velocity"])
    if not math.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise ValueError("scales.velocity must be finite and positive")
    resolve_inlet_velocity(config)
    return config


def smoke_test_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a tiny configuration that exercises the complete algorithm on CPU/MPS."""
    cfg = copy.deepcopy(config)
    cfg["seed"] = 7
    cfg["model"].update(
        {
            "transformer_layers": 2,
            "d_model": 64,
            "attention_heads": 4,
            "correction_heads": 1,
            "ffn_width": 128,
            "mlp_layers": 2,
            "mlp_width": 64,
            "fourier_frequencies": 8,
            "local_window": 8,
            "global_anchors": 4,
            "sparse_switch": 64,
            "modulation_layers": 1,
            "modulation_width": 32,
        }
    )
    cfg["sampling"].update(
        {
            "initial_interior": 96,
            "initial_boundary": 48,
            "initial_initial": 48,
            "max_points": 160,
            "adapt_interval": 1,
            "evaluation_points": 48,
        }
    )
    cfg["training"].update(
        {
            "adam_epochs": 2,
            "pde_batch": 12,
            "bc_batch": 8,
            "ic_batch": 8,
            "log_interval": 1,
            "checkpoint_interval": 2,
        }
    )
    cfg["training"]["lbfgs"].update({"iterations": 1, "history_size": 5, "inner_iterations": 1})
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
