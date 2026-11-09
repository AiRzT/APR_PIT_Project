from __future__ import annotations

import copy
import io
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apr_pit.config import load_config, smoke_test_config  # noqa: E402
from apr_pit.model import APRPiT  # noqa: E402
from apr_pit.physics import TunnelFirePhysics  # noqa: E402
from apr_pit.sampling import (  # noqa: E402
    APRController,
    CollocationPool,
    coordinate_dimensions,
    sample_boundary,
    sample_initial,
)


def small_config(dimensions: int = 2, inlet_mode: str = "velocity") -> dict:
    config = smoke_test_config(
        load_config(PROJECT_ROOT / "configs" / f"tunnel_{dimensions}d.yaml")
    )
    config["model"].update(
        velocity_conditioning=True,
        spatial_dimensions=dimensions,
        input_dim=dimensions + 2,
        transformer_layers=1,
        d_model=16,
        ffn_width=24,
        mlp_width=16,
        fourier_frequencies=4,
        modulation_width=12,
    )
    config["boundary"] = {
        "inlet_mode": inlet_mode,
        "inlet_velocity": 0.65 if inlet_mode == "velocity" else 0.0,
    }
    return config


def uniform_fields(coordinates: torch.Tensor, config: dict, velocity: float) -> dict:
    # Retain the coordinate graph so the boundary operator can differentiate.
    zero = coordinates[:, 0:1] * 0.0
    fields = {
        "u": zero + velocity,
        "w": zero,
        "pressure": zero,
        "temperature": zero + float(config["physics"]["ambient_temperature"]),
        "smoke": zero,
    }
    if config["model"]["spatial_dimensions"] == 3:
        fields["v"] = zero
    return fields


class VelocityConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1234)

    def test_inlet_velocity_is_exact_in_2d_and_3d(self) -> None:
        for dimensions in (2, 3):
            config = small_config(dimensions)
            model = APRPiT(config).eval()
            coordinates = torch.full((3, dimensions + 1), 0.4)
            coordinates[:, 0] = torch.tensor([0.01, 0.5, 1.0])
            coordinates[:, 1] = 0.0
            for velocity in (0.0, 0.3, 0.65, 1.8):
                with self.subTest(dimensions=dimensions, velocity=velocity):
                    prediction = model(coordinates, inlet_velocity=velocity)
                    torch.testing.assert_close(
                        prediction["u"], torch.full((3, 1), velocity), rtol=0, atol=1e-7
                    )
                    for name in (("w",) if dimensions == 2 else ("v", "w")):
                        torch.testing.assert_close(
                            prediction[name], torch.zeros(3, 1), rtol=0, atol=0
                        )
            default_prediction = model(coordinates)
            torch.testing.assert_close(default_prediction["u"], torch.full((3, 1), 0.65))

    def test_velocity_conditions_temperature_without_updating_weights(self) -> None:
        model = APRPiT(small_config()).eval()
        coordinates = torch.tensor([[0.2, 0.25, 0.3], [0.7, 0.6, 0.8]])
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        low = model(coordinates, inlet_velocity=0.3)
        high = model(coordinates, inlet_velocity=1.8)
        self.assertGreater(float((low["temperature"] - high["temperature"]).abs().max().detach()), 1e-5)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_lifting_first_and_second_derivatives_include_product_rule(self) -> None:
        config = small_config()
        model = APRPiT(config)
        coordinates = torch.tensor([[0.2, 0.25, 0.3], [0.7, 0.6, 0.8]], requires_grad=True)

        def analytic_decoder(_module, _inputs, output):
            x = coordinates[:, 1:2]
            return torch.cat((2.0 + 3.0 * x, output[:, 1:] * 0.0), dim=1)

        with model.decoder.register_forward_hook(analytic_decoder):
            prediction = model(coordinates, inlet_velocity=0.3)
        velocity_scale = float(config["scales"]["velocity"])
        x = coordinates[:, 1:2]
        torch.testing.assert_close(prediction["u"], 0.3 + velocity_scale * x * (2 + 3 * x))
        first = torch.autograd.grad(prediction["u"].sum(), coordinates, create_graph=True)[0]
        second = torch.autograd.grad(first[:, 1].sum(), coordinates)[0]
        torch.testing.assert_close(first[:, 1:2], velocity_scale * (2 + 6 * x))
        torch.testing.assert_close(second[:, 1:2], torch.full((2, 1), 6 * velocity_scale))

    def test_conditioned_pde_uses_only_space_time_coordinates(self) -> None:
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions):
                config = small_config(dimensions)
                model = APRPiT(config)
                physics = TunnelFirePhysics(config)
                coordinates = (0.2 + 0.6 * torch.rand(3, dimensions + 1)).requires_grad_(True)
                residual = physics.residuals(model(coordinates, inlet_velocity=0.3), coordinates)
                self.assertEqual(residual["vector"].shape, (3, dimensions + 3))
                self.assertTrue(torch.isfinite(residual["vector"]).all())
                loss = physics.pde_loss(residual)
                loss.backward()
                self.assertEqual(coordinates.grad.shape, coordinates.shape)
                self.assertTrue(torch.isfinite(coordinates.grad).all())
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_attention_geometry_has_no_velocity_axis(self) -> None:
        for dimensions in (2, 3):
            for sparse_switch in (1, 64):
                with self.subTest(dimensions=dimensions, sparse_switch=sparse_switch):
                    config = small_config(dimensions)
                    config["model"]["sparse_switch"] = sparse_switch
                    model = APRPiT(config)
                    coordinates = torch.rand(4, dimensions + 1)
                    seen = []

                    def capture_coordinates(_module, args):
                        seen.append(args[1].detach().clone())

                    with model.blocks[0].attention.register_forward_pre_hook(capture_coordinates):
                        model(coordinates, inlet_velocity=1.8)
                    self.assertEqual(len(seen), 1)
                    torch.testing.assert_close(seen[0], coordinates)

    def test_boundary_target_is_requested_nonzero_velocity(self) -> None:
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions):
                config = small_config(dimensions)
                physics = TunnelFirePhysics(config)
                coordinates = torch.full((4, dimensions + 1), 0.4)
                coordinates[:, 1] = 0.0
                coordinates.requires_grad_(True)
                tags = torch.zeros(4, dtype=torch.long)
                correct = physics.boundary_loss(
                    uniform_fields(coordinates, config, 1.8), coordinates, tags, inlet_velocity=1.8
                )
                wrong = physics.boundary_loss(
                    uniform_fields(coordinates, config, 0.0), coordinates, tags, inlet_velocity=1.8
                )
                self.assertAlmostEqual(float(correct.detach()), 0.0, places=7)
                self.assertAlmostEqual(
                    float(wrong.detach()), (1.8 / float(config["scales"]["velocity"])) ** 2, places=6
                )

    def test_open_inlet_allows_nonzero_uniform_flow(self) -> None:
        config = small_config(inlet_mode="open")
        physics = TunnelFirePhysics(config)
        coordinates = torch.tensor([[0.2, 0.0, 0.3], [0.7, 0.0, 0.8]], requires_grad=True)
        tags = torch.zeros(2, dtype=torch.long)
        fields = uniform_fields(coordinates, config, 1.8)
        self.assertAlmostEqual(float(physics.boundary_loss(fields, coordinates, tags).detach()), 0.0, places=7)
        fields["pressure"] = fields["pressure"] + 1.0
        self.assertGreater(float(physics.boundary_loss(fields, coordinates, tags).detach()), 0.0)

    def test_open_mode_does_not_apply_velocity_lifting(self) -> None:
        config = small_config(inlet_mode="open")
        model = APRPiT(config)
        with torch.no_grad():
            model.decoder[-1].weight.zero_()
            model.decoder[-1].bias.fill_(2.0)
        coordinates = torch.tensor([[0.2, 0.0, 0.3], [0.7, 0.6, 0.8]])
        prediction = model(coordinates)
        expected = torch.full((2, 1), 2.0 * float(config["scales"]["velocity"]))
        torch.testing.assert_close(prediction["u"], expected)
        torch.testing.assert_close(prediction["w"], expected)
        with self.assertRaises(ValueError):
            model(coordinates, inlet_velocity=0.3)

    def test_legacy_checkpoint_loads_strictly_and_rejects_velocity_override(self) -> None:
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions):
                config = small_config(dimensions)
                config["model"].pop("velocity_conditioning")
                config["model"]["input_dim"] = dimensions + 1
                config.pop("boundary")
                original = APRPiT(config).eval()
                coordinates = torch.rand(3, dimensions + 1)
                expected = original(coordinates)
                stream = io.BytesIO()
                torch.save({"config": copy.deepcopy(config), "model": original.state_dict()}, stream)
                stream.seek(0)
                checkpoint = torch.load(stream, weights_only=False)
                restored = APRPiT(checkpoint["config"]).eval()
                restored.load_state_dict(checkpoint["model"], strict=True)
                actual = restored(coordinates)
                for name in original.field_names:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    restored(coordinates, inlet_velocity=0.3)

    def test_apr_and_boundary_samples_keep_space_time_dimension(self) -> None:
        for dimensions in (2, 3):
            with self.subTest(dimensions=dimensions):
                config = small_config(dimensions)
                coordinate_count = coordinate_dimensions(config)
                self.assertEqual(coordinate_count, dimensions + 1)
                generator = torch.Generator().manual_seed(42)
                pool = CollocationPool.initialize(24, generator, dimensions=coordinate_count)
                pool.residual_scores = torch.linspace(0.1, 1.0, 24).reshape(-1, 1)
                APRController(config).refine(pool, generator)
                self.assertEqual(pool.points.shape[1], coordinate_count)
                points, tags = sample_boundary(24, generator, dimensions=coordinate_count)
                self.assertEqual(points.shape, (24, coordinate_count))
                self.assertEqual(set(tags.tolist()), set(range(2 * dimensions)))
                self.assertTrue((points[:, 0] > 0).all())
                initial = sample_initial(24, generator, dimensions=coordinate_count)
                self.assertTrue((initial[:, 0] == 0).all())
                self.assertTrue(((initial[:, 1:] > 0) & (initial[:, 1:] < 1)).all())


if __name__ == "__main__":
    unittest.main()
