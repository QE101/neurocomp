"""Tests for configuration system."""

import pytest
from graph_brain.config import GraphBrainConfig


class TestConfigLoading:
    def test_default_yaml_loads(self):
        cfg = GraphBrainConfig.from_yaml("configs/default.yaml")
        assert cfg.nodes.n_total == 5000
        assert cfg.nodes.n_excitatory == 4000

    def test_small_test_yaml_loads(self):
        cfg = GraphBrainConfig.from_yaml("configs/small_test.yaml")
        assert cfg.nodes.n_total == 100
        assert cfg.simulation.device == "cpu"

    def test_default_constructor(self):
        cfg = GraphBrainConfig()
        assert cfg.nodes.n_total == 5000
        assert cfg.edges.stdp.enabled is True

    def test_from_dict(self):
        cfg = GraphBrainConfig.from_dict({"nodes": {"n_excitatory": 200}})
        assert cfg.nodes.n_excitatory == 200
        assert cfg.nodes.n_pv == 350  # default preserved


class TestConfigValidation:
    def test_negative_nodes_rejected(self):
        with pytest.raises(Exception):
            GraphBrainConfig.from_dict({"nodes": {"n_excitatory": -1}})

    def test_invalid_activation_rejected(self):
        with pytest.raises(Exception):
            GraphBrainConfig.from_dict({"nodes": {"basal_activation": "tanh"}})

    def test_invalid_device_rejected(self):
        with pytest.raises(Exception):
            GraphBrainConfig.from_dict({"simulation": {"device": "tpu"}})

    def test_stdp_bounds(self):
        cfg = GraphBrainConfig()
        assert cfg.edges.stdp.w_min < cfg.edges.stdp.w_max


class TestConfigOverrides:
    def test_with_overrides(self):
        cfg = GraphBrainConfig()
        cfg2 = cfg.with_overrides(**{"nodes.n_excitatory": 100})
        assert cfg2.nodes.n_excitatory == 100
        assert cfg.nodes.n_excitatory == 4000  # original unchanged

    def test_nested_override(self):
        cfg = GraphBrainConfig()
        cfg2 = cfg.with_overrides(**{"edges.stdp.learning_rate": 0.05})
        assert cfg2.edges.stdp.learning_rate == 0.05


class TestConfigSerialization:
    def test_roundtrip_yaml(self, tmp_path):
        cfg = GraphBrainConfig()
        path = tmp_path / "test_config.yaml"
        cfg.save_yaml(path)
        cfg2 = GraphBrainConfig.from_yaml(path)
        assert cfg2.nodes.n_total == cfg.nodes.n_total
        assert cfg2.edges.stdp.tau_plus == cfg.edges.stdp.tau_plus
