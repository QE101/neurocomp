"""Shared test fixtures for graph_brain tests."""

import pytest
import torch

from graph_brain.config import GraphBrainConfig


@pytest.fixture
def small_config() -> GraphBrainConfig:
    """Tiny config for fast unit tests (N=100)."""
    return GraphBrainConfig.from_yaml("configs/small_test.yaml")


@pytest.fixture
def default_config() -> GraphBrainConfig:
    """Default config (N=5000). For slower integration tests."""
    return GraphBrainConfig.from_yaml("configs/default.yaml")


@pytest.fixture
def cpu_config() -> GraphBrainConfig:
    """Small config forced to CPU."""
    return GraphBrainConfig(
        nodes=GraphBrainConfig().nodes.model_copy(update={
            "n_excitatory": 80,
            "n_pv": 7,
            "n_sst": 7,
            "n_vip": 6,
        }),
        simulation=GraphBrainConfig().simulation.model_copy(update={
            "device": "cpu",
            "seed": 42,
        }),
        viz=GraphBrainConfig().viz.model_copy(update={"enabled": False}),
    )


@pytest.fixture
def small_graph(cpu_config):
    """Initialized graph with N=100 on CPU."""
    from graph_brain.core.graph import NeuromorphicGraph
    graph = NeuromorphicGraph(cpu_config)
    graph.initialize()
    return graph


@pytest.fixture
def device():
    """Return available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"
