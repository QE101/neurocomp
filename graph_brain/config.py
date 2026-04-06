"""Configuration system using Pydantic v2 for validation + OmegaConf for YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class SpatialConfig(BaseModel):
    """3D spatial layout for node positions."""
    dimensions: int = 3
    volume_size: list[float] = Field(default=[1.0, 1.0, 1.0])


class ConnectivityTypeConfig(BaseModel):
    """Distance-dependent connectivity parameters for one edge type."""
    p_max: float = Field(ge=0.0, le=1.0, description="Max connection probability")
    sigma: float = Field(gt=0.0, description="Gaussian decay width (normalized coords)")
    source_types: list[str] = Field(description="Allowed source node types")
    target_types: list[str] = Field(description="Allowed target node types")
    plastic: bool = True
    constant_k: Optional[int] = Field(default=None, ge=1,
        description="Target constant degree per node. If set, overrides p_max to achieve this degree regardless of N.")


class ConnectivityConfig(BaseModel):
    """Per-edge-type connectivity parameters."""
    driving: ConnectivityTypeConfig = ConnectivityTypeConfig(
        p_max=0.3, sigma=0.15,
        source_types=["EXCITATORY"], target_types=["EXCITATORY"],
    )
    modulatory: ConnectivityTypeConfig = ConnectivityTypeConfig(
        p_max=0.2, sigma=0.25,
        source_types=["EXCITATORY"], target_types=["EXCITATORY"],
    )
    inhib_perisomatic: ConnectivityTypeConfig = ConnectivityTypeConfig(
        p_max=0.5, sigma=0.10,
        source_types=["PV"], target_types=["EXCITATORY"],
    )
    inhib_dendritic: ConnectivityTypeConfig = ConnectivityTypeConfig(
        p_max=0.4, sigma=0.12,
        source_types=["SST"], target_types=["EXCITATORY", "VIP"],
    )
    disinhibition: ConnectivityTypeConfig = ConnectivityTypeConfig(
        p_max=0.4, sigma=0.10,
        source_types=["VIP"], target_types=["SST"],
    )
    electrical: ConnectivityTypeConfig = ConnectivityTypeConfig(
        p_max=0.3, sigma=0.05,
        source_types=["PV"], target_types=["PV"],
        plastic=False,
    )
    retrograde: ConnectivityTypeConfig = ConnectivityTypeConfig(
        p_max=0.1, sigma=0.15,
        source_types=["EXCITATORY"], target_types=["EXCITATORY"],
    )
    max_radius: float = Field(default=0.5, gt=0.0, description="Hard distance cutoff")


class NodeConfig(BaseModel):
    """Node counts and two-compartment model parameters."""
    n_excitatory: int = Field(default=4000, ge=1)
    n_pv: int = Field(default=350, ge=0)
    n_sst: int = Field(default=350, ge=0)
    n_vip: int = Field(default=300, ge=0)

    spatial: SpatialConfig = SpatialConfig()

    # Two-compartment dynamics
    basal_tau: float = Field(default=10.0, gt=0.0, description="Basal membrane time constant (ms)")
    apical_tau: float = Field(default=20.0, gt=0.0, description="Apical time constant (ms)")
    basal_activation: str = Field(default="softplus", pattern="^(softplus|relu)$")
    apical_center: float = Field(default=1.0, description="Sigmoid center for gating (g(0)=1 when center=1)")
    apical_slope: float = Field(default=1.0, gt=0.0, description="Sigmoid steepness")
    noise_std: float = Field(default=0.01, ge=0.0)
    dt: float = Field(default=1.0, gt=0.0, description="Simulation timestep (ms)")

    # Intrinsic plasticity
    ip_enabled: bool = True
    ip_learning_rate: float = Field(default=1e-4, ge=0.0)
    ip_target_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    ip_tau: float = Field(default=1000.0, gt=0.0, description="Activity EMA time constant (ms)")

    @property
    def n_total(self) -> int:
        return self.n_excitatory + self.n_pv + self.n_sst + self.n_vip


class STDPConfig(BaseModel):
    """STDP learning rule parameters (Bi & Poo 1998)."""
    enabled: bool = True
    learning_rate: float = Field(default=0.01, ge=0.0)
    tau_plus: float = Field(default=20.0, gt=0.0, description="LTP time constant (ms)")
    tau_minus: float = Field(default=20.0, gt=0.0, description="LTD time constant (ms)")
    a_plus: float = Field(default=0.005, ge=0.0, description="LTP magnitude")
    a_minus: float = Field(default=0.00525, ge=0.0, description="LTD magnitude (slight bias)")
    w_min: float = Field(default=0.0, description="Minimum weight")
    w_max: float = Field(default=1.0, description="Maximum weight")
    update_interval: int = Field(default=1, ge=1, description="Update every N steps")


class HomeostaticConfig(BaseModel):
    """Homeostatic synaptic scaling (Turrigiano 2008)."""
    enabled: bool = True
    tau: float = Field(default=10000.0, gt=0.0, description="Scaling time constant (ms)")
    target_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    update_interval: int = Field(default=100, ge=1)


class STPConfig(BaseModel):
    """Short-term plasticity: Tsodyks-Markram model."""
    enabled: bool = True
    tau_facilitation: float = Field(default=500.0, gt=0.0, description="Facilitation recovery (ms)")
    tau_depression: float = Field(default=200.0, gt=0.0, description="Depression recovery (ms)")
    U_baseline: float = Field(default=0.2, gt=0.0, le=1.0, description="Baseline release probability")


class StructuralConfig(BaseModel):
    """Homeostatic structural plasticity: activity-driven edge creation/pruning."""
    enabled: bool = True
    update_interval: int = Field(default=500, ge=1, description="Steps between structural updates")
    # Growth: quiet nodes sprout new connections
    growth_rate: float = Field(default=0.1, ge=0.0, le=1.0,
                               description="Fraction of deficit edges to grow per update")
    # Pruning: weak/unused edges die
    prune_threshold: float = Field(default=0.01, ge=0.0,
                                   description="Edges with weight below this are pruned")
    # Energy cost per edge (metabolic constraint)
    edge_cost: float = Field(default=1e-5, ge=0.0,
                             description="Per-edge energy penalty — prevents runaway growth")
    # Max edges per node (memory safety)
    max_degree: int = Field(default=5000, ge=1,
                            description="Hard cap on incoming edges per node")


class EdgeConfig(BaseModel):
    """Edge-level configuration: connectivity + learning rules."""
    connectivity: ConnectivityConfig = ConnectivityConfig()
    stdp: STDPConfig = STDPConfig()
    homeostatic: HomeostaticConfig = HomeostaticConfig()
    stp: STPConfig = STPConfig()
    structural: StructuralConfig = StructuralConfig()


class SimulationConfig(BaseModel):
    """Simulation execution parameters."""
    seed: int = Field(default=42, ge=0)
    n_steps: int = Field(default=10000, ge=1)
    record_interval: int = Field(default=10, ge=1)
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
    log_interval: int = Field(default=100, ge=1)
    checkpoint_interval: int = Field(default=1000, ge=1)
    checkpoint_dir: str = "checkpoints"


class VizConfig(BaseModel):
    """Visualization parameters."""
    enabled: bool = True
    update_interval: int = Field(default=50, ge=1)
    node_color_field: str = "output"
    show_edges: bool = False
    max_edges_displayed: int = Field(default=5000, ge=0)


class HierarchyConfig(BaseModel):
    """Predictive coding hierarchy configuration."""
    enabled: bool = False
    n_levels: int = Field(default=2, ge=2, le=5)
    split_axis: int = Field(default=2, ge=0, le=2, description="Spatial axis for level split (0=x, 1=y, 2=z)")
    error_ratio: float = Field(default=0.4, gt=0.0, lt=1.0, description="Legacy: unused with universal error model")
    # Time-constant scaling (KEY for hierarchy)
    time_scale_factor: float = Field(default=3.0, gt=1.0, description="Tau multiplier per level. Level L has tau * factor^(L-1)")
    # PC dynamics
    pc_learning_rate: float = Field(default=0.05, gt=0.0, description="Representation update rate from error signals")
    precision_base: float = Field(default=1.0, gt=0.0, description="Base precision weighting for error signals")
    # Inter-level wiring
    inter_level_k: int = Field(default=15, ge=1, description="Target inter-level connections per source node")
    inter_level_sigma: float = Field(default=0.5, gt=0.0, description="Distance decay for inter-level connections")
    inter_level_init_weight: float = Field(default=0.15, gt=0.0, description="Initial weight for inter-level edges")
    # Input
    pattern_duration: int = Field(default=50, ge=1, description="Steps each pattern is presented")
    input_strength: float = Field(default=2.0, gt=0.0, description="Amplitude of sensory input injection")


class HippocampalConfig(BaseModel):
    """Hippocampal fast encoding + sleep replay consolidation."""
    enabled: bool = False
    n_dg: int = Field(default=2000, ge=100, description="Dentate gyrus population size")
    n_ca3: int = Field(default=500, ge=50, description="CA3 population size")
    dg_sparsity: float = Field(default=0.02, gt=0.0, lt=1.0, description="Fraction of DG nodes active per pattern")
    dg_fan_in: int = Field(default=2000, ge=10, description="Cortical inputs per DG node (sparse projection)")
    ca3_sparsity: float = Field(default=0.10, gt=0.0, lt=1.0, description="Fraction of CA3 active per pattern")
    encoding_lr: float = Field(default=0.5, gt=0.0, description="One-shot Hebbian learning rate")
    replay_strength: float = Field(default=0.3, gt=0.0, le=1.0, description="Replay injection strength")
    replay_lr_scale: float = Field(default=0.1, gt=0.0, le=1.0, description="Cortical lr multiplier during replay")
    max_patterns: int = Field(default=20, ge=1, description="Max stored patterns")
    replay_interleave: int = Field(default=5, ge=1, description="Replay cycles per sleep phase")
    replay_steps: int = Field(default=30, ge=1, description="Steps per pattern replay")


class GraphBrainConfig(BaseModel):
    """Top-level configuration for the neuromorphic graph."""
    nodes: NodeConfig = NodeConfig()
    edges: EdgeConfig = EdgeConfig()
    simulation: SimulationConfig = SimulationConfig()
    viz: VizConfig = VizConfig()
    hierarchy: HierarchyConfig = HierarchyConfig()
    hippocampal: HippocampalConfig = HippocampalConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> GraphBrainConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_dict(cls, data: dict) -> GraphBrainConfig:
        """Load config from dictionary."""
        return cls.model_validate(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def with_overrides(self, **kwargs) -> GraphBrainConfig:
        """Return a new config with dotted-path overrides applied.

        Example: config.with_overrides(**{"nodes.n_excitatory": 100})
        """
        data = self.model_dump()
        for key, value in kwargs.items():
            parts = key.split(".")
            d = data
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = value
        return GraphBrainConfig.model_validate(data)


def resolve_device(device: str) -> str:
    """Resolve 'auto' to actual device string."""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
